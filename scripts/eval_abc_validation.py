#!/usr/bin/env python
"""Evaluate the published ABC-DiT checkpoint on the task-filtered LeRobot validation split.

Uses ABC's absolute actions, checkpoint mean/std statistics, CLIP prompt, image
preprocessing, 30-step horizon, and training flow loss (reference prefix settings).
Action reconstruction uses no prefix. Images and states use evaluation preprocessing.
"""

import argparse
import dataclasses
import hashlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from eval_validation_common import (
    DEFAULT_TASK_NAME,
    select_episodes,
    summarize_action_errors,
    task_name,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# Install this checkout's LeRobot 0.5.1 compatibility hooks for ABC-130K's
# language columns. This changes dataset reading only; ABC uses its own transforms.
from openpi.training import data_loader as _lerobot_compat  # noqa: F401

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "abc"))
from abc_minimal.config import ClipConfig, DiTConfig, FlowConfig  # noqa: E402
from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained  # noqa: E402
from abc_minimal.preprocess import normalize, parse_norm_stats, resize_pad_normalize  # noqa: E402


class ABCValidationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, stats):
        self.dataset = dataset
        self.stats = stats

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        images = {
            target: resize_pad_normalize(row[f"observation.images.{source}"])
            for target, source in (("top", "top"), ("left", "left_wrist"), ("right", "right_wrist"))
        }
        return {
            "state": normalize(row["observation.state"].numpy(), self.stats["state"]).astype(
                np.float32
            ),
            "actions": normalize(row["action"].numpy(), self.stats["actions"]).astype(np.float32),
            "images": images,
            "state_is_masked": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE / "abc/cache/bottles_75k.pt")
    parser.add_argument("--task-name", type=task_name, default=DEFAULT_TASK_NAME)
    parser.add_argument("--val-repo-id", default="abc_130k_v3_val")
    parser.add_argument(
        "--val-root",
        default="/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_val",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    for name in ("batch_size", "num_denoising_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    meta = LeRobotDatasetMetadata(args.val_repo_id, root=args.val_root)
    episodes, total_frames = select_episodes(meta, args.task_name)
    logging.info("task=%r episodes=%d frames=%d", args.task_name, len(episodes), total_frames)
    # mmap avoids reading the 8 GB weight tensors during the CPU-only inspection.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if "norm_stats" not in checkpoint:
        raise ValueError("The ABC checkpoint must contain its training normalization statistics")
    stats = parse_norm_stats(checkpoint["norm_stats"])
    stats_json = json.dumps(
        {key: {k: v.tolist() for k, v in values.items()} for key, values in stats.items()},
        sort_keys=True,
    )
    config = DiTConfig()
    state_dict = checkpoint["model"]
    if tuple(state_dict["pos_embed"].shape) != (1, config.chunk_length, config.hidden_size):
        raise ValueError("Checkpoint architecture differs from the published ABC DiTConfig")
    step = checkpoint.get("step")
    logging.info(
        "ABC checkpoint step=%s horizon=%d; using embedded mean/std statistics",
        step,
        config.chunk_length,
    )
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Run model evaluation in a GPU allocation")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = DiTPolicy(config)
    load_pretrained(model, args.checkpoint)
    del checkpoint, state_dict
    model.to(device).eval()
    model.img_backbone.set_bfloat16(True)  # Same DINO autocast setting as ABC training.
    clip_config = ClipConfig(cache_dir=str(WORKSPACE / "abc/cache/clip"))
    for name in (clip_config.model_name, clip_config.bpe_name):
        if not (Path(clip_config.cache_dir) / name).is_file():
            raise FileNotFoundError(Path(clip_config.cache_dir) / name)
    embedder = CLIPTextEmbedder(clip_config, device=device)
    task_vec = embedder.encode([args.task_name]).to(device)
    del embedder
    flow = FlowConfig()
    dataset = ABCValidationDataset(
        LeRobotDataset(
            args.val_repo_id,
            root=args.val_root,
            episodes=episodes,
            tolerance_s=1e-3,
            delta_timestamps={"action": [i / meta.fps for i in range(config.chunk_length)]},
        ),
        stats,
    )
    batches = len(dataset) // args.batch_size
    if args.max_batches is not None:
        batches = min(batches, args.max_batches)
    if batches == 0:
        raise ValueError("No full validation batches")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    shape = (config.chunk_length, config.action_dim)
    se_norm, ae_norm, se_raw = (np.zeros(shape, dtype=np.float64) for _ in range(3))
    scale = torch.as_tensor(stats["actions"]["std"] + 1e-6, device=device)
    frames, loss_sum = 0, 0.0
    start = time.monotonic()
    with torch.no_grad():
        for i, batch in enumerate(itertools.islice(loader, batches)):
            batch = {
                key: {cam: x.to(device, non_blocking=True) for cam, x in value.items()}
                if isinstance(value, dict)
                else value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            batch["task_vec_clip"] = task_vec.expand(args.batch_size, -1)
            # Sampling has its own generator; adding flow loss cannot change predictions.
            sample_generator = torch.Generator(device=device).manual_seed(args.seed + i)
            noise = torch.randn(batch["actions"].shape, device=device, generator=sample_generator)
            pred = model.sample_actions(batch, num_steps=args.num_denoising_steps, noise=noise)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.seed + 0x56414C + i)
                loss = model(
                    batch,
                    max_action_prefix=flow.max_action_prefix,
                    prefix_conditioning_prob=flow.prefix_conditioning_prob,
                    prefix_noise_scale=flow.prefix_noise_scale,
                )
            # Training logs average these per-batch objectives. All batches have the same size.
            loss_sum += float(loss)
            diff = (pred - batch["actions"]).float()
            se_norm += diff.square().sum(0).cpu().numpy()
            ae_norm += diff.abs().sum(0).cpu().numpy()
            se_raw += (diff * scale).square().sum(0).cpu().numpy()
            frames += diff.shape[0]
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == batches:
                logging.info(
                    "batch %d/%d frames=%d mse_norm=%.5f val_flow_loss=%.5f elapsed=%.0fs",
                    i + 1,
                    batches,
                    frames,
                    se_norm.sum() / (frames * np.prod(shape)),
                    loss_sum / (i + 1),
                    time.monotonic() - start,
                )

    metrics = summarize_action_errors(se_norm, ae_norm, se_raw, frames)
    metrics["val_flow_loss"] = loss_sum / batches
    result = {
        "checkpoint_dir": str(args.checkpoint.resolve()),
        "step": step,
        "model_family": "abc_dit",
        "task_name": args.task_name,
        "val_repo_id": args.val_repo_id,
        "val_root": args.val_root,
        "val_episodes": episodes,
        "val_frames_total": total_frames,
        "frames_evaluated": frames,
        "batch_size": args.batch_size,
        "num_batches": batches,
        "seed": args.seed,
        "num_denoising_steps": args.num_denoising_steps,
        "action_horizon": config.chunk_length,
        "action_dim": config.action_dim,
        "model_action_dim": config.action_dim,
        "delta_joint_actions": False,
        "quantile_norm": False,
        "normalization": "mean_std",
        "norm_stats_path": str(args.checkpoint.resolve()) + "#norm_stats",
        "norm_stats_sha256": hashlib.sha256(stats_json.encode()).hexdigest(),
        "compute_val_loss": True,
        "flow_loss_train_mode": False,
        "flow_loss_definition": "mean of per-batch ABC training flow objectives, with random action prefixes",
        "flow_time_distribution": "Uniform(0, 1)",
        "flow_loss_draws_per_frame": 1,
        "flow_config": dataclasses.asdict(flow),
        "flow_config_source": "ABC reference training defaults; checkpoint has no training config",
        "state_masking": False,
        "image_augmentation": False,
        "sampling_action_prefix_length": 0,
        "parameter_dtype": "float32 (DINO uses bfloat16 autocast, as in training)",
        "comparison_horizon": 30,
        "drop_last": True,
        "episode_end_padding": "repeat last action; included in metrics",
        "metrics": metrics,
        "wall_time_s": time.monotonic() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    temp.replace(args.output)
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
