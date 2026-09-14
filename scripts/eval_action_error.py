#!/usr/bin/env python
"""Validation action error and training flow loss for a pi0.5 checkpoint.

Runs standalone -- no training loop.  Loads a saved OpenPI checkpoint, streams the
validation episodes of one task through the same input pipeline the run was trained
with, samples an action chunk for every frame, and scores it against the recorded
actions. Also computes model.compute_loss with train=False by default.

This mirrors the ``val_recon_error`` metric in ``abc/abc_minimal/train_loop.py``.
Two properties of pi0.5's action representation shape the metric definitions:

  * pi0.5 pads YAM's 14 action dims out to the model's 32 (``PadStatesAndActions``
    zero-fills the target tail). Main action metrics use the real dims; the
    separately named all-dimension MSE includes predicted padding. Flow loss uses
    all 32 dimensions to preserve the training objective.
  * ``LeRobotYamDataConfig.use_delta_joint_actions`` puts the arm joints in delta
    space; the grippers stay absolute.  Prediction and target share the same state,
    so the state cancels in their difference: the error in absolute joint space
    equals the error in delta space.  Undoing the quantile normalization is
    therefore enough to report radians, with no state term needed.

Norm stats default to the checkpoint's own ``assets/`` directory. An explicit
--norm-stats-path can select the statistics originally used by the training run.

Example:
    python scripts/eval_action_error.py \
        --checkpoint-dir .../pi05_abc130k_task/pi05_abc_put_..._17183582/4000 \
        --task-name 'put the plastic bottles in the bin'
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from eval_validation_common import (
    DEFAULT_TASK_NAME,
    file_sha256,
    resolve_checkpoint_stats,
    select_episodes,
    summarize_action_errors,
    task_name,
)
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding as _sharding

DEFAULT_BASE_CONFIG = "pi05_abc130k"
DEFAULT_VAL_REPO_ID = "abc_130k_v3_val"
DEFAULT_VAL_ROOT = "/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_val"
DEFAULT_OUTPUT_ROOT = (
    "/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/eval-action-error"
)

# Per arm: 6 joints then 1 gripper.  The gripper dims are normalized to [0, 1] in the
# dataset, so their error is not in radians and is reported separately.
DIMS_PER_ARM = 7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Checkpoint step directory, e.g. .../pi05_abc.../2000",
    )
    p.add_argument(
        "--task-name",
        type=task_name,
        default=DEFAULT_TASK_NAME,
        help="Exact task string or underscore slug",
    )
    p.add_argument(
        "--base-config", default=DEFAULT_BASE_CONFIG, help="OpenPI config the run was derived from"
    )
    p.add_argument("--val-repo-id", default=DEFAULT_VAL_REPO_ID)
    p.add_argument("--val-root", default=DEFAULT_VAL_ROOT)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap the number of batches (default: full split)",
    )
    p.add_argument(
        "--num-denoising-steps", type=int, default=10, help="Flow-matching steps in sample_actions"
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--seed", type=int, default=0, help="Fixes the sampling noise so runs are comparable"
    )
    p.add_argument("--fsdp-devices", type=int, default=1)
    p.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: derived under DEFAULT_OUTPUT_ROOT)",
    )
    p.add_argument(
        "--norm-stats-path",
        type=pathlib.Path,
        help="Optional training norm_stats.json; defaults to checkpoint assets",
    )
    p.add_argument(
        "--compute-val-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute the training flow-matching objective with train=False",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the episode selection and exit before loading weights",
    )
    return p.parse_args()


def patch_episode_filter(episode_ids: list[int]) -> None:
    """Restrict LeRobotDataset to `episode_ids`.

    This OpenPI checkout does not expose LeRobot's `episodes=` filter through its
    config, so the launcher for these runs injects it the same way.  Patching
    __init__ rather than replacing the class keeps dataset objects pickleable by
    spawned data-loader workers.
    """
    original_init = _data_loader.lerobot_dataset.LeRobotDataset.__init__

    def filtered_init(self, repo_id, root=None, episodes=None, *args, **kwargs):
        if episodes is not None:
            raise ValueError("Expected OpenPI not to set episodes")
        return original_init(self, repo_id, root, episode_ids, *args, **kwargs)

    _data_loader.lerobot_dataset.LeRobotDataset.__init__ = filtered_init


def build_eval_config(
    args: argparse.Namespace, checkpoint_dir: pathlib.Path
) -> _config.TrainConfig:
    """Take the training config and repoint it at the validation split.

    `assets_dir` is set to the checkpoint's own assets so `create_base_config` picks
    up exactly the norm stats this checkpoint was trained with.
    """
    base = _config.get_config(args.base_config)
    stats_path = args.norm_stats_path or resolve_checkpoint_stats(
        checkpoint_dir, base.data.assets.asset_id
    )
    stats_path = stats_path.resolve(strict=True)
    if stats_path.name != "norm_stats.json":
        raise ValueError("--norm-stats-path must point to a norm_stats.json file")
    args.resolved_norm_stats_path = stats_path
    data_factory = dataclasses.replace(
        base.data,
        repo_id=args.val_repo_id,
        lerobot_root=args.val_root,
        assets=dataclasses.replace(
            base.data.assets,
            assets_dir=str(stats_path.parent.parent),
            asset_id=stats_path.parent.name,
        ),
    )
    return dataclasses.replace(
        base, data=data_factory, batch_size=args.batch_size, num_workers=args.num_workers
    )


def unnormalize_scale(data_config: _config.DataConfig, action_dim: int) -> np.ndarray:
    """Per-dim factor converting a normalized action difference back to raw units.

    Both normalizations are affine, so a *difference* only needs the scale term.
    """
    stats = data_config.norm_stats["actions"]
    if data_config.use_quantile_norm:
        return (
            np.asarray(stats.q99)[:action_dim] - np.asarray(stats.q01)[:action_dim] + 1e-6
        ) / 2.0
    return np.asarray(stats.std)[:action_dim] + 1e-6


def main() -> None:
    # force=True: importing openpi/lerobot already installs a root handler, which would
    # otherwise make basicConfig a silent no-op and swallow every message below.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    args = parse_args()
    for name in ("batch_size", "num_denoising_steps", "fsdp_devices"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max_batches must be positive")

    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        raise FileNotFoundError(
            f"No params/ under {checkpoint_dir}; point --checkpoint-dir at a step directory."
        )

    config = build_eval_config(args, checkpoint_dir)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError(
            f"No norm stats for asset id {data_config.asset_id!r} under {checkpoint_dir / 'assets'}"
        )

    num_arms = config.data.num_arms
    action_dim = DIMS_PER_ARM * num_arms

    meta = LeRobotDatasetMetadata(args.val_repo_id, root=args.val_root)
    episode_ids, total_frames = select_episodes(meta, args.task_name)
    logging.info(
        f"val split: task={args.task_name!r} episodes={len(episode_ids)} frames={total_frames} "
        f"hours={total_frames / meta.fps / 3600:.2f}"
    )
    if args.dry_run:
        logging.info("Dry run: stopping before the model is loaded.")
        return

    patch_episode_filter(episode_ids)
    dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    dataset = _data_loader.transform_dataset(dataset, data_config)

    num_batches = len(dataset) // args.batch_size
    if num_batches == 0:
        raise ValueError(
            f"Batch size {args.batch_size} exceeds the {len(dataset)} available validation frames."
        )
    if args.max_batches is not None:
        num_batches = min(num_batches, args.max_batches)

    mesh = _sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS)
    )
    loader = _data_loader.DataLoaderImpl(
        data_config,
        _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=args.batch_size,
            sharding=data_sharding,
            shuffle=False,
            num_batches=num_batches,
            num_workers=args.num_workers,
            seed=args.seed,
        ),
    )

    logging.info(f"Loading model from {params_dir}")
    model = config.model.load(_model.restore_params(params_dir, dtype=jnp.bfloat16))
    model.eval()
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    # Omit the train kwarg when calling this wrapper: its default False stays a
    # Python constant during JIT, disabling image augmentation.
    compute_loss = nnx_utils.module_jit(model.compute_loss) if args.compute_val_loss else None

    scale = jnp.asarray(unnormalize_scale(data_config, action_dim), dtype=jnp.float32)
    horizon = config.model.action_horizon
    base_rng = jax.random.key(args.seed)

    # Accumulate sums rather than per-batch means so the totals are exact and the
    # per-horizon / per-dim breakdowns come out of the same pass.
    se_norm_hd = np.zeros((horizon, action_dim), dtype=np.float64)  # squared error, normalized
    ae_norm_hd = np.zeros((horizon, action_dim), dtype=np.float64)  # absolute error, normalized
    se_raw_hd = np.zeros((horizon, action_dim), dtype=np.float64)  # squared error, raw units
    # The literal F.mse_loss(pred, actions) of train_loop.py, over the full padded
    # tensor. Reported alongside the masked metric so the padding convention is
    # visible rather than assumed; see the module docstring.
    se_norm_padded = 0.0
    num_frames = 0
    loss_sum = 0.0
    loss_count = 0
    loss_base_rng = jax.random.fold_in(base_rng, 0x56414C)

    start = time.monotonic()
    after_first = None  # set once batch 1 lands, so the rate excludes JIT compilation
    for i, (observation, actions) in enumerate(loader):
        rng = jax.random.fold_in(base_rng, i)
        with _sharding.set_mesh(mesh):
            pred = sample_actions(rng, observation, num_steps=args.num_denoising_steps)
            if compute_loss is not None:
                chunk_loss = compute_loss(
                    jax.random.fold_in(loss_base_rng, i), observation, actions
                )
                loss_sum += float(jnp.sum(chunk_loss.astype(jnp.float32)))
                loss_count += chunk_loss.size
        full_diff = (pred - actions).astype(jnp.float32)
        se_norm_padded += float(jnp.sum(jnp.square(full_diff)))
        diff = full_diff[..., :action_dim]
        se_norm_hd += np.asarray(jnp.sum(jnp.square(diff), axis=0), dtype=np.float64)
        ae_norm_hd += np.asarray(jnp.sum(jnp.abs(diff), axis=0), dtype=np.float64)
        se_raw_hd += np.asarray(jnp.sum(jnp.square(diff * scale), axis=0), dtype=np.float64)
        num_frames += diff.shape[0]
        elapsed = time.monotonic() - start
        if i == 0:
            after_first = elapsed
            logging.info(f"batch 1/{num_batches} took {elapsed:.0f}s including JIT compilation")
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == num_batches:
            running = se_norm_hd.sum() / (num_frames * horizon * action_dim)
            # Batch 1 is all compilation, so it has no meaningful steady-state rate yet.
            rate = f"{(elapsed - after_first) / i:.2f}s/batch" if i else "compiling"
            loss_log = f"  val_flow_loss={loss_sum / loss_count:.5f}" if loss_count else ""
            logging.info(
                f"batch {i + 1}/{num_batches}  frames={num_frames}  mse_norm={running:.5f}{loss_log}  {rate}"
            )

    metrics = summarize_action_errors(
        se_norm_hd,
        ae_norm_hd,
        se_raw_hd,
        num_frames,
        padded_sum=se_norm_padded,
        model_dim=config.model.action_dim,
    )
    if loss_count:
        metrics["val_flow_loss"] = loss_sum / loss_count

    results = {
        "checkpoint_dir": str(checkpoint_dir),
        "step": checkpoint_dir.name,
        "base_config": args.base_config,
        "task_name": args.task_name,
        "val_repo_id": args.val_repo_id,
        "val_root": args.val_root,
        "val_episodes": episode_ids,
        "val_frames_total": total_frames,
        "frames_evaluated": num_frames,
        "batch_size": args.batch_size,
        "num_batches": num_batches,
        "num_denoising_steps": args.num_denoising_steps,
        "action_horizon": horizon,
        "action_dim": action_dim,
        "seed": args.seed,
        "quantile_norm": bool(data_config.use_quantile_norm),
        "delta_joint_actions": bool(config.data.use_delta_joint_actions),
        "model_family": "pi05",
        "normalization": "quantile" if data_config.use_quantile_norm else "mean_std",
        "norm_stats_path": str(args.resolved_norm_stats_path),
        "norm_stats_sha256": file_sha256(args.resolved_norm_stats_path),
        "compute_val_loss": args.compute_val_loss,
        "flow_loss_definition": f"mean squared velocity error over batch, horizon, all {config.model.action_dim} model dimensions",
        "flow_time_distribution": "0.001 + 0.999 * Beta(1.5, 1)",
        "flow_loss_train_mode": False,
        "flow_loss_draws_per_frame": 1,
        "model_action_dim": config.model.action_dim,
        "parameter_source": "checkpoint params (EMA when enabled during training)",
        "parameter_dtype": "bfloat16",
        "comparison_horizon": min(30, horizon),
        "drop_last": True,
        "episode_end_padding": "repeat last action; included in metrics",
        "metrics": metrics,
        "wall_time_s": time.monotonic() - start,
    }

    output = args.output
    if output is None:
        task_slug = "".join(c if c.isalnum() else "_" for c in args.task_name.lower()).strip("_")
        run_name = checkpoint_dir.parent.name
        output = f"{DEFAULT_OUTPUT_ROOT}/{run_name}_step{checkpoint_dir.name}_{task_slug}.json"
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_output.write_text(json.dumps(results, indent=2, allow_nan=False))
    temp_output.replace(output_path)

    m = results["metrics"]
    print("=" * 64)
    print(f"checkpoint        {checkpoint_dir}")
    print(f"task              {args.task_name}")
    print(f"val episodes      {episode_ids} ({num_frames} frames scored)")
    print(
        f"chunk / dims      horizon={horizon}  action_dim={action_dim}  denoise_steps={args.num_denoising_steps}"
    )
    print("-" * 64)
    print(f"mse_norm          {m['mse_norm']:.5f}   <- analogue of train_loop.py val_recon_error")
    print(
        f"mse_norm_all_dims {m['mse_norm_all_dims']:.5f}   same, including predicted padding dims"
    )
    if loss_count:
        print(
            f"val_flow_loss     {metrics['val_flow_loss']:.5f}   training objective, eval mode, all model dims"
        )
    print(f"rmse_norm         {m['rmse_norm']:.5f}")
    print(f"mae_norm          {m['mae_norm']:.5f}")
    print(
        f"joint_rmse_rad    {m['joint_rmse_rad']:.5f}   rad, absolute joint space, averaged over the chunk"
    )
    print(f"gripper_rmse      {m['gripper_rmse']:.5f}   normalized gripper travel")
    print("-" * 64)
    print(f"{'chunk position':<18}{'mse_norm':>12}{'joint_rmse_rad':>18}")
    for h in [0, horizon // 4, horizon // 2, 3 * horizon // 4, horizon - 1]:
        print(
            f"  t+{h:<15d}{m['per_horizon_mse_norm'][h]:>12.5f}{m['per_horizon_joint_rmse_rad'][h]:>18.5f}"
        )
    print("-" * 64)
    print(f"wrote             {output_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
