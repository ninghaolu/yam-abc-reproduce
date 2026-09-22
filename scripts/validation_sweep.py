#!/usr/bin/env python3
"""Prepare, run, and collect a fixed snapshot of the requested put-task checkpoint sweep."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "yam-abc-reproduce"
OPENPI = REPO / "third_party/policy/openpi"
TASK = "put the plastic bottles in the bin"
TASK_STATS = WORKSPACE / "norm_stats_abc/put_the_plastic_bottles_in_the_bin/norm_stats.json"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def prepare_run(output, checkpoint_run, interval, val_root, batch_size, num_workers):
    """Freeze a single pi0.5 run, using each checkpoint's saved statistics."""
    if (output / "manifest.json").exists():
        raise FileExistsError("A sweep manifest already exists; use a new output directory")
    if interval <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError("interval/batch_size must be positive and num_workers nonnegative")
    if not (val_root / "meta/info.json").is_file():
        raise FileNotFoundError(val_root / "meta/info.json")
    entries = []
    for checkpoint in sorted(checkpoint_run.resolve().iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not checkpoint.is_dir() or not checkpoint.name.isdigit():
            continue
        step = int(checkpoint.name)
        if step <= 0 or step % interval:
            continue
        if not (checkpoint / "params/_METADATA").is_file():
            raise ValueError(f"Checkpoint lacks finalized params metadata: {checkpoint}")
        stats_paths = sorted((checkpoint / "assets").glob("*/norm_stats.json"))
        if len(stats_paths) != 1:
            raise ValueError(f"Ambiguous checkpoint stats: {checkpoint}")
        stats = stats_paths[0]
        entries.append({
            "id": f"pi05_all_data_{step}", "group": "pi05_all_data", "step": step,
            "checkpoint": str(checkpoint), "checkpoint_norm_stats": str(stats),
            "norm_stats_path": str(stats),
            "norm_stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            "action_horizon": 50,
        })
    if not entries:
        raise ValueError(f"No checkpoints at interval {interval} under {checkpoint_run}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_name": TASK, "val_repo_id": "abc_130k_v3_val",
        "val_root": str(val_root.resolve()), "batch_size": batch_size,
        "num_workers": num_workers, "num_denoising_steps": 10, "seed": 0,
        "checkpoints": entries,
    }
    atomic_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        f"# Bottles validation sweep\n\nTask: {TASK}. Checkpoint interval: {interval}.\n"
        "Uses each checkpoint's saved global normalization statistics and EMA/inference parameters.\n"
        "Computes native validation flow loss plus sampled action error with ten denoising steps.\n"
        "Fixed seed 0; final incomplete batch dropped. Physical joint/gripper RMSE and normalized MSE "
        "are collected in checkpoint_metrics.csv/tsv. Capped evaluations live separately in smoke/.\n"
    )
    collect(output)
    print(json.dumps(manifest, indent=2))


def prepare(output):
    if (output / "manifest.json").exists():
        raise FileExistsError("A sweep manifest already exists; use a new output directory")
    task_hash = hashlib.sha256(TASK_STATS.read_bytes()).hexdigest()
    entries = []
    for group, relative in (
        ("pi05_task", "pi05_abc130k_task/pi05_abc_put_the_plastic_bottles_in_the_bin_17183582"),
        ("pi05_all_data", "pi05_abc130k/pi05_abc130k_16923747"),
    ):
        run = OPENPI / "checkpoints" / relative
        for checkpoint in sorted(
            (p for p in run.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)
        ):
            step = int(checkpoint.name)
            if step < 4000 or step % 4000:
                continue
            if not (checkpoint / "params/_METADATA").is_file():
                raise ValueError(f"Checkpoint lacks finalized params metadata: {checkpoint}")
            stats_paths = sorted((checkpoint / "assets").glob("*/norm_stats.json"))
            if len(stats_paths) != 1:
                raise ValueError(f"Ambiguous checkpoint stats: {checkpoint}")
            stats = stats_paths[0]
            digest = hashlib.sha256(stats.read_bytes()).hexdigest()
            if group == "pi05_task" and digest != task_hash:
                raise ValueError(f"Task checkpoint stats differ from requested task stats: {stats}")
            entries.append(
                {
                    "id": f"{group}_{step}",
                    "group": group,
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_norm_stats": str(stats),
                    "norm_stats_path": str(TASK_STATS if group == "pi05_task" else stats),
                    "norm_stats_sha256": digest,
                    "action_horizon": 50,
                }
            )
    abc_checkpoint = WORKSPACE / "abc/cache/bottles_75k.pt"
    if not abc_checkpoint.is_file():
        raise FileNotFoundError(abc_checkpoint)
    entries.append(
        {
            "id": "abc_75000",
            "group": "abc_dit",
            "step": 75000,
            "checkpoint": str(abc_checkpoint),
            "norm_stats_path": str(abc_checkpoint) + "#norm_stats",
            "action_horizon": 30,
        }
    )
    manifest = {
        # The Slurm wrapper deliberately uses the cluster's Python 3.9.
        "created_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "task_name": TASK,
        "val_repo_id": "abc_130k_v3_val",
        "val_root": "/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_val",
        "batch_size": 32,
        "num_workers": 8,
        "num_denoising_steps": 10,
        "seed": 0,
        "checkpoints": entries,
    }
    atomic_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# Put-task validation loss and action error\n\n"
        "manifest.json freezes the checkpoint selection (4k, 8k, ... at preparation time). "
        "Each model evaluates the exact put-the-plastic-bottles task in abc_130k_v3_val. "
        "Batch size 32, seed 0, ten action-sampling steps, no action prefix during sampling. "
        "The final incomplete batch is dropped; episode-end actions repeat.\n\n"
        "Task pi0.5 checkpoints use norm_stats_abc/put_the_plastic_bottles_in_the_bin/norm_stats.json, "
        "verified byte-identical to each checkpoint's saved stats. All-data pi0.5 checkpoints use their own "
        "saved global stats. ABC uses its embedded absolute-action mean/std statistics.\n\n"
        "val_flow_loss is the model's native training flow objective in evaluation mode. Pi0.5 averages "
        "all 32 model dimensions and 50 timesteps, using Beta(1.5,1) noise levels. ABC averages per-batch "
        "training objectives over 14 dimensions and 30 timesteps, with Uniform(0,1) noise levels and the "
        "reference training defaults for action prefixes (max 4, probability 1, noise scale 0.05). "
        "The ABC checkpoint does not store its training configuration. Image augmentation and state "
        "masking are disabled in validation. Pi0.5 loads saved inference/EMA parameters.\n\n"
        "Native normalized MSE and flow loss use different representations across model families and "
        "normalization sets. Compare joint_rmse_rad_first_30_steps and gripper_rmse_first_30_steps for "
        "a common horizon and physical units. JSON results preserve per-horizon errors. This is offline "
        "demonstration agreement, not simulated task success. ABC's published bottles policy is evaluated "
        "on the same real put-task split; this does not establish that it was trained on that task.\n\n"
        "smoke/ contains capped checks, results/ contains full evaluations. checkpoint_metrics.csv/tsv "
        "collect full results only. Use scripts/validation_sweep.py collect --output-dir <this directory> "
        "to refresh. Running tasks also refresh automatically. Slurm accounting is authoritative if a "
        "job is forcibly terminated before it can write status.\n"
    )
    collect(output)
    print(json.dumps(manifest, indent=2))


def collect(output):
    manifest = json.loads((output / "manifest.json").read_text())
    with (output / "collect.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for entry in manifest["checkpoints"]:
            result_path = output / "results" / (entry["id"] + ".json")
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            metrics = result.get("metrics", {})
            status_path = output / "status" / (entry["id"] + ".json")
            status = (
                json.loads(status_path.read_text())
                if status_path.exists()
                else {"status": "not_submitted"}
            )
            row = {
                "group": entry["group"],
                "step": entry["step"],
                "status": "completed" if result else status["status"],
                "job_id": status.get("job_id", ""),
            }
            for metric in (
                "val_flow_loss",
                "mse_norm",
                "joint_rmse_rad",
                "gripper_rmse",
                "joint_rmse_rad_first_30_steps",
                "gripper_rmse_first_30_steps",
                "first_step_mse_norm",
            ):
                row[metric] = metrics.get(metric, "")
            row.update(
                {
                    "frames_evaluated": result.get("frames_evaluated", ""),
                    "action_horizon": entry["action_horizon"],
                    "batch_size": manifest["batch_size"],
                    "checkpoint": entry["checkpoint"],
                    "norm_stats_path": entry["norm_stats_path"],
                    "source_json": str(result_path),
                }
            )
            rows.append(row)
        for extension, delimiter in (("csv", ","), ("tsv", "\t")):
            dest = output / f"checkpoint_metrics.{extension}"
            temp = dest.with_suffix(dest.suffix + ".tmp")
            with temp.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter)
                writer.writeheader()
                writer.writerows(rows)
            temp.replace(dest)
    return rows


def run(output, index, max_batches):
    manifest = json.loads((output / "manifest.json").read_text())
    entry = manifest["checkpoints"][index]
    folder = "smoke" if max_batches else "results"
    result = output / folder / (entry["id"] + ".json")
    status_path = output / ("smoke-status" if max_batches else "status") / (entry["id"] + ".json")
    status = {
        "status": "running",
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "checkpoint": entry["checkpoint"],
    }
    atomic_json(status_path, status)
    collect(output)
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(WORKSPACE / "openpi-cache/huggingface"),
            "HF_DATASETS_CACHE": str(WORKSPACE / "openpi-cache/huggingface/datasets"),
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "LEROBOT_DECODER_CACHE_SIZE": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "4",
        }
    )
    for key in ("MAX_BATCHES", "NORM_STATS_PATH", "JAX_PLATFORMS"):
        env.pop(key, None)
    if entry["group"] == "abc_dit":
        env["JAX_PLATFORMS"] = "cpu"
        env["LD_LIBRARY_PATH"] = "/scratch/nl2752/miniconda3/lib" + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        command = [
            str(REPO / ".venv/bin/python"),
            str(REPO / "scripts/eval_abc_validation.py"),
            "--checkpoint",
            entry["checkpoint"],
            "--output",
            str(result),
        ]
        for key in (
            "task_name",
            "val_repo_id",
            "val_root",
            "batch_size",
            "num_workers",
            "num_denoising_steps",
            "seed",
        ):
            command.extend(["--" + key.replace("_", "-"), str(manifest[key])])
        if max_batches:
            command.extend(["--max-batches", str(max_batches)])
    else:
        env.update(
            {
                "CHECKPOINT_DIR": entry["checkpoint"],
                "NORM_STATS_PATH": entry["norm_stats_path"],
                "OUTPUT": str(result),
                "DRY_RUN": "0",
                "COMPUTE_VAL_LOSS": "1",
                "BASE_CONFIG": "pi05_abc130k",
                "FSDP_DEVICES": "1",
            }
        )
        for key in (
            "task_name",
            "val_repo_id",
            "val_root",
            "batch_size",
            "num_workers",
            "num_denoising_steps",
            "seed",
        ):
            env[key.upper()] = str(manifest[key])
        if max_batches:
            env["MAX_BATCHES"] = str(max_batches)
        command = ["bash", str(REPO / "slurm/eval_abc_pi05_action_error.bash")]
    try:
        rc = subprocess.run(command, env=env, cwd=REPO).returncode
        if rc == 0 and not result.exists():
            rc = 1
        status["status"] = "completed" if rc == 0 and result.exists() else f"failed_exit_{rc}"
        status["exit_code"] = rc
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        atomic_json(status_path, status)
        collect(output)
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "collect"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--checkpoint-run", type=Path, help="Prepare only this pi0.5 run; omit for the legacy comparison")
    parser.add_argument("--interval", type=int, default=8000)
    parser.add_argument("--val-root", type=Path, default=Path("/projects/data/datasets/abc_130k_v3_val"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.action == "prepare":
        if args.checkpoint_run:
            prepare_run(output, args.checkpoint_run, args.interval, args.val_root, args.batch_size, args.num_workers)
        else:
            prepare(output)
    elif args.action == "collect":
        print(json.dumps(collect(output), indent=2))
    else:
        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        sys.exit(run(output, index, args.max_batches))


if __name__ == "__main__":
    main()
