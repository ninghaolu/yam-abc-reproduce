"""Shared selection, normalization discovery, and metrics for offline validation."""

import difflib
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_TASK_NAME = "put the plastic bottles in the bin"


def task_name(value: str) -> str:
    """Accept either a directory slug or the exact dataset task text."""
    return " ".join(value.replace("_", " ").split())


def select_episodes(meta, requested_task: str) -> tuple[list[int], int]:
    available = sorted({str(task) for tasks in meta.episodes["tasks"] for task in tasks})
    if requested_task not in available:
        suggestions = difflib.get_close_matches(requested_task, available, n=5, cutoff=0.2)
        raise ValueError(f"Unknown exact ABC task {requested_task!r}. Closest tasks: {suggestions}")
    selected, frames, mixed = [], 0, []
    for index, tasks, length in zip(
        meta.episodes["episode_index"], meta.episodes["tasks"], meta.episodes["length"], strict=True
    ):
        names = {str(task) for task in tasks}
        if requested_task not in names:
            continue
        if names != {requested_task}:
            mixed.append((int(index), sorted(names)))
            continue
        selected.append(int(index))
        frames += int(length)
    if mixed:
        raise ValueError(f"Requested task occurs in mixed-task episodes: {mixed[:5]}")
    if not selected:
        raise ValueError(f"No validation episodes found for {requested_task!r}")
    return selected, frames


def resolve_checkpoint_stats(checkpoint: Path, preferred_asset: str | None) -> Path:
    candidates = sorted((checkpoint / "assets").glob("*/norm_stats.json"))
    if preferred_asset:
        preferred = checkpoint / "assets" / preferred_asset / "norm_stats.json"
        if preferred in candidates:
            return preferred
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Cannot unambiguously select checkpoint normalization statistics: {candidates}"
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_action_errors(
    se_norm_hd, ae_norm_hd, se_raw_hd, num_frames, *, padded_sum=None, model_dim=None
):
    """Reduce sums over frames; each remaining array axis is [horizon, real dim]."""
    if num_frames <= 0:
        raise ValueError("No validation frames were evaluated")
    horizon, action_dim = se_norm_hd.shape
    if action_dim % 7:
        raise ValueError("Expected six joints and one gripper per arm")
    joints = [d for d in range(action_dim) if d % 7 != 6]
    grippers = [d for d in range(action_dim) if d % 7 == 6]
    denom = num_frames * horizon * action_dim
    mse = se_norm_hd.sum() / denom
    raw_d = se_raw_hd.sum(axis=0) / (num_frames * horizon)
    norm_h = se_norm_hd.sum(axis=1) / (num_frames * action_dim)
    joint_h = np.sqrt(se_raw_hd[:, joints].mean(axis=1) / num_frames)
    comparison_horizon = min(30, horizon)
    raw_30_d = se_raw_hd[:comparison_horizon].sum(axis=0) / (num_frames * comparison_horizon)
    metrics = {
        "mse_norm": float(mse),
        "rmse_norm": float(np.sqrt(mse)),
        "mae_norm": float(ae_norm_hd.sum() / denom),
        "joint_rmse_rad": float(np.sqrt(raw_d[joints].mean())),
        "gripper_rmse": float(np.sqrt(raw_d[grippers].mean())),
        "joint_rmse_rad_first_step": float(joint_h[0]),
        "joint_rmse_rad_last_step": float(joint_h[-1]),
        "per_dim_rmse_raw": np.sqrt(raw_d).tolist(),
        "per_horizon_mse_norm": norm_h.tolist(),
        "per_horizon_joint_rmse_rad": joint_h.tolist(),
        "first_step_mse_norm": float(norm_h[0]),
        "last_step_mse_norm": float(norm_h[-1]),
        "joint_rmse_rad_first_30_steps": float(np.sqrt(raw_30_d[joints].mean())),
        "gripper_rmse_first_30_steps": float(np.sqrt(raw_30_d[grippers].mean())),
    }
    if padded_sum is not None:
        metrics["mse_norm_all_dims"] = float(padded_sum / (num_frames * horizon * model_dim))
    return metrics
