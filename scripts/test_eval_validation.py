"""Tests for selection, checkpoint statistics, and cross-horizon metric reductions."""

from types import SimpleNamespace

import numpy as np
import pytest
from eval_validation_common import (
    resolve_checkpoint_stats,
    select_episodes,
    summarize_action_errors,
    task_name,
)


def test_checkpoint_stats_preserve_global_and_discover_task(tmp_path):
    global_stats = tmp_path / "assets/abc130k_yam/norm_stats.json"
    task_stats = tmp_path / "assets/put_the_plastic_bottles_in_the_bin/norm_stats.json"
    task_stats.parent.mkdir(parents=True)
    task_stats.write_text("{}")
    assert resolve_checkpoint_stats(tmp_path, "abc130k_yam") == task_stats
    global_stats.parent.mkdir(parents=True)
    global_stats.write_text("{}")
    assert resolve_checkpoint_stats(tmp_path, "abc130k_yam") == global_stats
    with pytest.raises(ValueError, match="unambiguously"):
        resolve_checkpoint_stats(tmp_path, None)


def test_task_slug_and_mixed_episode_rejection():
    requested = task_name("put_the_plastic_bottles_in_the_bin")
    other = "throw the plastic bottles in the bin"
    meta = SimpleNamespace(
        episodes={
            "episode_index": [2, 8, 9],
            "tasks": [[other], [requested], [requested]],
            "length": [10, 20, 30],
        }
    )
    assert select_episodes(meta, requested) == ([8, 9], 50)
    meta.episodes["tasks"][2] = [requested, other]
    with pytest.raises(ValueError, match="mixed-task"):
        select_episodes(meta, requested)


def test_metric_sums_weight_frames_and_separate_grippers_and_horizons():
    diff = np.ones((3, 50, 14))
    diff[1] *= 2
    diff[2] *= 3
    diff[:, 30:] *= 10
    scale = np.ones(14) * 2
    scale[[6, 13]] = 0.5
    raw = diff * scale
    se = (diff**2).sum(0)
    metrics = summarize_action_errors(
        se, np.abs(diff).sum(0), (raw**2).sum(0), 3, padded_sum=se.sum() + 100, model_dim=32
    )
    assert metrics["mse_norm"] == pytest.approx(np.mean(diff**2))
    assert metrics["mse_norm_all_dims"] == pytest.approx((se.sum() + 100) / (3 * 50 * 32))
    assert metrics["joint_rmse_rad_first_30_steps"] == pytest.approx(np.sqrt(14 / 3) * 2)
    assert metrics["gripper_rmse_first_30_steps"] == pytest.approx(np.sqrt(14 / 3) * 0.5)
    assert metrics["joint_rmse_rad_last_step"] == pytest.approx(
        metrics["joint_rmse_rad_first_step"] * 10
    )
    with pytest.raises(ValueError, match="No validation"):
        summarize_action_errors(se, se, se, 0)
