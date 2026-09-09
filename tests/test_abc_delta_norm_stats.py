import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from yam_abc_reproduce.data import abc_delta_norm_stats as delta_stats


def _write_dataset(root, state, action, episode_index, frame_index, split_at):
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(json.dumps({"total_frames": len(state), "fps": 30}))
    (root / "meta/stats.json").write_text("{}")
    index = np.arange(len(state), dtype=np.int64)
    for file_index, positions in enumerate((np.arange(split_at), np.arange(split_at, len(state)))):
        table = pa.table(
            {
                "observation.state": pa.array(
                    state[positions].tolist(), type=pa.list_(pa.float32())
                ),
                "action": pa.array(action[positions].tolist(), type=pa.list_(pa.float32())),
                "episode_index": pa.array(episode_index[positions]),
                "frame_index": pa.array(frame_index[positions]),
                "index": pa.array(index[positions]),
            }
        )
        pq.write_table(
            table, root / f"data/chunk-000/file-{file_index:03d}.parquet", row_group_size=2
        )


def _brute_force(state, action, episode_index, horizon, mask):
    chunks = []
    for anchor in range(len(state)):
        episode_positions = np.flatnonzero(episode_index == episode_index[anchor])
        episode_end = int(episode_positions[-1])
        transformed = []
        for offset in range(horizon):
            target = min(anchor + offset, episode_end)
            value = action[target].copy()
            value[mask] -= state[anchor, mask]
            transformed.append(value)
        chunks.extend(transformed)
    return np.asarray(chunks, dtype=np.float32)


def _expected_quantile(values, num_bins, quantile):
    result = []
    for dim in range(values.shape[1]):
        minimum = float(values[:, dim].min())
        maximum = float(values[:, dim].max())
        if minimum == maximum:
            result.append(minimum)
            continue
        counts, edges = np.histogram(values[:, dim], bins=num_bins, range=(minimum, maximum))
        index = np.searchsorted(np.cumsum(counts), quantile * len(values), side="left")
        result.append(edges[min(index, num_bins - 1)])
    return np.asarray(result)


def test_distributed_full_stats_match_brute_force_across_file_and_episode_boundaries(tmp_path):
    dim = 14
    horizon = 3
    num_bins = 32
    state = np.arange(7 * dim, dtype=np.float32).reshape(7, dim) / 10
    action = state + np.linspace(-0.2, 0.4, dim, dtype=np.float32)
    episode_index = np.asarray([0, 0, 1, 1, 1, 1, 2], dtype=np.int64)
    frame_index = np.asarray([0, 1, 0, 1, 2, 3, 0], dtype=np.int64)
    dataset_root = tmp_path / "dataset"
    # Episode 1 crosses the Parquet boundary, exercising inter-file lookahead.
    _write_dataset(dataset_root, state, action, episode_index, frame_index, split_at=4)

    config = delta_stats.ComputationConfig(
        dataset_root=dataset_root,
        output_dir=tmp_path / "assets",
        work_dir=tmp_path / "work",
        action_horizon=horizon,
        num_bins=num_bins,
        chunk_rows=2,
        num_arms=2,
    )
    for shard_index in range(2):
        delta_stats.run_pass1(config, shard_index=shard_index, num_shards=2)
    delta_stats.reduce_pass1(config, num_shards=2)
    for shard_index in range(2):
        delta_stats.run_pass2(config, shard_index=shard_index, num_shards=2)
    output_path = delta_stats.finalize(config, num_shards=2)

    payload = json.loads(output_path.read_text())["norm_stats"]
    mask = np.asarray(config.delta_mask)
    expected_actions = _brute_force(state, action, episode_index, horizon, mask)
    for key, expected in (("state", state), ("actions", expected_actions)):
        np.testing.assert_allclose(payload[key]["mean"], expected.mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(payload[key]["std"], expected.std(axis=0), atol=1e-6)
        np.testing.assert_allclose(
            payload[key]["q01"], _expected_quantile(expected, num_bins, 0.01), atol=1e-6
        )
        np.testing.assert_allclose(
            payload[key]["q99"], _expected_quantile(expected, num_bins, 0.99), atol=1e-6
        )

    manifest = json.loads((config.output_dir / "global_stats_manifest.json").read_text())
    assert manifest["state_count"] == len(state)
    assert manifest["action_count"] == len(state) * horizon
    assert manifest["all_dataset_rows_used"] is True


def test_task_subset_preserves_chunks_and_excludes_other_tasks(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "task_stats", Path(__file__).parents[1] / "scripts/compute_abc_task_norm_stats.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    state = np.arange(8 * 14, dtype=np.float32).reshape(8, 14)
    action = state + 0.25
    episodes = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    frames = np.array([0, 1, 0, 1, 2, 3, 0, 1])
    _write_dataset(root, state, action, episodes, frames, split_at=4)
    meta_dir = root / "meta/episodes/chunk-000"
    meta_dir.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 1, 2], "tasks": [["other"], ["target"], ["target"]], "length": [2, 4, 2]}), meta_dir / "file-000.parquet")
    subset = tmp_path / "subset"
    provenance = module.prepare_subset(root, subset, "target")
    assert provenance["episode_ids"] == [1, 2]
    assert provenance["total_frames"] == 6
    config = delta_stats.ComputationConfig(dataset_root=subset, output_dir=tmp_path / "output", work_dir=tmp_path / "work", action_horizon=3)
    result = json.loads(delta_stats.run_all(config).read_text())["norm_stats"]
    expected = _brute_force(state[2:], action[2:], episodes[2:], 3, np.array(config.delta_mask))
    np.testing.assert_allclose(result["actions"]["mean"], expected.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(result["state"]["mean"], state[2:].mean(axis=0), atol=1e-6)


def test_task_subset_rejects_mixed_task_episodes(tmp_path):
    import importlib.util
    from pathlib import Path
    import pytest

    spec = importlib.util.spec_from_file_location("task_stats", Path(__file__).parents[1] / "scripts/compute_abc_task_norm_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta_dir = tmp_path / "meta/episodes/chunk-000"
    meta_dir.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0], "tasks": [["target", "other"]], "length": [2]}), meta_dir / "file-000.parquet")
    with pytest.raises(ValueError, match="mixed-task"):
        module.select_episodes(tmp_path, "target")
