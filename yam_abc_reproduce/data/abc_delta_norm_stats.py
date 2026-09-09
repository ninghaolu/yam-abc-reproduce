"""Full-dataset normalization statistics for ABC/YAM PI0.5 fine-tuning.

This module intentionally reads only the tabular LeRobot columns. It reproduces
OpenPI's 50-step action chunk and YAM absolute-to-delta transform without loading
videos, prompts, or language annotations.

The computation is split into mergeable stages so it can run as Slurm arrays:

* pass1: global count/sum/squared-sum/min/max
* reduce-pass1: merge pass1 shards and establish fixed histogram bounds
* pass2: histogram every transformed value using the global bounds
* finalize: merge histograms and write OpenPI's norm_stats.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import functools
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
DEFAULT_DATASET_ROOT = Path(
    "/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_train"
)
DEFAULT_ACTION_HORIZON = 50
DEFAULT_NUM_BINS = 5000
DEFAULT_CHUNK_ROWS = 65_536
DEFAULT_NUM_ARMS = 2

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "third_party/policy/openpi/assets/pi05_abc130k/abc130k_yam"
DEFAULT_WORK_DIR = _REPO_ROOT.parent / "policy-finetuning-logs/pi05-abc130k/global-delta-norm-stats"


def make_delta_mask(num_arms: int) -> tuple[bool, ...]:
    """Return ``6 delta joints + 1 absolute gripper`` once per arm."""
    if num_arms <= 0:
        raise ValueError(f"num_arms must be positive, got {num_arms}")
    return tuple(value for _ in range(num_arms) for value in (*([True] * 6), False))


def discover_data_files(dataset_root: Path) -> tuple[Path, ...]:
    files = tuple(sorted(dataset_root.glob("data/chunk-*/file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No LeRobot Parquet files found below {dataset_root / 'data'}")
    return files


def dataset_fingerprint(dataset_root: Path, files: Sequence[Path]) -> str:
    """Fingerprint dataset identity without reading hundreds of GB of payload data."""
    digest = hashlib.sha256()
    for relative_meta_path in ("meta/info.json", "meta/stats.json"):
        path = dataset_root / relative_meta_path
        if path.exists():
            digest.update(relative_meta_path.encode())
            digest.update(path.read_bytes())
    for path in files:
        digest.update(str(path.relative_to(dataset_root)).encode())
    return digest.hexdigest()


def _declared_total_frames(dataset_root: Path) -> int | None:
    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        return None
    value = json.loads(info_path.read_text()).get("total_frames")
    return int(value) if value is not None else None


@dataclasses.dataclass(frozen=True)
class ComputationConfig:
    dataset_root: Path
    output_dir: Path
    work_dir: Path
    action_horizon: int = DEFAULT_ACTION_HORIZON
    num_bins: int = DEFAULT_NUM_BINS
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    num_arms: int = DEFAULT_NUM_ARMS

    def __post_init__(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")

    @property
    def delta_mask(self) -> tuple[bool, ...]:
        return make_delta_mask(self.num_arms)


@dataclasses.dataclass
class Moments:
    count: int
    total: np.ndarray
    total_sq: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def empty(cls, dim: int) -> Moments:
        return cls(
            count=0,
            total=np.zeros(dim, dtype=np.float64),
            total_sq=np.zeros(dim, dtype=np.float64),
            minimum=np.full(dim, np.inf, dtype=np.float64),
            maximum=np.full(dim, -np.inf, dtype=np.float64),
        )

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != self.total.size:
            raise ValueError(
                f"Expected (*, {self.total.size}) values for moments, got {values.shape}"
            )
        if values.shape[0] == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Encountered a non-finite state/action value")
        values64 = values.astype(np.float64, copy=False)
        self.count += values.shape[0]
        self.total += np.sum(values64, axis=0, dtype=np.float64)
        self.total_sq += np.sum(np.square(values64), axis=0, dtype=np.float64)
        self.minimum = np.minimum(self.minimum, np.min(values64, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(values64, axis=0))

    def merge(self, other: Moments) -> None:
        if self.total.shape != other.total.shape:
            raise ValueError("Cannot merge moments with different dimensions")
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq
        self.minimum = np.minimum(self.minimum, other.minimum)
        self.maximum = np.maximum(self.maximum, other.maximum)

    def fields(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_count": np.asarray(self.count, dtype=np.int64),
            f"{prefix}_total": self.total,
            f"{prefix}_total_sq": self.total_sq,
            f"{prefix}_minimum": self.minimum,
            f"{prefix}_maximum": self.maximum,
        }

    @classmethod
    def from_fields(cls, fields: dict[str, np.ndarray], prefix: str) -> Moments:
        return cls(
            count=int(fields[f"{prefix}_count"]),
            total=np.asarray(fields[f"{prefix}_total"], dtype=np.float64),
            total_sq=np.asarray(fields[f"{prefix}_total_sq"], dtype=np.float64),
            minimum=np.asarray(fields[f"{prefix}_minimum"], dtype=np.float64),
            maximum=np.asarray(fields[f"{prefix}_maximum"], dtype=np.float64),
        )


@dataclasses.dataclass(frozen=True)
class FileRows:
    action: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    index: np.ndarray
    state: np.ndarray | None = None

    def __len__(self) -> int:
        return self.action.shape[0]


@dataclasses.dataclass(frozen=True)
class PreparedFile:
    state: np.ndarray
    all_actions: np.ndarray
    episode_end_positions: np.ndarray
    anchor_count: int


def _vector_column(table: pa.Table, name: str, expected_dim: int) -> np.ndarray:
    column = table.column(name).combine_chunks()
    if column.null_count:
        raise ValueError(f"Column {name!r} contains null vectors")

    if pa.types.is_fixed_size_list(column.type):
        dim = column.type.list_size
    elif pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        offsets = column.offsets.to_numpy(zero_copy_only=False)
        lengths = np.diff(offsets)
        if lengths.size and not np.all(lengths == lengths[0]):
            raise ValueError(f"Column {name!r} contains variable-length vectors")
        dim = int(lengths[0]) if lengths.size else expected_dim
    else:
        raise TypeError(f"Expected a list column for {name!r}, got {column.type}")

    if dim != expected_dim:
        raise ValueError(f"Expected {name!r} dimension {expected_dim}, got {dim}")
    values = column.values.to_numpy(zero_copy_only=False)
    if values.size != len(column) * dim:
        raise ValueError(f"Unexpected flattened size for {name!r}: {values.size}")
    return np.asarray(values, dtype=np.float32).reshape(len(column), dim)


def _scalar_column(table: pa.Table, name: str) -> np.ndarray:
    column = table.column(name).combine_chunks()
    if column.null_count:
        raise ValueError(f"Column {name!r} contains null values")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def _read_table(path: Path, columns: Sequence[str], prefix_rows: int | None = None) -> pa.Table:
    parquet_file = pq.ParquetFile(path)
    if prefix_rows is None:
        return parquet_file.read(columns=list(columns), use_threads=False)

    pieces: list[pa.Table] = []
    rows = 0
    for row_group in range(parquet_file.num_row_groups):
        piece = parquet_file.read_row_group(row_group, columns=list(columns), use_threads=False)
        remaining = prefix_rows - rows
        if len(piece) > remaining:
            piece = piece.slice(0, remaining)
        pieces.append(piece)
        rows += len(piece)
        if rows >= prefix_rows:
            break
    if not pieces:
        return pa.table({name: [] for name in columns})
    return pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]


def _read_rows(
    path: Path, action_dim: int, prefix_rows: int | None = None, *, state: bool
) -> FileRows:
    columns = ["action", "episode_index", "frame_index", "index"]
    if state:
        columns.insert(0, "observation.state")
    table = _read_table(path, columns, prefix_rows)
    return FileRows(
        state=_vector_column(table, "observation.state", action_dim) if state else None,
        action=_vector_column(table, "action", action_dim),
        episode_index=_scalar_column(table, "episode_index"),
        frame_index=_scalar_column(table, "frame_index"),
        index=_scalar_column(table, "index"),
    )


def _validate_contiguous(
    episode_index: np.ndarray, frame_index: np.ndarray, global_index: np.ndarray
) -> None:
    if global_index.size <= 1:
        return
    if not np.all(np.diff(global_index) == 1):
        raise ValueError("Dataset rows are not contiguous in the global index")
    if np.any(np.diff(episode_index) < 0):
        raise ValueError("episode_index must be nondecreasing")
    same_episode = episode_index[1:] == episode_index[:-1]
    if not np.all(np.diff(frame_index)[same_episode] == 1):
        raise ValueError("frame_index is not contiguous within an episode")
    if not np.all(frame_index[1:][~same_episode] == 0):
        raise ValueError("A new episode does not begin at frame_index 0")


def _run_end_positions(episode_index: np.ndarray) -> np.ndarray:
    if episode_index.size == 0:
        return np.empty(0, dtype=np.int64)
    run_ends = np.concatenate(
        [np.flatnonzero(episode_index[1:] != episode_index[:-1]), [episode_index.size - 1]]
    )
    run_starts = np.concatenate([[0], run_ends[:-1] + 1])
    return np.repeat(run_ends, run_ends - run_starts + 1)


def _prepare_file(
    file_index: int, files: Sequence[Path], action_horizon: int, action_dim: int
) -> PreparedFile:
    current = _read_rows(files[file_index], action_dim, state=True)
    if current.state is None:
        raise AssertionError("State column unexpectedly missing")
    if len(current) == 0:
        raise ValueError(f"Empty Parquet file: {files[file_index]}")

    actions = [current.action]
    episodes = [current.episode_index]
    frames = [current.frame_index]
    indices = [current.index]
    rows_needed = action_horizon - 1
    next_file_index = file_index + 1
    while rows_needed > 0 and next_file_index < len(files):
        lookahead = _read_rows(
            files[next_file_index], action_dim, prefix_rows=rows_needed, state=False
        )
        if len(lookahead) == 0:
            next_file_index += 1
            continue
        actions.append(lookahead.action)
        episodes.append(lookahead.episode_index)
        frames.append(lookahead.frame_index)
        indices.append(lookahead.index)
        rows_needed -= len(lookahead)
        next_file_index += 1

    all_actions = np.concatenate(actions, axis=0)
    all_episodes = np.concatenate(episodes)
    all_frames = np.concatenate(frames)
    all_indices = np.concatenate(indices)
    _validate_contiguous(all_episodes, all_frames, all_indices)
    episode_ends = _run_end_positions(all_episodes)[: len(current)]
    return PreparedFile(
        state=current.state,
        all_actions=all_actions,
        episode_end_positions=episode_ends,
        anchor_count=len(current),
    )


def _transformed_action_batches(
    prepared: PreparedFile,
    action_horizon: int,
    delta_mask: Sequence[bool],
    chunk_rows: int,
) -> Iterable[np.ndarray]:
    mask = np.asarray(delta_mask, dtype=bool)
    for start in range(0, prepared.anchor_count, chunk_rows):
        stop = min(start + chunk_rows, prepared.anchor_count)
        anchor_positions = np.arange(start, stop, dtype=np.int64)
        states = prepared.state[start:stop]
        episode_ends = prepared.episode_end_positions[start:stop]
        for offset in range(action_horizon):
            target_positions = np.minimum(anchor_positions + offset, episode_ends)
            values = prepared.all_actions[target_positions].copy()
            values[:, mask] -= states[:, mask]
            yield values


def _compute_file_moments(
    file_index: int,
    *,
    files: Sequence[Path],
    action_horizon: int,
    delta_mask: Sequence[bool],
    chunk_rows: int,
) -> tuple[int, Moments, Moments]:
    action_dim = len(delta_mask)
    prepared = _prepare_file(file_index, files, action_horizon, action_dim)
    state_moments = Moments.empty(action_dim)
    action_moments = Moments.empty(action_dim)
    state_moments.update(prepared.state)
    for values in _transformed_action_batches(prepared, action_horizon, delta_mask, chunk_rows):
        action_moments.update(values)
    return file_index, state_moments, action_moments


def _histogram_update(
    histogram: np.ndarray, values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> None:
    num_bins = histogram.shape[1]
    spans = maximum - minimum
    indices = np.zeros(values.shape, dtype=np.int64)
    varying = spans > 0
    if np.any(varying):
        scaled = (
            (values[:, varying].astype(np.float64) - minimum[varying]) / spans[varying] * num_bins
        )
        indices[:, varying] = np.floor(scaled).astype(np.int64)
    np.clip(indices, 0, num_bins - 1, out=indices)
    indices += np.arange(values.shape[1], dtype=np.int64) * num_bins
    histogram += np.bincount(indices.ravel(), minlength=histogram.shape[0] * num_bins).reshape(
        histogram.shape
    )


def _compute_file_histograms(
    file_index: int,
    *,
    files: Sequence[Path],
    action_horizon: int,
    delta_mask: Sequence[bool],
    chunk_rows: int,
    num_bins: int,
    state_minimum: np.ndarray,
    state_maximum: np.ndarray,
    action_minimum: np.ndarray,
    action_maximum: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    action_dim = len(delta_mask)
    prepared = _prepare_file(file_index, files, action_horizon, action_dim)
    state_histogram = np.zeros((action_dim, num_bins), dtype=np.int64)
    action_histogram = np.zeros((action_dim, num_bins), dtype=np.int64)
    _histogram_update(state_histogram, prepared.state, state_minimum, state_maximum)
    for values in _transformed_action_batches(prepared, action_horizon, delta_mask, chunk_rows):
        _histogram_update(action_histogram, values, action_minimum, action_maximum)
    return file_index, state_histogram, action_histogram


def _map_in_order(
    worker,
    file_indices: Sequence[int],
    workers: int,
    label: str,
):
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        iterator = map(worker, file_indices)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(file_indices))
        )
        iterator = executor.map(worker, file_indices, chunksize=1)
    try:
        for completed, result in enumerate(iterator, start=1):
            file_index = result[0]
            print(
                f"{label}: {completed}/{len(file_indices)} file_index={file_index}",
                flush=True,
            )
            yield result
    finally:
        if executor is not None:
            executor.shutdown()


def _save_npz(path: Path, metadata: dict[str, Any], fields: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            np.savez_compressed(
                temp_file,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                **fields,
            )
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _load_npz(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        fields = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
            if name != "metadata_json"
        }
    return metadata, fields


def _identity(
    config: ComputationConfig,
    files: Sequence[Path],
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(config.dataset_root.resolve()),
        "dataset_fingerprint": fingerprint,
        "total_files": len(files),
        "action_horizon": config.action_horizon,
        "delta_mask": list(config.delta_mask),
        "num_bins": config.num_bins,
    }


def _assert_metadata(actual: dict[str, Any], expected: dict[str, Any], path: Path) -> None:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(
                f"Stale or incompatible intermediate {path}: {key} is "
                f"{actual.get(key)!r}, expected {expected_value!r}. Use --force to replace it."
            )


def _partial_path(work_dir: Path, stage: str, shard_index: int, num_shards: int) -> Path:
    return work_dir / stage / f"part-{shard_index:05d}-of-{num_shards:05d}.npz"


def _assigned_indices(total_files: int, shard_index: int, num_shards: int) -> tuple[int, ...]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} is outside [0, {num_shards})")
    indices = tuple(range(shard_index, total_files, num_shards))
    if not indices:
        raise ValueError(
            f"Shard {shard_index}/{num_shards} has no files; use no more than {total_files} shards"
        )
    return indices


def run_pass1(
    config: ComputationConfig,
    *,
    shard_index: int,
    num_shards: int,
    workers: int = 1,
    force: bool = False,
) -> Path:
    files = discover_data_files(config.dataset_root)
    fingerprint = dataset_fingerprint(config.dataset_root, files)
    assigned = _assigned_indices(len(files), shard_index, num_shards)
    output_path = _partial_path(config.work_dir, "pass1", shard_index, num_shards)
    expected_metadata = _identity(config, files, fingerprint) | {
        "stage": "pass1",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "file_indices": list(assigned),
    }
    if output_path.exists() and not force:
        metadata, _ = _load_npz(output_path)
        _assert_metadata(metadata, expected_metadata, output_path)
        print(f"Reusing completed pass1 shard: {output_path}")
        return output_path

    state_moments = Moments.empty(len(config.delta_mask))
    action_moments = Moments.empty(len(config.delta_mask))
    worker = functools.partial(
        _compute_file_moments,
        files=files,
        action_horizon=config.action_horizon,
        delta_mask=config.delta_mask,
        chunk_rows=config.chunk_rows,
    )
    for _, file_state, file_action in _map_in_order(
        worker, assigned, workers, f"pass1 shard {shard_index}/{num_shards}"
    ):
        state_moments.merge(file_state)
        action_moments.merge(file_action)

    if action_moments.count != state_moments.count * config.action_horizon:
        raise AssertionError("Action count does not equal state count times action horizon")
    metadata = expected_metadata | {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "state_count": state_moments.count,
        "action_count": action_moments.count,
    }
    _save_npz(
        output_path,
        metadata,
        state_moments.fields("state") | action_moments.fields("actions"),
    )
    print(f"Wrote {output_path}")
    return output_path


def reduce_pass1(config: ComputationConfig, *, num_shards: int, force: bool = False) -> Path:
    del force  # Reduction output is deterministic and atomically replaceable.
    files = discover_data_files(config.dataset_root)
    fingerprint = dataset_fingerprint(config.dataset_root, files)
    identity = _identity(config, files, fingerprint)
    state_moments = Moments.empty(len(config.delta_mask))
    action_moments = Moments.empty(len(config.delta_mask))
    covered: list[int] = []
    for shard_index in range(num_shards):
        path = _partial_path(config.work_dir, "pass1", shard_index, num_shards)
        metadata, fields = _load_npz(path)
        _assert_metadata(
            metadata,
            identity
            | {
                "stage": "pass1",
                "shard_index": shard_index,
                "num_shards": num_shards,
            },
            path,
        )
        covered.extend(metadata["file_indices"])
        state_moments.merge(Moments.from_fields(fields, "state"))
        action_moments.merge(Moments.from_fields(fields, "actions"))

    if sorted(covered) != list(range(len(files))):
        raise ValueError("Pass1 shards do not cover every Parquet file exactly once")
    if action_moments.count != state_moments.count * config.action_horizon:
        raise AssertionError("Global action count does not equal state count times horizon")
    declared_total_frames = _declared_total_frames(config.dataset_root)
    if declared_total_frames is not None and state_moments.count != declared_total_frames:
        raise ValueError(
            f"Processed {state_moments.count:,} anchors, but meta/info.json declares "
            f"{declared_total_frames:,} frames"
        )

    output_path = config.work_dir / "global_moments.npz"
    metadata = identity | {
        "stage": "reduce-pass1",
        "num_shards": num_shards,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "state_count": state_moments.count,
        "action_count": action_moments.count,
        "declared_total_frames": declared_total_frames,
    }
    _save_npz(
        output_path,
        metadata,
        state_moments.fields("state") | action_moments.fields("actions"),
    )
    print(
        f"Wrote {output_path}: state_count={state_moments.count:,} "
        f"action_count={action_moments.count:,}"
    )
    return output_path


def run_pass2(
    config: ComputationConfig,
    *,
    shard_index: int,
    num_shards: int,
    workers: int = 1,
    force: bool = False,
) -> Path:
    files = discover_data_files(config.dataset_root)
    fingerprint = dataset_fingerprint(config.dataset_root, files)
    identity = _identity(config, files, fingerprint)
    moments_path = config.work_dir / "global_moments.npz"
    moments_metadata, moments_fields = _load_npz(moments_path)
    _assert_metadata(moments_metadata, identity | {"stage": "reduce-pass1"}, moments_path)
    state_moments = Moments.from_fields(moments_fields, "state")
    action_moments = Moments.from_fields(moments_fields, "actions")

    assigned = _assigned_indices(len(files), shard_index, num_shards)
    output_path = _partial_path(config.work_dir, "pass2", shard_index, num_shards)
    expected_metadata = identity | {
        "stage": "pass2",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "file_indices": list(assigned),
    }
    if output_path.exists() and not force:
        metadata, _ = _load_npz(output_path)
        _assert_metadata(metadata, expected_metadata, output_path)
        print(f"Reusing completed pass2 shard: {output_path}")
        return output_path

    state_histogram = np.zeros((len(config.delta_mask), config.num_bins), dtype=np.int64)
    action_histogram = np.zeros_like(state_histogram)
    worker = functools.partial(
        _compute_file_histograms,
        files=files,
        action_horizon=config.action_horizon,
        delta_mask=config.delta_mask,
        chunk_rows=config.chunk_rows,
        num_bins=config.num_bins,
        state_minimum=state_moments.minimum,
        state_maximum=state_moments.maximum,
        action_minimum=action_moments.minimum,
        action_maximum=action_moments.maximum,
    )
    for _, file_state_histogram, file_action_histogram in _map_in_order(
        worker, assigned, workers, f"pass2 shard {shard_index}/{num_shards}"
    ):
        state_histogram += file_state_histogram
        action_histogram += file_action_histogram

    expected_state_count = sum(
        int(state_histogram[dim].sum()) for dim in range(state_histogram.shape[0])
    )
    expected_action_count = sum(
        int(action_histogram[dim].sum()) for dim in range(action_histogram.shape[0])
    )
    metadata = expected_metadata | {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "summed_state_dimension_counts": expected_state_count,
        "summed_action_dimension_counts": expected_action_count,
    }
    _save_npz(
        output_path,
        metadata,
        {
            "state_histogram": state_histogram,
            "actions_histogram": action_histogram,
        },
    )
    print(f"Wrote {output_path}")
    return output_path


def _quantile_from_histogram(
    histogram: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    count: int,
    quantile: float,
) -> np.ndarray:
    values = np.empty(histogram.shape[0], dtype=np.float64)
    target = quantile * count
    for dim, counts in enumerate(histogram):
        if minimum[dim] == maximum[dim]:
            values[dim] = minimum[dim]
            continue
        index = int(np.searchsorted(np.cumsum(counts), target, side="left"))
        index = min(index, histogram.shape[1] - 1)
        values[dim] = minimum[dim] + (maximum[dim] - minimum[dim]) * (index / histogram.shape[1])
    return values


def _statistics(
    moments: Moments, histogram: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if moments.count < 2:
        raise ValueError("Cannot compute statistics from fewer than two vectors")
    if not np.all(histogram.sum(axis=1) == moments.count):
        raise ValueError("Histogram counts do not match the moments count")
    mean = moments.total / moments.count
    variance = moments.total_sq / moments.count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    q01 = _quantile_from_histogram(histogram, moments.minimum, moments.maximum, moments.count, 0.01)
    q99 = _quantile_from_histogram(histogram, moments.minimum, moments.maximum, moments.count, 0.99)
    return mean, std, q01, q99


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(value, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _write_openpi_norm_stats(
    output_dir: Path,
    state_statistics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    action_statistics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Path:
    openpi_src = _REPO_ROOT / "third_party/policy/openpi/src"
    sys.path.insert(0, str(openpi_src))
    from openpi.shared import normalize  # noqa: PLC0415

    state_mean, state_std, state_q01, state_q99 = state_statistics
    action_mean, action_std, action_q01, action_q99 = action_statistics
    norm_stats = {
        "state": normalize.NormStats(mean=state_mean, std=state_std, q01=state_q01, q99=state_q99),
        "actions": normalize.NormStats(
            mean=action_mean, std=action_std, q01=action_q01, q99=action_q99
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".norm-stats-") as temp_dir:
        normalize.save(Path(temp_dir), norm_stats)
        os.replace(Path(temp_dir) / "norm_stats.json", output_path)
    return output_path


def finalize(config: ComputationConfig, *, num_shards: int, force: bool = False) -> Path:
    files = discover_data_files(config.dataset_root)
    fingerprint = dataset_fingerprint(config.dataset_root, files)
    identity = _identity(config, files, fingerprint)
    output_path = config.output_dir / "norm_stats.json"
    manifest_path = config.output_dir / "global_stats_manifest.json"
    if output_path.exists() and not force:
        if not manifest_path.exists():
            raise FileExistsError(
                f"{output_path} already exists without a matching global manifest; "
                "refusing to replace it. Pass --force after checking the target."
            )
        manifest = json.loads(manifest_path.read_text())
        _assert_metadata(manifest, identity, manifest_path)
        print(f"Reusing finalized global stats: {output_path}")
        return output_path

    moments_path = config.work_dir / "global_moments.npz"
    moments_metadata, moments_fields = _load_npz(moments_path)
    _assert_metadata(moments_metadata, identity | {"stage": "reduce-pass1"}, moments_path)
    state_moments = Moments.from_fields(moments_fields, "state")
    action_moments = Moments.from_fields(moments_fields, "actions")

    state_histogram = np.zeros((len(config.delta_mask), config.num_bins), dtype=np.int64)
    action_histogram = np.zeros_like(state_histogram)
    covered: list[int] = []
    for shard_index in range(num_shards):
        path = _partial_path(config.work_dir, "pass2", shard_index, num_shards)
        metadata, fields = _load_npz(path)
        _assert_metadata(
            metadata,
            identity
            | {
                "stage": "pass2",
                "shard_index": shard_index,
                "num_shards": num_shards,
            },
            path,
        )
        covered.extend(metadata["file_indices"])
        state_histogram += fields["state_histogram"]
        action_histogram += fields["actions_histogram"]
    if sorted(covered) != list(range(len(files))):
        raise ValueError("Pass2 shards do not cover every Parquet file exactly once")

    state_statistics = _statistics(state_moments, state_histogram)
    action_statistics = _statistics(action_moments, action_histogram)
    output_path = _write_openpi_norm_stats(config.output_dir, state_statistics, action_statistics)
    manifest = identity | {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "method": "full-parquet-two-pass-fixed-histogram",
        "state_count": state_moments.count,
        "action_count": action_moments.count,
        "declared_total_frames": moments_metadata.get("declared_total_frames"),
        "num_shards": num_shards,
        "work_dir": str(config.work_dir.resolve()),
        "norm_stats_path": str(output_path.resolve()),
        "quantiles": [0.01, 0.99],
        "histogram_quantiles_are_approximate": True,
        "all_dataset_rows_used": True,
    }
    _write_json_atomic(manifest_path, manifest)
    print(
        f"Wrote global OpenPI stats: {output_path}\n"
        f"state_count={state_moments.count:,} action_count={action_moments.count:,}\n"
        f"manifest={manifest_path}"
    )
    return output_path


def run_all(config: ComputationConfig, *, workers: int = 1, force: bool = False) -> Path:
    run_pass1(config, shard_index=0, num_shards=1, workers=workers, force=force)
    reduce_pass1(config, num_shards=1, force=force)
    run_pass2(config, shard_index=0, num_shards=1, workers=workers, force=force)
    return finalize(config, num_shards=1, force=force)


def _slurm_or_default(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute full global PI0.5 delta-action stats from ABC Parquet data"
    )
    parser.add_argument(
        "--stage",
        choices=("count-files", "pass1", "reduce-pass1", "pass2", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--num-arms", type=int, default=DEFAULT_NUM_ARMS)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=_slurm_or_default("SLURM_CPUS_PER_TASK", 1),
        help="Local worker processes within this shard",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace compatible-stage partials or an existing finalized asset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ComputationConfig(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        work_dir=args.work_dir.resolve(),
        action_horizon=args.action_horizon,
        num_bins=args.num_bins,
        chunk_rows=args.chunk_rows,
        num_arms=args.num_arms,
    )
    if args.stage == "count-files":
        print(len(discover_data_files(config.dataset_root)))
        return 0

    num_shards = args.num_shards or _slurm_or_default("SLURM_ARRAY_TASK_COUNT", 1)
    shard_index = (
        args.shard_index
        if args.shard_index is not None
        else _slurm_or_default("SLURM_ARRAY_TASK_ID", 0)
    )
    if args.stage == "pass1":
        run_pass1(
            config,
            shard_index=shard_index,
            num_shards=num_shards,
            workers=args.workers,
            force=args.force,
        )
    elif args.stage == "reduce-pass1":
        reduce_pass1(config, num_shards=num_shards, force=args.force)
    elif args.stage == "pass2":
        run_pass2(
            config,
            shard_index=shard_index,
            num_shards=num_shards,
            workers=args.workers,
            force=args.force,
        )
    elif args.stage == "finalize":
        finalize(config, num_shards=num_shards, force=args.force)
    elif args.stage == "all":
        if num_shards != 1 or shard_index != 0:
            raise ValueError("--stage all is a single-process coordinator and requires one shard")
        run_all(config, workers=args.workers, force=args.force)
    return 0
