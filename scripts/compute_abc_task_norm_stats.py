#!/usr/bin/env python3
"""Compute delta stats for the exact whole-episode task used by task finetuning.

The work directory holds a tabular-only subset (not a training dataset). Global
indices are compacted; episode IDs, frames, states and actions stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yam_abc_reproduce.data import abc_delta_norm_stats as stats

DEFAULT_TASK = "throw the plastic bottles in the bin"


def select_episodes(root: Path, task: str) -> dict[int, int]:
    selected = {}
    for path in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "tasks", "length"]).to_pylist():
            names = set(row["tasks"])
            if task not in names:
                continue
            if names != {task}:
                raise ValueError(f"Task occurs in mixed-task episode: {row}")
            episode = int(row["episode_index"])
            if episode in selected:
                raise ValueError(f"Duplicate episode metadata: {episode}")
            selected[episode] = int(row["length"])
    if not selected:
        raise ValueError(f"No whole episodes found for exact task {task!r}")
    return selected


def prepare_subset(root: Path, subset: Path, task: str) -> dict:
    selected = select_episodes(root, task)
    provenance = {
        "source_dataset_root": str(root.resolve()),
        "task_name": task,
        "episode_ids": sorted(selected),
        "total_episodes": len(selected),
        "total_frames": sum(selected.values()),
        "fps": json.loads((root / "meta/info.json").read_text())["fps"],
        "tabular_stats_subset_only": True,
    }
    marker = subset / "meta/info.json"
    if marker.exists():
        if json.loads(marker.read_text()) != provenance:
            raise ValueError("Work directory belongs to a different task/selection; use a new work directory")
        return provenance
    data_dir = subset / "data/chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted extraction must use a fresh work directory, preventing stale files.
    if list(data_dir.glob("*.parquet")):
        raise ValueError("Incomplete subset exists; use a new work directory")
    counts = dict.fromkeys(selected, 0)
    offset = 0
    output_index = 0
    columns = ["observation.state", "action", "episode_index", "frame_index", "index"]
    files = stats.discover_data_files(root)
    for file_index, path in enumerate(files):
        table = pq.read_table(path, columns=columns, filters=[("episode_index", "in", sorted(selected))], use_threads=False)
        if len(table):
            episodes = table["episode_index"].to_numpy()
            frames = table["frame_index"].to_numpy()
            for episode in np.unique(episodes):
                observed = frames[episodes == episode]
                expected = np.arange(counts[episode], counts[episode] + len(observed))
                if not np.array_equal(observed, expected):
                    raise ValueError(f"Missing or unordered frames in episode {episode}")
                counts[episode] += len(observed)
            table = table.set_column(table.schema.get_field_index("index"), "index", pa.array(np.arange(offset, offset + len(table), dtype=np.int64)))
            pq.write_table(table, data_dir / f"file-{output_index:06d}.parquet")
            offset += len(table)
            output_index += 1
        if file_index % 100 == 0 or file_index == len(files) - 1:
            print(f"Extracted {offset:,} task frames; scanned {file_index + 1}/{len(files)} files", flush=True)
    if counts != selected:
        raise ValueError("Extracted episode lengths do not match task metadata")
    stats._write_json_atomic(marker, provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default=os.environ.get("TASK_NAME", DEFAULT_TASK))
    parser.add_argument("--dataset-root", type=Path, default=stats.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    task = args.task_name.strip()
    slug = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_-]", "", task.lower().replace(" ", "_"))).strip("_")
    if not slug:
        raise ValueError("Task name must produce a usable directory name")
    output = args.output_dir or stats._REPO_ROOT.parent / "norm_stats_abc" / slug
    work = args.work_dir or output / "work"
    if args.inspect:
        selected = select_episodes(args.dataset_root, task)
        print(json.dumps({"task": task, "episodes": len(selected), "frames": sum(selected.values()), "output_dir": str(output)}, indent=2))
        return
    provenance = prepare_subset(args.dataset_root, work / "task-parquet", task)
    config = stats.ComputationConfig(dataset_root=work / "task-parquet", output_dir=output, work_dir=work / "delta-stats")
    stats.run_all(config, workers=args.workers)
    stats._write_json_atomic(output / "task_selection.json", provenance)


if __name__ == "__main__":
    main()
