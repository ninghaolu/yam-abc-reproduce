"""Deterministic episode subsets and partial LeRobot conversion for ABC-130K.

The source export is the layout produced by ``third_party/policy/abc/export_mcap.py``:

    <export_root>/<split>/episode_<uuid>/
        combined_camera-images-rgb.mp4
        episode_metadata.json
        states_actions.bin

Sampling is always performed on whole source episodes from the pinned raw
``files.tsv`` inventory. The exported files are only a conversion input; an
incomplete export is reported and is never silently removed from the population.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path, PurePosixPath

import numpy as np

SCHEMA_VERSION = 1
FPS = 30
STATE_DIM = 14
ACTION_DIM = 14
VALUES_PER_STEP = STATE_DIM + ACTION_DIM
EXPECTED_CAMERAS = ("top", "left", "right")
REQUIRED_FILES = (
    "combined_camera-images-rgb.mp4",
    "episode_metadata.json",
    "states_actions.bin",
)
# Backward-compatible public name used by callers of the first version.
EXPECTED_FILES = REQUIRED_FILES
SAMPLING_METHOD = "task_stratified_constrained_deficit_sha256_v1"


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    """One episode in the official source inventory."""

    split: str
    task_name: str
    episode_name: str
    raw_relative_path: str
    raw_mcap_size: int | None
    export_relative_path: str
    export_complete: bool
    export_files: tuple[str, ...]
    num_steps: int | None
    image_width: int | None = None
    image_height: int | None = None
    instruction: str | None = None

    @property
    def episode_uuid(self) -> str:
        return self.episode_name.removeprefix("episode_")

    def source_id(self, source_dataset: str, source_revision: str) -> str:
        return (
            f"{source_dataset}@{source_revision}/{self.split}/{self.task_name}/{self.episode_name}"
        )


@dataclasses.dataclass(frozen=True)
class SelectedEpisode:
    record: EpisodeRecord
    source_id: str
    score: str
    block_index: int
    lerobot_episode_index: int

    @property
    def block(self) -> str:
        if self.block_index >= 26:
            raise ValueError("at most 26 nested fractions are supported")
        return chr(ord("A") + self.block_index)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_immutable_json(path: Path, value: object) -> str:
    """Create a canonical JSON file, accepting only a byte-identical rerun."""

    content = _canonical_json_bytes(value)
    digest = hashlib.sha256(content).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different immutable manifest: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, flags, 0o444)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum = f"{digest}  {path.name}\n".encode()
    if checksum_path.exists():
        if checksum_path.read_bytes() != checksum:
            raise FileExistsError(f"checksum does not match immutable manifest: {checksum_path}")
    else:
        fd = os.open(checksum_path, flags, 0o444)
        with os.fdopen(fd, "wb") as stream:
            stream.write(checksum)
            stream.flush()
            os.fsync(stream.fileno())
    return digest


def _read_raw_train_inventory(source_manifest: Path) -> list[tuple[str, str, int | None]]:
    """Read train episodes from files.tsv or a resolution-filtered census.

    ``files.tsv`` lines have ``path<TAB>size``. A census has one source MCAP path
    per line and therefore records no raw file size.
    """

    episodes: list[tuple[str, str, int | None]] = []
    seen: set[str] = set()
    with source_manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 1:
                name, raw_size = fields[0], None
            elif len(fields) == 2:
                name, raw_size = fields[0], int(fields[1])
            else:
                raise ValueError(f"malformed source inventory line {line_number}: {line!r}")
            parts = PurePosixPath(name).parts
            if (
                len(parts) == 5
                and parts[0] == "data"
                and parts[1] == "train"
                and parts[3].startswith("episode_")
                and parts[4] == "episode.mcap"
            ):
                task_name, episode_name = parts[2], parts[3]
                if episode_name in seen:
                    raise ValueError(
                        f"duplicate train episode name in source manifest: {episode_name}"
                    )
                seen.add(episode_name)
                episodes.append((task_name, episode_name, raw_size))
    if not episodes:
        raise ValueError(f"no train episode.mcap entries found in {source_manifest}")
    return sorted(episodes, key=lambda item: (item[0], item[1]))


def _inspect_export_episode(
    export_root: Path, raw: tuple[str, str, int | None]
) -> EpisodeRecord:
    task_name, episode_name, raw_size = raw
    relative = Path("train") / episode_name
    episode_dir = export_root / relative
    if not episode_dir.is_dir():
        return EpisodeRecord(
            "train",
            task_name,
            episode_name,
            f"data/train/{task_name}/{episode_name}/episode.mcap",
            raw_size,
            relative.as_posix(),
            False,
            (),
            None,
        )

    names = tuple(sorted(entry.name for entry in os.scandir(episode_dir)))
    complete = all(name in names for name in REQUIRED_FILES)
    num_steps: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    instruction: str | None = None
    if complete:
        metadata_path = episode_dir / "episode_metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            num_steps = int(metadata["num_steps"])
            instruction_value = metadata.get("instruction", task_name)
            if not isinstance(instruction_value, str) or not instruction_value.strip():
                raise ValueError(f"invalid instruction: {instruction_value!r}")
            instruction = instruction_value.strip()
            if metadata.get("task_name") != task_name:
                raise ValueError(
                    f"task mismatch: metadata={metadata.get('task_name')!r}, source={task_name!r}"
                )
            if tuple(metadata.get("cameras", ())) != EXPECTED_CAMERAS:
                raise ValueError(f"unexpected cameras: {metadata.get('cameras')!r}")
            resolutions = metadata.get("camera_resolutions")
            if not isinstance(resolutions, dict):
                raise ValueError(f"unexpected camera resolutions: {resolutions!r}")
            camera_shapes = {tuple(resolutions.get(camera, ())) for camera in EXPECTED_CAMERAS}
            if len(camera_shapes) != 1:
                raise ValueError(f"camera resolutions are not uniform: {resolutions!r}")
            shape = next(iter(camera_shapes))
            if len(shape) != 2 or any(
                not isinstance(value, int) or value <= 0 for value in shape
            ):
                raise ValueError(f"invalid camera resolution: {shape!r}")
            image_width, image_height = shape
            expected_stacked = [image_width, image_height * len(EXPECTED_CAMERAS)]
            if (
                "stacked_resolution" in metadata
                and metadata["stacked_resolution"] != expected_stacked
            ):
                raise ValueError(
                    f"unexpected stacked resolution: {metadata.get('stacked_resolution')!r}"
                )
            expected_bands = {
                camera: [index * image_height, (index + 1) * image_height]
                for index, camera in enumerate(EXPECTED_CAMERAS)
            }
            if "view_row_bands" in metadata and metadata["view_row_bands"] != expected_bands:
                raise ValueError(f"unexpected view row bands: {metadata.get('view_row_bands')!r}")
            if metadata.get("alignment") != "fixed_clock_30hz_causal":
                raise ValueError(f"unexpected alignment: {metadata.get('alignment')!r}")
            if metadata.get("tick_ns") != 33_333_333:
                raise ValueError(f"unexpected tick_ns: {metadata.get('tick_ns')!r}")
            if "fps" in metadata and float(metadata["fps"]) != FPS:
                raise ValueError(f"unexpected fps: {metadata.get('fps')!r}")
            binary_size = (episode_dir / "states_actions.bin").stat().st_size
            if binary_size != num_steps * VALUES_PER_STEP * np.dtype(np.float64).itemsize:
                raise ValueError(
                    f"states_actions.bin is {binary_size} bytes; expected "
                    f"{num_steps * VALUES_PER_STEP * np.dtype(np.float64).itemsize}"
                )
            if (episode_dir / "combined_camera-images-rgb.mp4").stat().st_size <= 0:
                raise ValueError("camera video is empty")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid complete export {episode_dir}: {exc}") from exc
    elif "states_actions.bin" in names:
        binary_size = (episode_dir / "states_actions.bin").stat().st_size
        step_bytes = VALUES_PER_STEP * np.dtype(np.float64).itemsize
        if binary_size % step_bytes == 0:
            num_steps = binary_size // step_bytes

    return EpisodeRecord(
        "train",
        task_name,
        episode_name,
        f"data/train/{task_name}/{episode_name}/episode.mcap",
        raw_size,
        relative.as_posix(),
        complete,
        names,
        num_steps,
        image_width,
        image_height,
        instruction,
    )


def build_inventory(
    source_manifest: Path, export_root: Path, *, workers: int = 64
) -> list[EpisodeRecord]:
    """Join the official raw inventory to the flattened exported episodes."""

    raw_inventory = _read_raw_train_inventory(source_manifest)
    if workers < 1:
        raise ValueError("workers must be positive")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(
            pool.map(lambda raw: _inspect_export_episode(export_root, raw), raw_inventory)
        )

    raw_names = {record.episode_name for record in records}
    export_train = export_root / "train"
    exported_names = {
        entry.name
        for entry in os.scandir(export_train)
        if entry.is_dir(follow_symlinks=False) and entry.name.startswith("episode_")
    }
    extras = sorted(exported_names - raw_names)
    missing = sorted(raw_names - exported_names)
    if extras or missing:
        raise ValueError(
            "raw/export train episode directories differ: "
            f"{len(missing)} missing, {len(extras)} extra; "
            f"first missing={missing[:3]}, first extra={extras[:3]}"
        )
    return records


def _target_size(population: int, fraction: Decimal) -> int:
    return int((Decimal(population) * fraction).to_integral_value(rounding=ROUND_HALF_UP))


def _fraction_label(fraction: Decimal) -> str:
    percentage = fraction * 100
    if percentage != percentage.to_integral_value():
        raise ValueError(f"fraction must be an integer percentage for manifest naming: {fraction}")
    return f"{int(percentage):03d}"


def build_nested_selection(
    records: Sequence[EpisodeRecord],
    *,
    source_dataset: str,
    source_revision: str,
    seed: int,
    fractions: Sequence[Decimal],
) -> tuple[list[SelectedEpisode], dict[Decimal, int]]:
    """Build task-stratified, exactly sized, nested episode blocks."""

    ordered_fractions = sorted(set(fractions))
    if not ordered_fractions or ordered_fractions != list(fractions):
        raise ValueError("fractions must be unique and supplied in increasing order")
    if len(ordered_fractions) > 26:
        raise ValueError("at most 26 fractions are supported")
    if ordered_fractions[0] <= 0 or ordered_fractions[-1] > 1:
        raise ValueError("fractions must be in (0, 1]")

    by_task: dict[str, list[tuple[str, EpisodeRecord, str]]] = defaultdict(list)
    for record in records:
        source_id = record.source_id(source_dataset, source_revision)
        score = _stable_hash(str(seed), source_id)
        by_task[record.task_name].append((score, record, source_id))
    for members in by_task.values():
        members.sort(key=lambda item: (item[0], item[2]))

    population = len(records)
    targets = {fraction: _target_size(population, fraction) for fraction in ordered_fractions}
    quotas = {task: 0 for task in by_task}
    block_members: list[list[tuple[str, EpisodeRecord, str]]] = []
    task_tie = {task: _stable_hash(str(seed), "task", task) for task in by_task}

    for fraction in ordered_fractions:
        target = targets[fraction]
        previous = dict(quotas)
        ideals = {
            task: Decimal(target) * Decimal(len(members)) / Decimal(population)
            for task, members in by_task.items()
        }
        while sum(quotas.values()) < target:
            eligible = [task for task, members in by_task.items() if quotas[task] < len(members)]
            if not eligible:
                raise RuntimeError("not enough episodes to satisfy cumulative target")
            max_deficit = max(ideals[task] - Decimal(quotas[task]) for task in eligible)
            tied = [
                task for task in eligible if ideals[task] - Decimal(quotas[task]) == max_deficit
            ]
            chosen = min(tied, key=lambda task: (task_tie[task], task))
            quotas[chosen] += 1

        new_members: list[tuple[str, EpisodeRecord, str]] = []
        for task, members in by_task.items():
            new_members.extend(members[previous[task] : quotas[task]])
        new_members.sort(key=lambda item: (item[0], item[2]))
        block_members.append(new_members)

    selected: list[SelectedEpisode] = []
    for block_index, members in enumerate(block_members):
        for score, record, source_id in members:
            selected.append(
                SelectedEpisode(
                    record=record,
                    source_id=source_id,
                    score=score,
                    block_index=block_index,
                    lerobot_episode_index=len(selected),
                )
            )
    if len(selected) != targets[ordered_fractions[-1]]:
        raise AssertionError("internal error: maximum selection has the wrong size")
    return selected, targets


def _episode_json(selected: SelectedEpisode) -> dict[str, object]:
    record = selected.record
    return {
        "block": selected.block,
        "duration_seconds": None if record.num_steps is None else record.num_steps / FPS,
        "episode_uuid": record.episode_uuid,
        "export_complete": record.export_complete,
        "export_relative_path": record.export_relative_path,
        "image_resolution": [record.image_width, record.image_height],
        "instruction": record.instruction,
        "lerobot_episode_index": selected.lerobot_episode_index,
        "num_steps": record.num_steps,
        "raw_mcap_size": record.raw_mcap_size,
        "raw_relative_path": record.raw_relative_path,
        "sampling_score_sha256": selected.score,
        "source_episode_id": selected.source_id,
        "task_name": record.task_name,
    }


def write_manifests(
    *,
    records: Sequence[EpisodeRecord],
    selected: Sequence[SelectedEpisode],
    targets: dict[Decimal, int],
    fractions: Sequence[Decimal],
    manifest_root: Path,
    source_dataset: str,
    source_revision: str,
    source_manifest: Path,
    export_root: Path,
    seed: int,
    repo_id: str,
    output_root: Path,
    generator_code_sha256: str,
) -> dict[Decimal, tuple[Path, str]]:
    """Write the source inventory and each cumulative immutable manifest."""

    source_manifest_sha = _sha256_file(source_manifest)
    inventory_entries = [
        {
            "episode_uuid": record.episode_uuid,
            "export_complete": record.export_complete,
            "export_files": list(record.export_files),
            "export_relative_path": record.export_relative_path,
            "image_resolution": [record.image_width, record.image_height],
            "instruction": record.instruction,
            "num_steps": record.num_steps,
            "raw_mcap_size": record.raw_mcap_size,
            "raw_relative_path": record.raw_relative_path,
            "source_episode_id": record.source_id(source_dataset, source_revision),
            "task_name": record.task_name,
        }
        for record in records
    ]
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "generator": "yam_abc_reproduce.data.abc130k_subset",
        "generator_code_sha256": generator_code_sha256,
        "source": {
            "dataset": source_dataset,
            "revision": source_revision,
            "source_inventory": str(source_manifest.resolve()),
            "source_inventory_sha256": source_manifest_sha,
            "export_root": str(export_root.resolve()),
            "split": "train",
        },
        "episode_count": len(records),
        "complete_export_count": sum(record.export_complete for record in records),
        "episodes": inventory_entries,
    }
    inventory_path = manifest_root / "source_train_inventory.json"
    inventory_digest = _write_immutable_json(inventory_path, inventory)

    results: dict[Decimal, tuple[Path, str]] = {}
    for block_index, fraction in enumerate(fractions):
        target = targets[fraction]
        episodes = [entry for entry in selected if entry.block_index <= block_index]
        if len(episodes) != target:
            raise AssertionError(f"internal error: {fraction} manifest has {len(episodes)} entries")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generator": "yam_abc_reproduce.data.abc130k_subset",
            "generator_code_sha256": generator_code_sha256,
            "source": {
                "dataset": source_dataset,
                "revision": source_revision,
                "source_inventory_sha256": source_manifest_sha,
                "inventory_manifest": inventory_path.name,
                "inventory_manifest_sha256": inventory_digest,
                "split": "train",
                "population_episode_count": len(records),
            },
            "selection": {
                "seed": seed,
                "requested_fraction": str(fraction),
                "requested_episode_count": target,
                "rounding": "ROUND_HALF_UP",
                "sampling_method": SAMPLING_METHOD,
                "nested_fractions": [str(value) for value in fractions],
                "included_blocks": [chr(ord("A") + i) for i in range(block_index + 1)],
            },
            "lerobot": {
                "repo_id": repo_id,
                "root": str(output_root.resolve()),
                "fps": FPS,
                "mapping_semantics": (
                    "planned immutable index in A-then-B-then-C conversion order; "
                    "the conversion receipt confirms which prefix is physically present"
                ),
                "maximum_planned_episode_count": targets[fractions[-1]],
            },
            "episodes": [_episode_json(entry) for entry in episodes],
        }
        path = manifest_root / f"train_{_fraction_label(fraction)}.json"
        results[fraction] = (path, _write_immutable_json(path, manifest))
    return results


def _lerobot_features(image_width: int, image_height: int) -> dict[str, dict[str, object]]:
    image = {
        "dtype": "video",
        "shape": (image_height, image_width, 3),
        "names": ["height", "width", "channels"],
    }
    names = [
        *(f"left_joint_{index}" for index in range(6)),
        "left_gripper",
        *(f"right_joint_{index}" for index in range(6)),
        "right_gripper",
    ]
    return {
        "observation.images.top_rgb": dict(image),
        "observation.images.left_rgb": dict(image),
        "observation.images.right_rgb": dict(image),
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": names},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": names},
    }


def _add_exported_episode(dataset, episode_dir: Path, expected: SelectedEpisode) -> None:
    """Stream one vertically stacked export episode into a LeRobot writer."""

    import av

    num_steps = expected.record.num_steps
    if num_steps is None:
        raise ValueError(f"episode has no known length: {expected.source_id}")
    values = np.fromfile(episode_dir / "states_actions.bin", dtype=np.float64)
    if values.size != num_steps * VALUES_PER_STEP:
        raise ValueError(
            f"{expected.source_id}: found {values.size} state/action values; "
            f"expected {num_steps * VALUES_PER_STEP}"
        )
    values = values.reshape(num_steps, VALUES_PER_STEP).astype(np.float32)
    image_width = expected.record.image_width
    image_height = expected.record.image_height
    if image_width is None or image_height is None:
        raise ValueError(f"episode has no validated camera geometry: {expected.source_id}")

    decoded = 0
    with av.open(str(episode_dir / "combined_camera-images-rgb.mp4")) as container:
        for decoded, frame in enumerate(container.decode(video=0), 1):
            if decoded > num_steps:
                raise ValueError(
                    f"{expected.source_id}: video contains more than {num_steps} frames"
                )
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (image_height * len(EXPECTED_CAMERAS), image_width, 3):
                raise ValueError(
                    f"{expected.source_id}: unexpected decoded frame shape {rgb.shape}"
                )
            index = decoded - 1
            dataset.add_frame(
                {
                    "observation.images.top_rgb": np.ascontiguousarray(rgb[0:image_height]),
                    "observation.images.left_rgb": np.ascontiguousarray(
                        rgb[image_height : 2 * image_height]
                    ),
                    "observation.images.right_rgb": np.ascontiguousarray(
                        rgb[2 * image_height : 3 * image_height]
                    ),
                    "observation.state": values[index, :STATE_DIM],
                    "action": values[index, STATE_DIM:],
                    "task": expected.record.instruction or expected.record.task_name,
                }
            )
    if decoded != num_steps:
        raise ValueError(f"{expected.source_id}: decoded {decoded} frames; expected {num_steps}")
    dataset.save_episode()


def _metadata_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "meta").rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def convert_prefix(
    *,
    selected: Sequence[SelectedEpisode],
    target_count: int,
    export_root: Path,
    repo_id: str,
    output_root: Path,
    manifest_root: Path,
    fraction: Decimal,
    manifest_digest: str,
    vcodec: str,
    image_writer_threads: int,
    encoder_threads: int | None,
    log_every: int,
) -> None:
    """Create or resume the deterministic A/B/C prefix through ``target_count``."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    prefix = list(selected[:target_count])
    incomplete = [entry.source_id for entry in prefix if not entry.record.export_complete]
    if incomplete:
        details = "\n  ".join(incomplete)
        raise ValueError(
            "selected episodes have incomplete exports; repair these exact raw episodes before conversion:\n"
            f"  {details}"
        )

    geometries = {
        (entry.record.image_width, entry.record.image_height) for entry in prefix
    }
    if len(geometries) != 1:
        raise ValueError(f"selected episodes do not share one validated camera geometry: {geometries}")
    image_width, image_height = next(iter(geometries))
    if image_width is None or image_height is None:
        raise ValueError("selected episodes have no validated camera geometry")

    plan = {
        "schema_version": SCHEMA_VERSION,
        "repo_id": repo_id,
        "root": str(output_root.resolve()),
        "sampling_method": SAMPLING_METHOD,
        "maximum_planned_episode_count": len(selected),
        "maximum_selection_sha256": _stable_hash(*(entry.source_id for entry in selected)),
        "image_resolution": [image_width, image_height],
    }
    plan_path = output_root.parent / f"{output_root.name}.abc130k-plan.json"
    _write_immutable_json(plan_path, plan)

    receipt_path = manifest_root / f"conversion_{_fraction_label(fraction)}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_receipt = {
            "repo_id": repo_id,
            "root": str(output_root.resolve()),
            "converted_through_fraction": str(fraction),
            "converted_prefix_episode_count": target_count,
            "manifest_sha256": manifest_digest,
        }
        mismatches = {
            key: (receipt.get(key), expected)
            for key, expected in expected_receipt.items()
            if receipt.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"existing conversion receipt does not match this run: {mismatches}")
        if not (output_root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"conversion receipt exists but dataset is missing: {output_root}"
            )
        metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=output_root)
        if int(metadata.total_episodes) < target_count:
            raise ValueError(
                f"receipt claims {target_count} episodes but dataset has {metadata.total_episodes}"
            )
        print(f"verified existing conversion receipt: {receipt_path}", flush=True)
        return

    info_path = output_root / "meta" / "info.json"
    has_existing_dataset = info_path.exists()
    if has_existing_dataset:
        metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=output_root)
        existing = int(metadata.total_episodes)
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"non-empty output is not a LeRobot dataset: {output_root}")
    else:
        existing = 0

    if existing > len(selected):
        raise ValueError(
            f"existing dataset has {existing} episodes, beyond the planned maximum {len(selected)}"
        )
    if existing <= target_count:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "root": output_root,
            "batch_encoding_size": 1,
            "vcodec": vcodec,
            "image_writer_threads": image_writer_threads,
            "encoder_threads": encoder_threads,
        }
        if has_existing_dataset:
            dataset = LeRobotDataset.resume(repo_id=repo_id, **kwargs)
            # resume() does not expose metadata_buffer_size in LeRobot 0.5.1. Flush
            # each episode so a preemption cannot leave info.json ahead of episodes.parquet.
            dataset.meta._metadata_buffer_size = 1
        else:
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=FPS,
                features=_lerobot_features(image_width, image_height),
                use_videos=True,
                metadata_buffer_size=1,
                **kwargs,
            )

        try:
            for index in range(existing, target_count):
                entry = selected[index]
                if entry.lerobot_episode_index != index:
                    raise AssertionError("selection order/index mismatch")
                _add_exported_episode(
                    dataset, export_root / entry.record.export_relative_path, entry
                )
                completed = index + 1
                if completed % log_every == 0 or completed == target_count:
                    print(f"converted {completed}/{target_count}: {entry.source_id}", flush=True)
        finally:
            # This closes parquet writers and flushes metadata on normal errors and Ctrl-C.
            # LeRobot's resume constructor cleans up an interrupted episode on the next run.
            dataset.finalize()

    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=output_root)
    if int(metadata.total_episodes) < target_count:
        raise RuntimeError(
            f"conversion finished with {metadata.total_episodes} episodes; expected at least {target_count}"
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "repo_id": repo_id,
        "root": str(output_root.resolve()),
        "converted_through_fraction": str(fraction),
        "converted_prefix_episode_count": target_count,
        "manifest_sha256": manifest_digest,
        "index_range": [0, target_count - 1],
        "image_resolution": [image_width, image_height],
        "metadata_sha256": _metadata_tree_sha256(output_root),
    }
    _write_immutable_json(receipt_path, receipt)


def _parse_fractions(values: Iterable[str]) -> list[Decimal]:
    return [Decimal(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yam-abc-prepare-subsets",
        description=(
            "Create deterministic nested ABC-130K episode manifests and optionally "
            "convert one cumulative prefix to LeRobot."
        ),
    )
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        "--source-census",
        dest="source_manifest",
        type=Path,
        required=True,
        help="Official files.tsv or an authoritative resolution-filtered train census",
    )
    parser.add_argument("--source-dataset", default="XDOF/ABC-130k")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", nargs="+", default=["0.05", "0.10", "0.20"])
    parser.add_argument("--convert-through", default="0.10")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--vcodec", default="h264", choices=("h264", "libsvtav1", "hevc", "auto"))
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--manifests-only",
        action="store_true",
        help="write/verify manifests without creating or resuming LeRobot output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    fractions = _parse_fractions(args.fractions)
    convert_through = Decimal(args.convert_through)
    if convert_through not in fractions:
        raise ValueError("--convert-through must be one of --fractions")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")

    print("auditing official train inventory and exported episode metadata...", flush=True)
    records = build_inventory(args.source_manifest, args.source_export, workers=args.workers)
    complete = sum(record.export_complete for record in records)
    print(f"official train episodes: {len(records)}; complete exports: {complete}", flush=True)

    selected, targets = build_nested_selection(
        records,
        source_dataset=args.source_dataset,
        source_revision=args.source_revision,
        seed=args.seed,
        fractions=fractions,
    )
    manifests = write_manifests(
        records=records,
        selected=selected,
        targets=targets,
        fractions=fractions,
        manifest_root=args.manifest_root,
        source_dataset=args.source_dataset,
        source_revision=args.source_revision,
        source_manifest=args.source_manifest,
        export_root=args.source_export,
        seed=args.seed,
        repo_id=args.repo_id,
        output_root=args.out,
        generator_code_sha256=_sha256_file(Path(__file__)),
    )
    for fraction in fractions:
        path, digest = manifests[fraction]
        print(f"{fraction}: {targets[fraction]} episodes -> {path} ({digest})")

    if not args.manifests_only:
        manifest_path, manifest_digest = manifests[convert_through]
        print(
            f"converting through {convert_through}: {targets[convert_through]} episodes "
            f"using {manifest_path}",
            flush=True,
        )
        convert_prefix(
            selected=selected,
            target_count=targets[convert_through],
            export_root=args.source_export,
            repo_id=args.repo_id,
            output_root=args.out,
            manifest_root=args.manifest_root,
            fraction=convert_through,
            manifest_digest=manifest_digest,
            vcodec=args.vcodec,
            image_writer_threads=args.image_writer_threads,
            encoder_threads=args.encoder_threads,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
