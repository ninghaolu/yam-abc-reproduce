import json
from decimal import Decimal

import numpy as np
import pytest

from yam_abc_reproduce.data import abc130k_subset as subsets


def _record(task: str, index: int) -> subsets.EpisodeRecord:
    episode_name = f"episode_00000000-0000-0000-0000-{index:012d}"
    return subsets.EpisodeRecord(
        split="train",
        task_name=task,
        episode_name=episode_name,
        raw_relative_path=f"data/train/{task}/{episode_name}/episode.mcap",
        raw_mcap_size=1000 + index,
        export_relative_path=f"train/{episode_name}",
        export_complete=True,
        export_files=tuple(sorted(subsets.EXPECTED_FILES)),
        num_steps=20 + index,
    )


def _population() -> list[subsets.EpisodeRecord]:
    return [
        *(_record("task_a", index) for index in range(70)),
        *(_record("task_b", index + 70) for index in range(20)),
        *(_record("task_c", index + 90) for index in range(10)),
    ]


def test_nested_selection_is_exact_deterministic_and_disjoint():
    fractions = [Decimal("0.05"), Decimal("0.10"), Decimal("0.20")]
    kwargs = {
        "source_dataset": "XDOF/ABC-130k",
        "source_revision": "test-revision",
        "seed": 42,
        "fractions": fractions,
    }
    selected, targets = subsets.build_nested_selection(_population(), **kwargs)
    repeated, repeated_targets = subsets.build_nested_selection(_population(), **kwargs)

    assert (
        targets
        == repeated_targets
        == {
            Decimal("0.05"): 5,
            Decimal("0.10"): 10,
            Decimal("0.20"): 20,
        }
    )
    assert selected == repeated
    assert [entry.lerobot_episode_index for entry in selected] == list(range(20))
    assert [entry.block for entry in selected].count("A") == 5
    assert [entry.block for entry in selected].count("B") == 5
    assert [entry.block for entry in selected].count("C") == 10
    assert len({entry.source_id for entry in selected}) == 20

    other_seed, _ = subsets.build_nested_selection(_population(), **(kwargs | {"seed": 43}))
    assert [entry.source_id for entry in selected] != [entry.source_id for entry in other_seed]


def test_inventory_keeps_incomplete_export_in_official_population(tmp_path):
    source_manifest = tmp_path / "files.tsv"
    export_root = tmp_path / "export"
    (export_root / "train").mkdir(parents=True)
    lines = []
    for index in range(3):
        record = _record("task", index)
        lines.append(f"{record.raw_relative_path}\t{record.raw_mcap_size}\n")
        episode_dir = export_root / record.export_relative_path
        episode_dir.mkdir()
        np.zeros((record.num_steps, subsets.VALUES_PER_STEP), dtype=np.float64).tofile(
            episode_dir / "states_actions.bin"
        )
        if index < 2:
            (episode_dir / "combined_camera-images-rgb.mp4").write_bytes(b"video")
            (episode_dir / "episode_metadata.json").write_text(
                json.dumps(
                    {
                        "task_name": "task",
                        "cameras": ["top", "left", "right"],
                        "camera_resolutions": {
                            "top": [224, 224],
                            "left": [224, 224],
                            "right": [224, 224],
                        },
                        "alignment": "fixed_clock_30hz_causal",
                        "t0_ns": 0,
                        "tick_ns": 33_333_333,
                        "num_steps": record.num_steps,
                    }
                )
            )
    source_manifest.write_text("".join(lines))

    inventory = subsets.build_inventory(source_manifest, export_root, workers=2)

    assert len(inventory) == 3
    assert sum(record.export_complete for record in inventory) == 2
    assert inventory[2].num_steps == 22
    assert inventory[2].export_files == ("states_actions.bin",)


def test_immutable_json_accepts_only_identical_content(tmp_path):
    path = tmp_path / "manifest.json"
    first_digest = subsets._write_immutable_json(path, {"value": 1})
    assert subsets._write_immutable_json(path, {"value": 1}) == first_digest
    assert path.with_suffix(".json.sha256").exists()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        subsets._write_immutable_json(path, {"value": 2})


def test_default_cli_generates_three_nested_fractions():
    args = subsets.build_parser().parse_args(
        [
            "--source-export",
            "/export",
            "--source-manifest",
            "/source/files.tsv",
            "--source-revision",
            "revision",
            "--manifest-root",
            "/manifests",
            "--repo-id",
            "local/abc",
            "--out",
            "/lerobot",
        ]
    )
    assert args.fractions == ["0.05", "0.10", "0.20"]
    assert args.convert_through == "0.10"
