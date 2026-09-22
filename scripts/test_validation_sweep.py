"""Checkpoint snapshot selection must not include unrelated steps or unfinished saves."""

import json

import pytest
from validation_sweep import prepare_run


def test_single_run_snapshot(tmp_path):
    run = tmp_path / "checkpoints"
    val = tmp_path / "val"
    (val / "meta").mkdir(parents=True)
    (val / "meta/info.json").write_text("{}")
    for step in (0, 2000, 8000, 16000, 18000):
        checkpoint = run / str(step)
        (checkpoint / "params").mkdir(parents=True)
        (checkpoint / "params/_METADATA").write_text("{}")
        stats = checkpoint / "assets/abc130k_yam/norm_stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text("{}")
    (run / "wandb_id.txt").write_text("ignored")
    output = tmp_path / "output"
    prepare_run(output, run, 8000, val, 32, 2)
    manifest = json.loads((output / "manifest.json").read_text())
    assert [entry["step"] for entry in manifest["checkpoints"]] == [8000, 16000]
    assert manifest["task_name"] == "put the plastic bottles in the bin"
    assert all(entry["norm_stats_path"] == entry["checkpoint_norm_stats"] for entry in manifest["checkpoints"])
    assert len((output / "checkpoint_metrics.csv").read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        prepare_run(output, run, 8000, val, 32, 2)
    (run / "24000/params").mkdir(parents=True)
    with pytest.raises(ValueError, match="finalized"):
        prepare_run(tmp_path / "unfinished", run, 8000, val, 32, 2)
