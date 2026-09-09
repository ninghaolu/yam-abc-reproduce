import dataclasses
from types import SimpleNamespace

import datasets as hf_datasets
import jax
import lerobot.datasets.io_utils as lerobot_io_utils
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_lerobot_language_feature_compat():
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "language_persistent": {"dtype": "language", "shape": (1,), "names": None},
        "language_events": {"dtype": "language", "shape": (1,), "names": None},
    }

    hf_features = _data_loader._get_hf_features_with_language_compat(features)  # noqa: SLF001

    assert isinstance(hf_features["language_persistent"], hf_datasets.List)
    assert isinstance(hf_features["language_events"], hf_datasets.List)
    assert "timestamp" in hf_features["language_persistent"].feature
    assert "timestamp" not in hf_features["language_events"].feature


def test_remove_unused_lerobot_language_columns_before_tensor_formatting():
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "language_events": {"dtype": "language", "shape": (1,), "names": None},
    }
    dataset = hf_datasets.Dataset.from_dict(
        {
            "observation.state": [[1.0, 2.0]],
            "language_events": [[{"role": "user", "content": "move", "tool_calls": []}]],
        }
    )
    dataset.set_transform(lerobot_io_utils.hf_transform_to_torch)

    with pytest.raises(RuntimeError, match="Could not infer dtype of dict"):
        _ = dataset[0]

    dataset = _data_loader._remove_unused_lerobot_language_columns(dataset, features)  # noqa: SLF001
    item = dataset[0]

    assert "language_events" not in item
    torch.testing.assert_close(item["observation.state"], torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("load_method", ["try_load", "load_and_activate"])
@pytest.mark.parametrize("episodes", [None, [7]])
def test_reader_removes_language_before_index_mapping(monkeypatch, tmp_path, load_method, episodes):
    features = {
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "language_events": {"dtype": "language", "shape": (1,), "names": None},
    }
    dataset = hf_datasets.Dataset.from_dict({
        "index": [100, 101],
        "task_index": [3, 3],
        "language_events": [[{"role": "user", "content": "move", "tool_calls": []}]] * 2,
    })
    monkeypatch.setattr(
        _data_loader.lerobot_dataset_reader, "load_nested_dataset", lambda *args, **kwargs: dataset
    )
    reader = _data_loader.lerobot_dataset_reader.DatasetReader(
        meta=SimpleNamespace(features=features), root=tmp_path, episodes=episodes,
        tolerance_s=1e-3, video_backend="pyav", delta_timestamps=None, image_transforms=None,
    )
    monkeypatch.setattr(reader, "_check_cached_episodes_sufficient", lambda: True)

    getattr(reader, load_method)()

    assert "language_events" not in reader.hf_dataset.column_names
    assert reader.hf_dataset[0]["task_index"].item() == 3
    assert reader._absolute_to_relative_idx == (None if episodes is None else {100: 0, 101: 1})  # noqa: SLF001


def test_create_torch_dataset_forwards_lerobot_timestamp_tolerance(monkeypatch):
    observed_kwargs = {}

    class FakeMetadata:
        fps = 30

    class FakeLeRobotDataset:
        reader = None

    monkeypatch.setattr(
        _data_loader.lerobot_dataset,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: FakeMetadata(),
    )

    def create_dataset(*args, **kwargs):
        observed_kwargs.update(kwargs)
        return FakeLeRobotDataset()

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", create_dataset)

    data_config = _config.DataConfig(repo_id="example", lerobot_tolerance_s=1e-3)
    _data_loader.create_torch_dataset(data_config, action_horizon=2, model_config=object())

    assert observed_kwargs["tolerance_s"] == 1e-3


def test_pi05_abc130k_uses_relaxed_lerobot_timestamp_tolerance():
    config = _config.get_config("pi05_abc130k")

    assert config.data.lerobot_tolerance_s == 1e-3


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
