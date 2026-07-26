"""Unit tests for the configuration system."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from landmarking.config.schema import (
    LandmarkingConfig,
    PathConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from landmarking.config.resolver import resolve_path, resolve_dataset_dir


# ─── Path to default configs ─────────────────────────────────────────────────
DEFAULTS_DIR = Path(__file__).parent.parent / "defaults"


class TestFromJson:
    """Test loading config from JSON files."""

    def test_load_lizard_defaults(self):
        config = LandmarkingConfig.from_json(str(DEFAULTS_DIR / "lizard.json"))
        assert config.dataset.name == "lizard"
        assert config.dataset.num_landmarks == 9
        assert config.dataset.graph_topology == "chain"
        assert config.model.variant == "fused"
        assert config.training.epochs == 150
        assert config.training.seed == 42

    def test_load_wflw_defaults(self):
        config = LandmarkingConfig.from_json(str(DEFAULTS_DIR / "wflw.json"))
        assert config.dataset.name == "wflw"
        assert config.dataset.num_landmarks == 98
        assert config.dataset.graph_topology == "wflw"
        assert config.training.rot_factor == 30.0

    def test_roundtrip_json(self, tmp_path):
        """Config written to JSON and read back should be identical."""
        original = LandmarkingConfig.from_json(str(DEFAULTS_DIR / "lizard.json"))
        out_path = str(tmp_path / "test_config.json")
        original.to_json(out_path)
        reloaded = LandmarkingConfig.from_json(out_path)

        assert original.to_dict() == reloaded.to_dict()

    def test_all_fields_populated_lizard(self):
        config = LandmarkingConfig.from_json(str(DEFAULTS_DIR / "lizard.json"))
        # Verify key fields are populated (not None)
        assert config.paths is not None
        assert config.dataset is not None
        assert config.model is not None
        assert config.training is not None
        assert config.dataset.input_size == 512
        assert config.model.gnn_hidden == 128
        assert config.training.batch_size == 32

    def test_all_fields_populated_wflw(self):
        config = LandmarkingConfig.from_json(str(DEFAULTS_DIR / "wflw.json"))
        assert config.dataset.input_size == 512
        assert config.model.scale_indices == [0, 1, 2, 3]
        assert config.training.grad_clip == 0.5


class TestFromDict:
    """Test constructing config from dictionaries."""

    def test_empty_dict_uses_defaults(self):
        config = LandmarkingConfig.from_dict({})
        assert config.dataset.name == "lizard"
        assert config.dataset.num_landmarks == 9

    def test_partial_override(self):
        config = LandmarkingConfig.from_dict({
            "dataset": {"name": "wflw", "num_landmarks": 98},
        })
        assert config.dataset.name == "wflw"
        assert config.dataset.num_landmarks == 98
        # Other fields should have defaults
        assert config.training.epochs == 150

    def test_training_override(self):
        config = LandmarkingConfig.from_dict({
            "training": {"epochs": 50, "lr": 0.001},
        })
        assert config.training.epochs == 50
        assert config.training.lr == 0.001
        # Other training fields retain defaults
        assert config.training.batch_size == 32


class TestResolvePaths:
    """Test environment variable overrides via resolve_paths()."""

    def test_resolve_data_root_from_env(self, monkeypatch):
        monkeypatch.setenv("LANDMARKING_DATA_ROOT", "/custom/data")
        config = LandmarkingConfig.from_dict({})
        config.resolve_paths()
        assert config.paths.data_root == "/custom/data"

    def test_resolve_output_root_from_env(self, monkeypatch):
        monkeypatch.setenv("LANDMARKING_OUTPUT_ROOT", "/custom/output")
        config = LandmarkingConfig.from_dict({})
        config.resolve_paths()
        assert config.paths.output_root == "/custom/output"

    def test_resolve_dataset_dir_lizard(self):
        config = LandmarkingConfig.from_dict({
            "paths": {"data_root": "/data"},
            "dataset": {"name": "lizard"},
        })
        config.resolve_paths()
        assert config.dataset.data_dir == "/data/Lizard_data"

    def test_resolve_dataset_dir_wflw(self):
        config = LandmarkingConfig.from_dict({
            "paths": {"data_root": "/data"},
            "dataset": {"name": "wflw"},
        })
        config.resolve_paths()
        assert config.dataset.data_dir == "/data/WFLW_data"

    def test_explicit_data_dir_not_overridden(self):
        config = LandmarkingConfig.from_dict({
            "paths": {"data_root": "/data"},
            "dataset": {"name": "lizard", "data_dir": "/explicit/path"},
        })
        config.resolve_paths()
        assert config.dataset.data_dir == "/explicit/path"


class TestResolver:
    """Test resolver utility functions directly."""

    def test_resolve_path_returns_default_when_no_env(self, monkeypatch):
        monkeypatch.delenv("LANDMARKING_TEST_FIELD", raising=False)
        result = resolve_path("TEST_FIELD", "/default/path")
        assert result == "/default/path"

    def test_resolve_path_returns_env_when_set(self, monkeypatch):
        monkeypatch.setenv("LANDMARKING_TEST_FIELD", "/env/path")
        result = resolve_path("TEST_FIELD", "/default/path")
        assert result == "/env/path"

    def test_resolve_dataset_dir_lizard(self):
        result = resolve_dataset_dir("/root", "lizard")
        assert result == "/root/Lizard_data"

    def test_resolve_dataset_dir_wflw(self):
        result = resolve_dataset_dir("/root", "wflw")
        assert result == "/root/WFLW_data"

    def test_resolve_dataset_dir_unknown(self):
        result = resolve_dataset_dir("/root", "custom_dataset")
        assert result == "/root/custom_dataset"
