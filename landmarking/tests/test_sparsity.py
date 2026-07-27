"""Unit tests for the sparsity pilot feature.

Tests subset computation, config extension, IOD subset lookup,
and experiment config loading.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from landmarking.common.sparsity import make_subset_indices
from landmarking.config.schema import LandmarkingConfig
from landmarking.evaluation.metrics_wflw import (
    compute_nme,
    get_iod_indices_in_subset,
)


# ============================================================================
# make_subset_indices tests
# ============================================================================


class TestMakeSubsetIndices:
    """Tests for the make_subset_indices utility."""

    def test_step_1_returns_all_98(self):
        """step=1 returns all 98 indices (0..97)."""
        result = make_subset_indices(total=98, step=1)
        assert result == list(range(98))
        assert len(result) == 98

    def test_step_4_returns_25(self):
        """step=4 returns exactly 25 indices."""
        result = make_subset_indices(total=98, step=4)
        expected = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
        assert result == expected
        assert len(result) == 25

    def test_forced_60_72_always_present(self):
        """IOD landmarks 60 and 72 are always present."""
        # step=7: range(0,98,7) = [0,7,14,21,28,35,42,49,56,63,70,77,84,91]
        # Neither 60 nor 72 is in this range
        result = make_subset_indices(total=98, step=7)
        assert 60 in result
        assert 72 in result

    def test_step_4_forced_naturally_included(self):
        """For step=4, 60 and 72 are naturally in range(0,98,4)."""
        result_no_forced = sorted(set(range(0, 98, 4)))
        assert 60 in result_no_forced
        assert 72 in result_no_forced
        # So forced adds nothing extra
        result = make_subset_indices(total=98, step=4)
        assert result == result_no_forced

    def test_result_is_sorted(self):
        """Output is always sorted."""
        result = make_subset_indices(total=98, step=7)
        assert result == sorted(result)

    def test_result_has_no_duplicates(self):
        """Output contains no duplicates."""
        result = make_subset_indices(total=98, step=7)
        assert len(result) == len(set(result))

    def test_custom_forced(self):
        """Custom forced list is respected."""
        result = make_subset_indices(total=98, step=10, forced=[5, 95])
        assert 5 in result
        assert 95 in result

    def test_invalid_step_zero(self):
        """step=0 raises ValueError."""
        with pytest.raises(ValueError):
            make_subset_indices(total=98, step=0)

    def test_invalid_step_negative(self):
        """Negative step raises ValueError."""
        with pytest.raises(ValueError):
            make_subset_indices(total=98, step=-1)

    def test_invalid_step_too_large(self):
        """step >= total raises ValueError."""
        with pytest.raises(ValueError):
            make_subset_indices(total=98, step=98)

    def test_fraction_025_equals_step_4(self):
        """Fraction 0.25 → step=round(1/0.25)=4 → same as step=4."""
        step = round(1.0 / 0.25)
        assert step == 4
        result = make_subset_indices(total=98, step=step)
        assert len(result) == 25


# ============================================================================
# Config schema tests
# ============================================================================


class TestConfigSchemaExtension:
    """Tests for landmark_indices field in DatasetConfig."""

    def test_default_empty(self):
        """Default landmark_indices is empty list."""
        config = LandmarkingConfig()
        assert config.dataset.landmark_indices == []

    def test_empty_preserves_num_landmarks(self):
        """When landmark_indices is empty, num_landmarks is unchanged."""
        config = LandmarkingConfig.from_dict({
            "dataset": {"num_landmarks": 98}
        })
        config.resolve_paths()
        assert config.dataset.num_landmarks == 98

    def test_non_empty_sets_num_landmarks(self):
        """Non-empty landmark_indices auto-sets num_landmarks."""
        indices = [0, 4, 8, 12, 60, 72]
        config = LandmarkingConfig.from_dict({
            "dataset": {"landmark_indices": indices}
        })
        config.resolve_paths()
        assert config.dataset.num_landmarks == len(indices)

    def test_validate_accepts_valid(self):
        """validate() passes for valid indices."""
        config = LandmarkingConfig.from_dict({
            "dataset": {"landmark_indices": [0, 10, 60, 72, 97]}
        })
        config.validate()  # Should not raise

    def test_validate_rejects_out_of_range(self):
        """validate() raises ValueError for indices outside [0, 97]."""
        config = LandmarkingConfig.from_dict({
            "dataset": {"landmark_indices": [0, 10, 98]}
        })
        with pytest.raises(ValueError, match="Invalid landmark index"):
            config.validate()

    def test_validate_rejects_negative(self):
        """validate() raises ValueError for negative indices."""
        config = LandmarkingConfig.from_dict({
            "dataset": {"landmark_indices": [-1, 10, 60]}
        })
        with pytest.raises(ValueError, match="Invalid landmark index"):
            config.validate()

    def test_validate_rejects_duplicates(self):
        """validate() raises ValueError for duplicate indices."""
        config = LandmarkingConfig.from_dict({
            "dataset": {"landmark_indices": [0, 10, 10, 60, 72]}
        })
        with pytest.raises(ValueError, match="duplicates"):
            config.validate()

    def test_from_json_with_landmark_indices(self):
        """Config loads landmark_indices from JSON correctly."""
        config = LandmarkingConfig.from_dict({
            "dataset": {
                "name": "wflw",
                "landmark_indices": [0, 4, 8, 60, 72]
            }
        })
        assert config.dataset.landmark_indices == [0, 4, 8, 60, 72]


# ============================================================================
# IOD indices in subset
# ============================================================================


class TestGetIodIndicesInSubset:
    """Tests for get_iod_indices_in_subset."""

    def test_step_4_positions(self):
        """For step=4 subset, 60 is at position 15 and 72 at position 18."""
        indices = make_subset_indices(total=98, step=4)
        left, right = get_iod_indices_in_subset(indices)
        assert left == 15  # indices.index(60)
        assert right == 18  # indices.index(72)

    def test_full_98(self):
        """For full 98 set, positions are 60 and 72."""
        indices = list(range(98))
        left, right = get_iod_indices_in_subset(indices)
        assert left == 60
        assert right == 72

    def test_missing_60_raises(self):
        """Raises ValueError if 60 not in subset."""
        with pytest.raises(ValueError, match="60"):
            get_iod_indices_in_subset([0, 4, 8, 72])

    def test_missing_72_raises(self):
        """Raises ValueError if 72 not in subset."""
        with pytest.raises(ValueError, match="72"):
            get_iod_indices_in_subset([0, 4, 8, 60])


# ============================================================================
# NME backward compatibility
# ============================================================================


class TestNMEBackwardCompat:
    """Tests that compute_nme is backward compatible."""

    def test_default_iod_positions(self):
        """Default call uses positions 60 and 72 (backward compat)."""
        gt = np.zeros((98, 2))
        gt[60] = [10, 50]
        gt[72] = [90, 50]
        pred = gt.copy()
        nme = compute_nme(pred, gt)
        assert nme == 0.0

    def test_iod_zero_returns_none(self):
        """Returns None when IOD is zero."""
        gt = np.zeros((98, 2))
        # Both IOD landmarks at same position
        gt[60] = [50, 50]
        gt[72] = [50, 50]
        pred = gt.copy()
        nme = compute_nme(pred, gt)
        assert nme is None

    def test_custom_iod_positions(self):
        """Custom iod_left/iod_right work correctly."""
        # Create 25-landmark array
        gt = np.random.rand(25, 2) * 100
        gt[15] = [10, 50]  # position of 60 in step=4 subset
        gt[18] = [90, 50]  # position of 72 in step=4 subset
        pred = gt.copy()
        nme = compute_nme(pred, gt, iod_left=15, iod_right=18)
        assert nme == 0.0


# ============================================================================
# Experiment configs load
# ============================================================================


class TestExperimentConfigs:
    """Tests that experiment configs load without error."""

    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent / "config" / "experiments" / "sparsity_25"

    def test_heatmap_config_loads(self, config_dir):
        """heatmap.json loads and has correct fields."""
        config = LandmarkingConfig.from_json(str(config_dir / "heatmap.json"))
        assert config.model.variant == "heatmap"
        assert config.dataset.num_landmarks == 25
        assert len(config.dataset.landmark_indices) == 25
        assert config.training.epochs == 60
        assert config.training.seed == 42

    def test_hrnet_coord_config_loads(self, config_dir):
        """hrnet_coord.json loads and has correct fields."""
        config = LandmarkingConfig.from_json(str(config_dir / "hrnet_coord.json"))
        assert config.model.variant == "hrnet_coord"
        assert config.dataset.num_landmarks == 25
        assert len(config.dataset.landmark_indices) == 25

    def test_fused_global_config_loads(self, config_dir):
        """fused_global.json loads and has correct fields."""
        config = LandmarkingConfig.from_json(str(config_dir / "fused_global.json"))
        assert config.model.variant == "fused_global"
        assert config.dataset.num_landmarks == 25
        assert config.dataset.graph_topology == "chain"
        assert len(config.dataset.landmark_indices) == 25
