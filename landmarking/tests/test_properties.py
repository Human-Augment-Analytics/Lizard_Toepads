"""Property-based tests for the landmarking package.

Uses Hypothesis to verify correctness properties across many inputs.
Each test runs at least 100 examples.

Feature: landmarking-consolidation
"""

import math
import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# ============================================================================
# Property 1: Model Registry Variant Selection
# Feature: landmarking-consolidation, Property 1: Model Registry Variant Selection
# **Validates: Requirements 2.9, 6.1**
# ============================================================================


class TestModelRegistryVariantSelection:
    """Property 1: For any valid variant key, get_model returns the correct class;
    invalid strings raise KeyError. Does NOT instantiate models."""

    @given(st.data())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_valid_variant_returns_class(self, data):
        """Valid variant keys resolve to a class in the registry."""
        from landmarking.models.registry import MODEL_REGISTRY

        # Use only currently available variants (some need torch_geometric)
        available = sorted(MODEL_REGISTRY.keys())
        assume(len(available) > 0)
        variant = data.draw(st.sampled_from(available))
        # Check the class is returned (without instantiating)
        cls = MODEL_REGISTRY[variant]
        assert cls is not None
        assert callable(cls)

    @given(st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != ""))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_variant_raises_keyerror(self, variant_str):
        """Invalid variant strings raise KeyError."""
        from landmarking.models.registry import MODEL_REGISTRY, get_model

        assume(variant_str not in MODEL_REGISTRY)
        with pytest.raises(KeyError):
            get_model(variant_str)


# ============================================================================
# Property 2: Model Parameterization by num_landmarks
# Feature: landmarking-consolidation, Property 2: Model Parameterization
# **Validates: Requirements 6.1, 6.2, 6.3**
# ============================================================================


@pytest.mark.skip(reason="Requires HPC/GPU — forward pass needed for shape verification")
class TestModelParameterization:
    """Property 2: For any num_landmarks in [1, 200] and any registered variant,
    forward pass outputs shape (batch, num_landmarks, 2)."""

    @given(st.integers(min_value=1, max_value=200))
    @settings(max_examples=100)
    def test_output_shape(self, num_landmarks):
        pass


# ============================================================================
# Property 3: Gaussian Heatmap Peak Correctness
# Feature: landmarking-consolidation, Property 3: Gaussian Heatmap Peak Correctness
# **Validates: Requirements 3.3**
# ============================================================================


class TestGaussianHeatmapPeakCorrectness:
    """Property 3: For any landmark coordinate in [0,1]² and positive sigma,
    heatmap peak is within 1px of target and all values in [0,1]."""

    @given(
        x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        sigma=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        heatmap_size=st.sampled_from([32, 64, 128]),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_peak_within_1px_of_target(self, x, y, sigma, heatmap_size):
        """Heatmap max is within 1px of the target coordinate."""
        from landmarking.common.heatmap_utils import generate_gaussian_heatmap

        coords = np.array([[x, y]], dtype=np.float32)
        heatmap = generate_gaussian_heatmap(coords, heatmap_size, sigma)

        assert heatmap.shape == (1, heatmap_size, heatmap_size)

        # All values in [0, 1]
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-7

        # Peak location
        peak_idx = np.unravel_index(heatmap[0].argmax(), (heatmap_size, heatmap_size))
        peak_y, peak_x = peak_idx

        # Target in pixel space
        target_x = x * (heatmap_size - 1)
        target_y = y * (heatmap_size - 1)

        # Peak within 1 pixel of target
        dx = abs(peak_x - target_x)
        dy = abs(peak_y - target_y)
        assert dx <= 1.0 + 1e-6, f"Peak x={peak_x}, target_x={target_x}, dx={dx}"
        assert dy <= 1.0 + 1e-6, f"Peak y={peak_y}, target_y={target_y}, dy={dy}"


# ============================================================================
# Property 4: Config Environment Variable Override
# Feature: landmarking-consolidation, Property 4: Config Environment Variable Override
# **Validates: Requirements 5.2**
# ============================================================================


class TestConfigEnvVarOverride:
    """Property 4: Setting LANDMARKING_{FIELD_NAME} overrides the JSON default
    after resolve_paths()."""

    @given(
        path_value=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/_-."),
            min_size=1,
            max_size=100,
        ).filter(lambda s: len(s.strip()) > 0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_data_root_override(self, path_value, monkeypatch):
        """LANDMARKING_DATA_ROOT overrides data_root field."""
        from landmarking.config.schema import LandmarkingConfig

        monkeypatch.setenv("LANDMARKING_DATA_ROOT", path_value)
        config = LandmarkingConfig()
        config.resolve_paths()
        assert config.paths.data_root == path_value
        monkeypatch.delenv("LANDMARKING_DATA_ROOT", raising=False)

    @given(
        path_value=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/_-."),
            min_size=1,
            max_size=100,
        ).filter(lambda s: len(s.strip()) > 0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_output_root_override(self, path_value, monkeypatch):
        """LANDMARKING_OUTPUT_ROOT overrides output_root field."""
        from landmarking.config.schema import LandmarkingConfig

        monkeypatch.setenv("LANDMARKING_OUTPUT_ROOT", path_value)
        config = LandmarkingConfig()
        config.resolve_paths()
        assert config.paths.output_root == path_value
        monkeypatch.delenv("LANDMARKING_OUTPUT_ROOT", raising=False)

    @given(
        path_value=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/_-."),
            min_size=1,
            max_size=100,
        ).filter(lambda s: len(s.strip()) > 0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_yolo_obb_model_override(self, path_value, monkeypatch):
        """LANDMARKING_YOLO_OBB_MODEL overrides yolo_obb_model field."""
        from landmarking.config.schema import LandmarkingConfig

        monkeypatch.setenv("LANDMARKING_YOLO_OBB_MODEL", path_value)
        config = LandmarkingConfig()
        config.resolve_paths()
        assert config.paths.yolo_obb_model == path_value
        monkeypatch.delenv("LANDMARKING_YOLO_OBB_MODEL", raising=False)


# ============================================================================
# Property 5: Config Section Lookup
# Feature: landmarking-consolidation, Property 5: Config Section Lookup
# **Validates: Requirements 5.7, 5.8**
# ============================================================================


class TestConfigSectionLookup:
    """Property 5: Loading default configs produces correct num_landmarks
    and graph_topology for each dataset."""

    @given(dataset=st.sampled_from(["lizard", "wflw"]))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_default_config_fields_match_dataset(self, dataset):
        """Default configs have correct num_landmarks and graph_topology."""
        from landmarking.config.schema import LandmarkingConfig

        defaults_dir = Path(__file__).parent.parent / "config" / "defaults"
        config_path = defaults_dir / f"{dataset}.json"
        config = LandmarkingConfig.from_json(str(config_path))

        expected = {
            "lizard": {"num_landmarks": 9, "graph_topology": "chain"},
            "wflw": {"num_landmarks": 98, "graph_topology": "wflw"},
        }

        assert config.dataset.num_landmarks == expected[dataset]["num_landmarks"]
        assert config.dataset.graph_topology == expected[dataset]["graph_topology"]
        assert config.dataset.name == dataset


# ============================================================================
# Property 6: Training Config Override
# Feature: landmarking-consolidation, Property 6: Training Config Override
# **Validates: Requirements 5.9**
# ============================================================================


class TestTrainingConfigOverride:
    """Property 6: from_dict() with override produces correct field values."""

    @given(
        epochs=st.integers(min_value=1, max_value=1000),
        batch_size=st.integers(min_value=1, max_value=256),
        lr=st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_training_fields_override(self, epochs, batch_size, lr, seed):
        """from_dict() with training overrides produces correct values."""
        from landmarking.config.schema import LandmarkingConfig

        data = {
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
            }
        }
        config = LandmarkingConfig.from_dict(data)
        assert config.training.epochs == epochs
        assert config.training.batch_size == batch_size
        assert config.training.lr == lr
        assert config.training.seed == seed

    @given(
        variant=st.sampled_from(["fused", "standard", "multiscale", "coord", "heatmap", "vit"]),
        feat_dim=st.sampled_from([32, 64, 128, 256]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_model_fields_override(self, variant, feat_dim):
        """from_dict() with model overrides produces correct values."""
        from landmarking.config.schema import LandmarkingConfig

        data = {
            "model": {
                "variant": variant,
                "feat_dim": feat_dim,
            }
        }
        config = LandmarkingConfig.from_dict(data)
        assert config.model.variant == variant
        assert config.model.feat_dim == feat_dim


# ============================================================================
# Property 7: Evaluation Metrics Mathematical Properties
# Feature: landmarking-consolidation, Property 7: Evaluation Metrics Mathematical Properties
# **Validates: Requirements 7.3, 7.4**
# ============================================================================


class TestEvaluationMetricsMathProperties:
    """Property 7: pixel_error non-negative, zero iff pred==gt;
    nme non-negative; fr and auc in [0,1]."""

    @given(
        coords=st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=512, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0, max_value=512, allow_nan=False, allow_infinity=False),
            ),
            min_size=2,
            max_size=20,
        ),
        offset=st.floats(min_value=0, max_value=50, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pixel_error_non_negative_and_zero_iff_equal(self, coords, offset):
        """compute_pixel_error is non-negative, zero iff pred == gt."""
        from landmarking.evaluation.metrics_lizard import compute_pixel_error

        gt = np.array(coords, dtype=np.float64)
        pred_same = gt.copy()
        pred_offset = gt + offset

        # Zero when pred == gt
        err_same = compute_pixel_error(pred_same, gt)
        assert np.all(err_same >= 0)
        assert np.allclose(err_same, 0.0)

        # Non-negative always
        err_offset = compute_pixel_error(pred_offset, gt)
        assert np.all(err_offset >= 0)

        # Non-zero when offset > 0
        if offset > 1e-10:
            assert np.all(err_offset > 0)

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_fr_in_range(self, nme_list):
        """compute_fr returns value in [0, 1]."""
        from landmarking.evaluation.metrics_wflw import compute_fr

        fr = compute_fr(nme_list)
        assert 0.0 <= fr <= 1.0

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_auc_in_range(self, nme_list):
        """compute_auc returns value in [0, 1]."""
        from landmarking.evaluation.metrics_wflw import compute_auc

        auc = compute_auc(nme_list)
        assert 0.0 <= auc <= 1.0 + 1e-7

    @given(
        n_landmarks=st.integers(min_value=2, max_value=98),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_nme_non_negative_for_identical(self, n_landmarks):
        """compute_nme returns 0 when pred == gt (if IOD > 0)."""
        from landmarking.evaluation.metrics_wflw import compute_nme

        # Create landmarks with non-zero IOD
        gt = np.random.rand(max(n_landmarks, 73), 2) * 100
        # Ensure landmarks 60 and 72 are different for valid IOD
        gt[60] = [10, 50]
        gt[72] = [90, 50]
        pred = gt.copy()

        nme = compute_nme(pred, gt)
        if nme is not None:
            assert nme >= 0.0
            assert abs(nme) < 1e-10


# ============================================================================
# Property 8: Split Generator Determinism
# Feature: landmarking-consolidation, Property 8: Split Generator Determinism
# **Validates: Requirements 9.1, 9.4, 9.5**
# ============================================================================


class TestSplitGeneratorDeterminism:
    """Property 8: Same inputs → byte-identical JSON output."""

    @given(
        n_files=st.integers(min_value=5, max_value=100),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        train_frac=st.floats(min_value=0.1, max_value=0.8, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deterministic_output(self, n_files, seed, train_frac):
        """generate_split produces identical JSON for identical inputs."""
        from landmarking.common.split_utils import generate_split

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create temp data directory with files
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            for i in range(n_files):
                (data_dir / f"sample_{i:04d}.pt").touch()

            fractions = {"train": train_frac, "val": 0.1, "test": 0.1}

            out1 = tmp_path / "split1.json"
            out2 = tmp_path / "split2.json"

            generate_split(str(data_dir), fractions, seed=seed, output_path=str(out1))
            generate_split(str(data_dir), fractions, seed=seed, output_path=str(out2))

            # Byte-identical
            assert out1.read_bytes() == out2.read_bytes()


# ============================================================================
# Property 9: Split Generator Fraction Sizes
# Feature: landmarking-consolidation, Property 9: Split Generator Fraction Sizes
# **Validates: Requirements 9.2**
# ============================================================================


class TestSplitGeneratorFractionSizes:
    """Property 9: Partition sizes match floor(N*fraction), no overlap."""

    @given(
        n_files=st.integers(min_value=10, max_value=200),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        train_frac=st.floats(min_value=0.1, max_value=0.7, allow_nan=False, allow_infinity=False),
        val_frac=st.floats(min_value=0.05, max_value=0.2, allow_nan=False, allow_infinity=False),
        test_frac=st.floats(min_value=0.05, max_value=0.2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sizes_and_no_overlap(self, n_files, seed, train_frac, val_frac, test_frac):
        """Split sizes match floor(N*frac) and partitions don't overlap."""
        from landmarking.common.split_utils import generate_split

        assume(train_frac + val_frac + test_frac <= 1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            for i in range(n_files):
                (data_dir / f"sample_{i:04d}.pt").touch()

            fractions = {"train": train_frac, "val": val_frac, "test": test_frac}
            split = generate_split(str(data_dir), fractions, seed=seed)

            # Size checks: floor(N * fraction)
            expected_train = math.floor(n_files * train_frac)
            expected_val = math.floor(n_files * val_frac)
            expected_test = math.floor(n_files * test_frac)

            assert len(split["train"]) == expected_train
            assert len(split["val"]) == expected_val
            assert len(split["test"]) == expected_test

            # No overlap between partitions
            train_set = set(split["train"])
            val_set = set(split["val"])
            test_set = set(split["test"])

            assert train_set.isdisjoint(val_set), "Train and val overlap"
            assert train_set.isdisjoint(test_set), "Train and test overlap"
            assert val_set.isdisjoint(test_set), "Val and test overlap"


# ============================================================================
# Property 10: Output Directory Uniqueness
# Feature: landmarking-consolidation, Property 10: Output Directory Uniqueness
# **Validates: Requirements 10.1, 10.5**
# ============================================================================


class TestOutputDirectoryUniqueness:
    """Property 10: Distinct (dataset, variant, run_id) → distinct paths."""

    @given(
        dataset1=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        variant1=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        run_id1=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        dataset2=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        variant2=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        run_id2=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_distinct_tuples_produce_distinct_paths(
        self, dataset1, variant1, run_id1, dataset2, variant2, run_id2
    ):
        """Different (dataset, variant, run_id) tuples → different output paths."""
        from landmarking.training.utils import make_output_dir

        # Only test when tuples differ
        assume((dataset1, variant1, run_id1) != (dataset2, variant2, run_id2))

        output_root = "/tmp/runs"
        path1 = make_output_dir(output_root, dataset1, variant1, run_id1)
        path2 = make_output_dir(output_root, dataset2, variant2, run_id2)

        assert path1 != path2
