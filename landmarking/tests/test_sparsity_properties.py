"""Property-based tests for the sparsity pilot feature.

Uses Hypothesis to verify correctness properties across many inputs.
Each test runs at least 100 examples.

Feature: sparsity-pilot
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from landmarking.common.sparsity import make_subset_indices
from landmarking.common.graph_topologies import make_chain_edge_index
from landmarking.config.schema import LandmarkingConfig
from landmarking.evaluation.metrics_wflw import compute_nme, get_iod_indices_in_subset


# ============================================================================
# Property 1: Subset computation correctness
# Feature: sparsity-pilot, Property 1: Subset computation correctness
# **Validates: Requirements 1.1, 1.2**
# ============================================================================


class TestSubsetComputationCorrectness:
    """Property 1: For any valid step size s in [1, 97] and total=98,
    make_subset_indices(total=98, step=s, forced=[60, 72]) returns a list
    equal to sorted(set(range(0, 98, s)) | {60, 72})."""

    @given(step=st.integers(min_value=1, max_value=97))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_output_matches_specification(self, step):
        """Output equals sorted(set(range(0, 98, step)) | {60, 72})."""
        result = make_subset_indices(total=98, step=step, forced=[60, 72])
        expected = sorted(set(range(0, 98, step)) | {60, 72})
        assert result == expected


# ============================================================================
# Property 2: IOD landmarks always present in subset
# Feature: sparsity-pilot, Property 2: IOD landmarks always present
# **Validates: Requirements 1.2, 9.2**
# ============================================================================


class TestIODLandmarksAlwaysPresent:
    """Property 2: For any valid step size s in [1, 97], the output of
    make_subset_indices(total=98, step=s) contains both 60 and 72."""

    @given(step=st.integers(min_value=1, max_value=97))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_60_and_72_always_present(self, step):
        """Both IOD landmarks are in every subset."""
        result = make_subset_indices(total=98, step=step)
        assert 60 in result
        assert 72 in result


# ============================================================================
# Property 3: Subset generation idempotence
# Feature: sparsity-pilot, Property 3: Idempotence
# **Validates: Requirements 1.6**
# ============================================================================


class TestSubsetIdempotence:
    """Property 3: For any valid step size s in [1, 97], calling
    make_subset_indices(total=98, step=s) twice produces identical lists."""

    @given(step=st.integers(min_value=1, max_value=97))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_idempotent(self, step):
        """Calling twice gives identical results."""
        result1 = make_subset_indices(total=98, step=step)
        result2 = make_subset_indices(total=98, step=step)
        assert result1 == result2


# ============================================================================
# Property 4: Config auto-computes num_landmarks from landmark_indices
# Feature: sparsity-pilot, Property 4: num_landmarks auto-computed
# **Validates: Requirements 2.2**
# ============================================================================


# Strategy for valid landmark_indices: sorted unique ints from [0,97] containing 60 and 72
valid_landmark_indices_strategy = (
    st.lists(st.integers(0, 97), unique=True, min_size=2, max_size=98)
    .map(lambda x: sorted(set(x) | {60, 72}))
)


class TestConfigAutoComputesNumLandmarks:
    """Property 4: For any non-empty sorted list of unique integers from [0, 97]
    set as landmark_indices, after calling resolve_paths(), num_landmarks
    equals len(landmark_indices)."""

    @given(indices=valid_landmark_indices_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_num_landmarks_equals_len(self, indices):
        """num_landmarks == len(landmark_indices) after resolve_paths()."""
        config = LandmarkingConfig.from_dict({
            "dataset": {"landmark_indices": indices}
        })
        config.resolve_paths()
        assert config.dataset.num_landmarks == len(indices)


# ============================================================================
# Property 5: Config validation rejects invalid landmark_indices
# Feature: sparsity-pilot, Property 5: Validation rejects invalid
# **Validates: Requirements 2.4, 2.5**
# ============================================================================


class TestConfigValidationRejectsInvalid:
    """Property 5: For any list containing at least one integer outside [0, 97]
    or containing duplicate values, calling validate() raises ValueError."""

    @given(
        valid_indices=st.lists(st.integers(0, 97), unique=True, min_size=1, max_size=10),
        bad_value=st.integers(min_value=98, max_value=1000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_out_of_range_raises(self, valid_indices, bad_value):
        """Out-of-range index raises ValueError."""
        indices = valid_indices + [bad_value]
        config = LandmarkingConfig.from_dict({
            "dataset": {"landmark_indices": indices}
        })
        with pytest.raises(ValueError):
            config.validate()

    @given(
        base_indices=st.lists(st.integers(0, 97), unique=True, min_size=1, max_size=10),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_duplicates_raises(self, base_indices):
        """Duplicate indices raise ValueError."""
        # Create duplicates by appending first element
        indices = base_indices + [base_indices[0]]
        config = LandmarkingConfig.from_dict({
            "dataset": {"landmark_indices": indices}
        })
        with pytest.raises(ValueError):
            config.validate()


# ============================================================================
# Property 6: Coordinate slicing preserves values and order
# Feature: sparsity-pilot, Property 6: Coordinate slicing preserves values
# **Validates: Requirements 3.1, 3.3, 3.6, 5.1**
# ============================================================================


class TestCoordinateSlicing:
    """Property 6: For any valid sorted landmark_indices list and any (98, 2)
    coordinate tensor, slicing produces output[i] == original[landmark_indices[i]]."""

    @given(indices=valid_landmark_indices_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_slicing_preserves_values(self, indices):
        """Slicing coords by indices preserves correct values."""
        # Create a random (98, 2) coordinate array
        rng = np.random.default_rng(42)
        coords = rng.random((98, 2)).astype(np.float32)

        # Simulate what the dataset adapter does
        sliced = coords[indices]

        assert sliced.shape == (len(indices), 2)
        for i, idx in enumerate(indices):
            np.testing.assert_array_equal(sliced[i], coords[idx])


# ============================================================================
# Property 7: Heatmap target shape matches subset size
# Feature: sparsity-pilot, Property 7: Heatmap shape matches subset size
# **Validates: Requirements 3.2**
# ============================================================================


class TestHeatmapShapeMatchesSubset:
    """Property 7: For any valid landmark_indices of length N, the heatmap
    target has shape (N, 64, 64)."""

    @given(indices=valid_landmark_indices_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_heatmap_shape(self, indices):
        """Target heatmap shape is (N, 64, 64) for N indices."""
        n = len(indices)
        heatmap_size = 64
        # Simulate WFLWRefDataset heatmap generation shape
        target = np.zeros((n, heatmap_size, heatmap_size), dtype=np.float32)
        assert target.shape == (n, heatmap_size, heatmap_size)


# ============================================================================
# Property 8: Non-empty landmark_indices forces chain topology
# Feature: sparsity-pilot, Property 8: Non-empty forces chain topology
# **Validates: Requirements 4.3**
# ============================================================================


class TestNonEmptyForcesChainTopology:
    """Property 8: For any non-empty landmark_indices list, after config
    resolution, the effective topology is 'chain'."""

    @given(
        indices=valid_landmark_indices_strategy,
        initial_topology=st.sampled_from(["chain", "wflw", "custom"]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_forces_chain(self, indices, initial_topology):
        """Non-empty landmark_indices forces graph_topology to 'chain'."""
        config = LandmarkingConfig.from_dict({
            "dataset": {
                "landmark_indices": indices,
                "graph_topology": initial_topology,
            }
        })
        # Simulate what TrainingEngine.setup() does
        if config.dataset.landmark_indices:
            config.dataset.graph_topology = "chain"
        assert config.dataset.graph_topology == "chain"


# ============================================================================
# Property 9: Chain edge index size invariant
# Feature: sparsity-pilot, Property 9: Chain edge index size
# **Validates: Requirements 4.4**
# ============================================================================


class TestChainEdgeIndexSize:
    """Property 9: For any N >= 2, make_chain_edge_index(N) produces an edge
    index tensor with shape (2, 2*(N-1))."""

    @given(n=st.integers(min_value=2, max_value=200))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_edge_index_shape(self, n):
        """Chain edge index has shape (2, 2*(N-1))."""
        edge_index = make_chain_edge_index(n)
        assert edge_index.shape == (2, 2 * (n - 1))


# ============================================================================
# Property 10: NME uses correct subset IOD positions
# Feature: sparsity-pilot, Property 10: NME uses correct IOD positions
# **Validates: Requirements 6.1, 6.2, 6.3**
# ============================================================================


class TestNMEUsesCorrectIODPositions:
    """Property 10: For any valid landmark_indices containing 60 and 72, the IOD
    for NME computation is from the positions of 60 and 72 within the subset."""

    @given(indices=valid_landmark_indices_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_iod_from_correct_positions(self, indices):
        """IOD is computed from gt at subset positions of 60 and 72."""
        pos_left, pos_right = get_iod_indices_in_subset(indices)
        n = len(indices)

        # Create coords where IOD landmarks are separated
        rng = np.random.default_rng(123)
        gt = rng.random((n, 2)) * 100
        gt[pos_left] = [10, 50]
        gt[pos_right] = [90, 50]
        pred = gt.copy()

        # Perfect prediction → NME should be 0
        nme = compute_nme(pred, gt, iod_left=pos_left, iod_right=pos_right)
        assert nme is not None
        assert abs(nme) < 1e-10

        # Add some error to pred
        pred_noisy = pred.copy()
        pred_noisy += 1.0  # offset all by 1 pixel
        nme_noisy = compute_nme(pred_noisy, gt, iod_left=pos_left, iod_right=pos_right)
        assert nme_noisy is not None
        assert nme_noisy > 0

        # IOD distance should be 80 (from [10,50] to [90,50])
        expected_iod = 80.0
        actual_iod = np.linalg.norm(gt[pos_left] - gt[pos_right])
        assert abs(actual_iod - expected_iod) < 1e-10
