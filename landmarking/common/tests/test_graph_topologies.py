"""Unit tests for graph topology utilities."""

import torch
import pytest
from hypothesis import given, settings, strategies as st

from landmarking.common.graph_topologies import (
    make_chain_edge_index,
    make_wflw_edge_index,
    make_cephalometric_edge_index,
    make_subsampled_cephalometric_edge_index,
    get_edge_index,
)


class TestMakeChainEdgeIndex:
    def test_shape_for_9_landmarks(self):
        ei = make_chain_edge_index(9)
        assert ei.shape == (2, 16)  # 2*(9-1) = 16 edges

    def test_shape_for_2_landmarks(self):
        ei = make_chain_edge_index(2)
        assert ei.shape == (2, 2)  # 2*(2-1) = 2 edges

    def test_bidirectional(self):
        ei = make_chain_edge_index(3)
        # Should have edges: 0->1, 1->0, 1->2, 2->1
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        assert (0, 1) in edges
        assert (1, 0) in edges
        assert (1, 2) in edges
        assert (2, 1) in edges

    def test_dtype(self):
        ei = make_chain_edge_index(5)
        assert ei.dtype == torch.long


class TestMakeWflwEdgeIndex:
    def test_shape(self):
        ei = make_wflw_edge_index()
        assert ei.shape == (2, 186)

    def test_max_node_index(self):
        ei = make_wflw_edge_index()
        assert ei.max().item() == 97  # 98 landmarks, 0-indexed

    def test_bidirectional(self):
        ei = make_wflw_edge_index()
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        # Check pupil edges are bidirectional
        assert (96, 64) in edges
        assert (64, 96) in edges
        assert (97, 72) in edges
        assert (72, 97) in edges


class TestGetEdgeIndex:
    def test_chain_topology(self):
        ei = get_edge_index("chain", num_landmarks=9)
        assert ei.shape == (2, 16)

    def test_wflw_topology(self):
        ei = get_edge_index("wflw")
        assert ei.shape == (2, 186)

    def test_chain_requires_num_landmarks(self):
        with pytest.raises(ValueError):
            get_edge_index("chain")

    def test_unknown_topology_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_edge_index("unknown_topology")

    def test_cephalometric_topology_matches_factory(self):
        # get_edge_index("cephalometric") returns the cephalometric edge index.
        ei = get_edge_index("cephalometric")
        expected = make_cephalometric_edge_index()
        assert torch.equal(ei, expected)

    def test_cephalometric_subset_uses_zero_based_indices(self):
        # A subset via get_edge_index uses 0-based indices into the subset.
        subset = [0, 1, 3, 9, 18]
        ei = get_edge_index("cephalometric", landmark_indices=subset)
        assert ei.dtype == torch.long
        assert ei.shape[0] == 2
        if ei.shape[1] > 0:
            assert ei.min().item() >= 0
            assert ei.max().item() <= len(subset) - 1


class TestMakeCephalometricEdgeIndex:
    def test_shape(self):
        ei = make_cephalometric_edge_index()
        assert ei.shape[0] == 2
        E = ei.shape[1]
        # Bidirectional expansion of unique undirected pairs => E is even.
        assert E % 2 == 0

    def test_dtype(self):
        ei = make_cephalometric_edge_index()
        assert ei.dtype == torch.long

    def test_index_bounds(self):
        ei = make_cephalometric_edge_index()
        assert ei.max().item() == 18  # 19 landmarks, 0-indexed
        assert ei.min().item() == 0

    def test_no_isolated_node(self):
        ei = make_cephalometric_edge_index()
        present = set(ei[0].tolist()) | set(ei[1].tolist())
        # Every index 0..18 appears in at least one edge.
        assert present == set(range(19))

    def test_bidirectional(self):
        ei = make_cephalometric_edge_index()
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        for u, v in edges:
            assert (v, u) in edges


class TestSubsampledCephalometricEdgeIndex:
    def test_indices_within_subset_range(self):
        subset = [0, 1, 3, 9, 18]
        ei = make_subsampled_cephalometric_edge_index(subset)
        assert ei.dtype == torch.long
        assert ei.shape[0] == 2
        if ei.shape[1] > 0:
            assert ei.min().item() >= 0
            assert ei.max().item() <= len(subset) - 1

    def test_bidirectional(self):
        subset = [0, 1, 3, 9, 18]
        ei = make_subsampled_cephalometric_edge_index(subset)
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        for u, v in edges:
            assert (v, u) in edges

    def test_all_19_passthrough(self):
        subset = list(range(19))
        ei = make_subsampled_cephalometric_edge_index(subset)
        assert ei.dtype == torch.long
        assert ei.shape[0] == 2
        assert ei.min().item() >= 0
        assert ei.max().item() <= 18

    def test_tiny_subset_no_edges_returns_empty(self):
        # A subset with no surviving adjacencies returns shape (2, 0).
        subset = [0]
        ei = make_subsampled_cephalometric_edge_index(subset)
        assert ei.shape == (2, 0)
        assert ei.dtype == torch.long


# Feature: cephalometric-dataset, Property 5: Cephalometric topology invariant
class TestCephalometricTopologyInvariant:
    @settings(max_examples=100)
    @given(st.integers(min_value=0, max_value=2**31 - 1))
    def test_topology_invariant(self, _seed):
        # Validates: Requirements 4.1, 4.2, 4.4
        # The topology is deterministic; re-assert invariants on each example.
        ei = make_cephalometric_edge_index()
        # Shape (2, E) and dtype torch.long.
        assert ei.shape[0] == 2
        assert ei.dim() == 2
        assert ei.dtype == torch.long
        # All node indices in [0, 18].
        assert ei.min().item() >= 0
        assert ei.max().item() <= 18
        # Bidirectional: for every (u, v) the reverse (v, u) exists.
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        for u, v in edges:
            assert (v, u) in edges
        # No isolated node: every index 0..18 appears in at least one edge.
        present = set(ei[0].tolist()) | set(ei[1].tolist())
        assert present == set(range(19))


# Feature: cephalometric-dataset, Property 6: get_edge_index dispatch equivalence
class TestCephalometricDispatchEquivalence:
    @settings(max_examples=100)
    @given(st.integers(min_value=0, max_value=2**31 - 1))
    def test_dispatch_equals_factory(self, _seed):
        # Validates: Requirements 4.3
        assert torch.equal(
            get_edge_index("cephalometric"),
            make_cephalometric_edge_index(),
        )


# Feature: cephalometric-dataset, Property 7: Subsampled topology index bounds and bidirectionality
class TestSubsampledCephalometricInvariant:
    @settings(max_examples=100)
    @given(
        st.lists(
            st.integers(min_value=0, max_value=18),
            unique=True,
            min_size=1,
            max_size=19,
        ).map(sorted)
    )
    def test_subset_bounds_and_bidirectional(self, subset):
        # Validates: Requirements 4.5, 9.2
        ei = get_edge_index("cephalometric", landmark_indices=subset)
        # dtype torch.long and shape (2, E).
        assert ei.dtype == torch.long
        assert ei.shape[0] == 2
        assert ei.dim() == 2
        E = ei.shape[1]
        if E > 0:
            # All node indices in [0, len(subset) - 1], none outside.
            assert ei.min().item() >= 0
            assert ei.max().item() <= len(subset) - 1
            # Edge set is bidirectional.
            edges = set(zip(ei[0].tolist(), ei[1].tolist()))
            for u, v in edges:
                assert (v, u) in edges
