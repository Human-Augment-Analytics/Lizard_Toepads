"""Unit tests for graph topology utilities."""

import torch
import pytest

from landmarking.common.graph_topologies import (
    make_chain_edge_index,
    make_wflw_edge_index,
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
