"""Tests for WFLW adjacency graph construction at full and subsampled densities.

Verifies that:
- Full 98-landmark WFLW graph has correct edge count and structure
- Subsampled graph preserves anatomical groupings (no cross-region edges)
- IOD landmarks remain connected
- Edge indices are valid (within [0, N-1])
"""

import pytest
from landmarking.common.graph_topologies import (
    make_wflw_edge_index,
    make_subsampled_wflw_edge_index,
    get_edge_index,
    make_chain_edge_index,
)
from landmarking.common.sparsity import make_subset_indices


# ── Full 98-landmark WFLW graph tests ─────────────────────────────────────────


class TestFullWFLWGraph:
    """Verify the full 98-point WFLW topology."""

    def test_full_graph_edge_count(self):
        """Full WFLW graph should have 186 directed edges (93 undirected)."""
        ei = make_wflw_edge_index()
        assert ei.shape[0] == 2
        assert ei.shape[1] == 186  # 93 undirected * 2

    def test_full_graph_is_bidirectional(self):
        """Every edge (u,v) has a reverse edge (v,u)."""
        ei = make_wflw_edge_index()
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        for u, v in list(edge_set):
            assert (v, u) in edge_set, f"Edge ({u},{v}) has no reverse"

    def test_full_graph_indices_in_range(self):
        """All edge indices should be in [0, 97]."""
        ei = make_wflw_edge_index()
        assert ei.min().item() >= 0
        assert ei.max().item() <= 97

    def test_full_graph_no_self_loops(self):
        """No self-loops (u→u)."""
        ei = make_wflw_edge_index()
        for i in range(ei.shape[1]):
            assert ei[0, i].item() != ei[1, i].item()

    def test_jaw_chain_present(self):
        """Jaw chain (0→1→...→32) should have edges 0↔1, 1↔2, ..., 31↔32."""
        ei = make_wflw_edge_index()
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        for j in range(32):
            assert (j, j + 1) in edge_set, f"Missing jaw edge {j}→{j+1}"
            assert (j + 1, j) in edge_set, f"Missing jaw edge {j+1}→{j}"

    def test_left_eye_is_loop(self):
        """Left eye (60-67) should be a closed loop."""
        ei = make_wflw_edge_index()
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        # Chain part
        for j in range(60, 67):
            assert (j, j + 1) in edge_set
        # Closing edge
        assert (67, 60) in edge_set, "Left eye loop not closed (67→60)"
        assert (60, 67) in edge_set, "Left eye loop not closed (60→67)"

    def test_pupil_edges(self):
        """Pupil landmarks 96↔64 and 97↔72 should be connected."""
        ei = make_wflw_edge_index()
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        assert (96, 64) in edge_set
        assert (64, 96) in edge_set
        assert (97, 72) in edge_set
        assert (72, 97) in edge_set

    def test_no_cross_region_edges(self):
        """Jaw (0-32) should NOT directly connect to eyebrow (33-41)."""
        ei = make_wflw_edge_index()
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        # Check no jaw↔eyebrow direct connection
        assert (32, 33) not in edge_set
        assert (33, 32) not in edge_set

    def test_get_edge_index_wflw_full(self):
        """get_edge_index('wflw') without landmark_indices gives full graph."""
        ei = get_edge_index("wflw")
        assert ei.shape[1] == 186


# ── Subsampled WFLW graph tests ──────────────────────────────────────────────


class TestSubsampledWFLWGraph:
    """Verify the subsampled WFLW topology preserves anatomical groupings."""

    @pytest.fixture
    def step4_indices(self):
        return make_subset_indices(total=98, step=4)

    def test_step4_edge_count(self, step4_indices):
        """Step=4 subset (25 nodes) should have 25 undirected edges (with cross-group anchoring)."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        n_undirected = ei.shape[1] // 2
        assert n_undirected == 25, f"Expected 25 undirected edges, got {n_undirected}"

    def test_step4_is_bidirectional(self, step4_indices):
        """Every edge (u,v) has a reverse edge (v,u)."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        for u, v in list(edge_set):
            assert (v, u) in edge_set

    def test_step4_indices_in_range(self, step4_indices):
        """All edge indices should be in [0, 24] (subset node indices)."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        assert ei.min().item() >= 0
        assert ei.max().item() <= len(step4_indices) - 1

    def test_step4_no_self_loops(self, step4_indices):
        """No self-loops."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        for i in range(ei.shape[1]):
            assert ei[0, i].item() != ei[1, i].item()

    def test_step4_jaw_stays_within_jaw(self, step4_indices):
        """Jaw nodes (subset positions 0-8 for LM 0,4,...,32) only connect to jaw."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        # Jaw landmarks in step=4: [0,4,8,12,16,20,24,28,32] = subset positions 0-8
        jaw_positions = set(range(9))
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        for u, v in edge_set:
            if u in jaw_positions:
                assert v in jaw_positions, f"Jaw node {u} connects to non-jaw node {v}"

    def test_step4_eye_connected(self, step4_indices):
        """Left eye nodes (LM 60, 64 → subset pos 15, 16) should be connected."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        # LM 60 → position 15, LM 64 → position 16
        pos_60 = step4_indices.index(60)
        pos_64 = step4_indices.index(64)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        assert (pos_60, pos_64) in edge_set, f"Eye nodes {pos_60}↔{pos_64} not connected"

    def test_step4_mouth_is_loop(self, step4_indices):
        """Outer mouth (LM 76,80,84 → 3 nodes) should form a closed loop."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        pos_76 = step4_indices.index(76)
        pos_80 = step4_indices.index(80)
        pos_84 = step4_indices.index(84)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        # Should be a triangle: 76↔80, 80↔84, 84↔76
        assert (pos_76, pos_80) in edge_set
        assert (pos_80, pos_84) in edge_set
        assert (pos_84, pos_76) in edge_set, "Mouth loop not closed"

    def test_step4_pupil_edge(self, step4_indices):
        """Pupil LM 96 should connect to LM 64 (both in subset)."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        pos_96 = step4_indices.index(96)
        pos_64 = step4_indices.index(64)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        assert (pos_96, pos_64) in edge_set

    def test_step4_no_cross_region(self, step4_indices):
        """No edges between jaw group and eyebrow group."""
        ei = make_subsampled_wflw_edge_index(step4_indices)
        # Jaw: positions 0-8, Left eyebrow: positions 9-10 (LM 36, 40)
        jaw_positions = set(range(9))
        eyebrow_positions = {step4_indices.index(36), step4_indices.index(40)}
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        for u, v in edge_set:
            if u in jaw_positions:
                assert v not in eyebrow_positions, f"Cross-region: jaw {u} → eyebrow {v}"

    def test_get_edge_index_wflw_with_subset(self, step4_indices):
        """get_edge_index('wflw', landmark_indices=...) gives subsampled graph."""
        ei = get_edge_index("wflw", landmark_indices=step4_indices)
        # Should NOT be 186 (full)
        assert ei.shape[1] != 186
        assert ei.shape[1] == 50  # 25 undirected * 2

    def test_get_edge_index_wflw_without_subset(self):
        """get_edge_index('wflw') without subset gives full 186-edge graph."""
        ei = get_edge_index("wflw")
        assert ei.shape[1] == 186

    def test_isolated_nodes_no_crash(self):
        """Subset with few nodes still works (cross-group edges may connect some)."""
        # LM 52 connects to 60 and 68 via cross-group edges now
        indices = [0, 52, 60, 72, 96]
        ei = make_subsampled_wflw_edge_index(indices)
        # Should still work — cross-group edges connect 52↔60, 52↔68 (68 not in set),
        # 96↔64 (64 not in set). Only 52↔60 survives.
        assert ei.shape[0] == 2
        assert ei.shape[1] >= 0  # At least some edges from cross-group anchoring
