"""Unit tests for the eyes-only experiment configuration.

Verifies:
- Subsampled WFLW graph preserves both eye loops + pupil edges
- Flip permutation correctly swaps right eye ↔ left eye landmarks
- IOD landmarks (60, 72) are at correct positions in subset
- No isolated nodes
"""

import numpy as np
import pytest

from landmarking.common.graph_topologies import make_subsampled_wflw_edge_index
from landmarking.datasets.wflw.dataset import WFLWDataset
from landmarking.evaluation.metrics_wflw import get_iod_indices_in_subset


EYES_INDICES = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 96, 97]


class TestEyesOnlyGraph:
    """Verify the eyes-only subsampled graph."""

    def test_edge_count(self):
        """Should have 18 undirected edges: 8 (right loop) + 8 (left loop) + 2 (pupils)."""
        ei = make_subsampled_wflw_edge_index(EYES_INDICES)
        n_undirected = ei.shape[1] // 2
        assert n_undirected == 18, f"Expected 18 undirected edges, got {n_undirected}"

    def test_right_eye_is_loop(self):
        """Right eye (subset positions 0-7) should form a closed loop."""
        ei = make_subsampled_wflw_edge_index(EYES_INDICES)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        # Chain: 0↔1↔2↔3↔4↔5↔6↔7
        for j in range(7):
            assert (j, j + 1) in edge_set, f"Missing right eye edge {j}↔{j+1}"
        # Closing edge: 7↔0
        assert (7, 0) in edge_set, "Right eye loop not closed (7→0)"
        assert (0, 7) in edge_set, "Right eye loop not closed (0→7)"

    def test_left_eye_is_loop(self):
        """Left eye (subset positions 8-15) should form a closed loop."""
        ei = make_subsampled_wflw_edge_index(EYES_INDICES)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        # Chain: 8↔9↔10↔11↔12↔13↔14↔15
        for j in range(8, 15):
            assert (j, j + 1) in edge_set, f"Missing left eye edge {j}↔{j+1}"
        # Closing edge: 15↔8
        assert (15, 8) in edge_set, "Left eye loop not closed (15→8)"
        assert (8, 15) in edge_set, "Left eye loop not closed (8→15)"

    def test_pupil_edges(self):
        """Pupil nodes connect to eye centers: 96→64 (pos 16→4), 97→72 (pos 17→12)."""
        ei = make_subsampled_wflw_edge_index(EYES_INDICES)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        # LM 96 → position 16, LM 64 → position 4
        assert (16, 4) in edge_set, "Pupil 96 not connected to eye center 64"
        # LM 97 → position 17, LM 72 → position 12
        assert (17, 12) in edge_set, "Pupil 97 not connected to eye center 72"

    def test_no_cross_eye_edges(self):
        """Right eye nodes (0-7) should NOT directly connect to left eye nodes (8-15)."""
        ei = make_subsampled_wflw_edge_index(EYES_INDICES)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i].item(), ei[1, i].item()))
        right_eye = set(range(8))
        left_eye = set(range(8, 16))
        for u, v in edge_set:
            if u in right_eye and v in left_eye:
                assert False, f"Cross-eye edge found: {u}→{v}"
            if u in left_eye and v in right_eye:
                assert False, f"Cross-eye edge found: {u}→{v}"

    def test_indices_in_range(self):
        """All edge indices should be in [0, 17]."""
        ei = make_subsampled_wflw_edge_index(EYES_INDICES)
        assert ei.min().item() >= 0
        assert ei.max().item() <= 17

    def test_no_isolated_nodes(self):
        """Every node should have at least one edge."""
        ei = make_subsampled_wflw_edge_index(EYES_INDICES)
        connected = set()
        for i in range(ei.shape[1]):
            connected.add(ei[0, i].item())
            connected.add(ei[1, i].item())
        assert len(connected) == 18, f"Only {len(connected)}/18 nodes connected"


class TestEyesOnlyFlipPermutation:
    """Verify flip augmentation correctly swaps right↔left eyes."""

    def test_flip_perm_swaps_eyes(self):
        """Right eye positions (0-7) should swap with left eye (8-15)."""
        ds = WFLWDataset(
            pt_paths=[],
            landmark_indices=EYES_INDICES,
            augment=True,
        )
        perm = ds.flip_perm
        # Right eye (pos 0-7) should map to left eye (pos 8-15)
        # LM 60↔68 → pos 0↔8
        assert perm[0] == 8, f"Pos 0 (LM60) should swap to pos 8 (LM68), got {perm[0]}"
        assert perm[8] == 0, f"Pos 8 (LM68) should swap to pos 0 (LM60), got {perm[8]}"
        # LM 64↔72 → pos 4↔12
        assert perm[4] == 12, f"Pos 4 (LM64) should swap to pos 12 (LM72), got {perm[4]}"
        assert perm[12] == 4, f"Pos 12 (LM72) should swap to pos 4 (LM64), got {perm[12]}"
        # LM 67↔75 → pos 7↔15
        assert perm[7] == 15, f"Pos 7 (LM67) should swap to pos 15 (LM75), got {perm[7]}"
        assert perm[15] == 7, f"Pos 15 (LM75) should swap to pos 7 (LM67), got {perm[15]}"
        # Pupils: LM 96↔97 → pos 16↔17
        assert perm[16] == 17, f"Pos 16 (LM96) should swap to pos 17 (LM97), got {perm[16]}"
        assert perm[17] == 16, f"Pos 17 (LM97) should swap to pos 16 (LM96), got {perm[17]}"

    def test_flip_perm_is_involution(self):
        """Applying the permutation twice returns identity (swap is its own inverse)."""
        ds = WFLWDataset(
            pt_paths=[],
            landmark_indices=EYES_INDICES,
            augment=True,
        )
        perm = ds.flip_perm
        # perm[perm[i]] == i for all i
        for i in range(18):
            assert perm[perm[i]] == i, f"perm[perm[{i}]] = {perm[perm[i]]}, expected {i}"

    def test_all_positions_have_partner(self):
        """Every position in the eyes subset has a flip partner (no orphans)."""
        ds = WFLWDataset(
            pt_paths=[],
            landmark_indices=EYES_INDICES,
            augment=True,
        )
        perm = ds.flip_perm
        # No position should map to itself (no midline landmarks in eyes)
        for i in range(18):
            assert perm[i] != i, f"Position {i} maps to itself (orphan)"


class TestEyesOnlyIOD:
    """Verify IOD landmarks are at correct positions."""

    def test_iod_positions(self):
        """LM 60 at position 0, LM 72 at position 12."""
        left, right = get_iod_indices_in_subset(EYES_INDICES)
        assert left == 0, f"LM 60 should be at position 0, got {left}"
        assert right == 12, f"LM 72 should be at position 12, got {right}"
