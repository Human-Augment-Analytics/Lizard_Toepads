"""Flip-aware mean shape utilities for GCN initialization.

When training with horizontal flip augmentation, the GCN's initial
coordinate estimate must match the flipped image. This module computes
the flipped mean shape (x-mirrored + landmark reordering) for use in
the training engine.
"""

import numpy as np
import torch

# WFLW 98-point flip pairs (same as in ref_transforms.py / graph_topology.py)
WFLW_FLIP_PAIRS = [
    [0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26],
    [7, 25], [8, 24], [9, 23], [10, 22], [11, 21], [12, 20], [13, 19],
    [14, 18], [15, 17],
    [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50],
    [39, 49], [40, 48], [41, 47],
    [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75],
    [66, 74], [67, 73],
    [55, 59], [56, 58],
    [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
    [88, 92], [89, 91], [95, 93], [96, 97],
]


def _build_flip_perm_98() -> np.ndarray:
    """Build the 98-point flip permutation array."""
    perm = np.arange(98, dtype=np.int64)
    for i, j in WFLW_FLIP_PAIRS:
        perm[i] = j
        perm[j] = i
    return perm


FLIP_PERM_98 = _build_flip_perm_98()


def compute_flipped_mean_shape(
    mean_shape: torch.Tensor,
    num_landmarks: int,
    landmark_indices: list = None,
) -> torch.Tensor:
    """Compute the horizontally-flipped mean shape for GCN initialization.

    For the full 98-point case: mirror x coords and reorder via FLIP_PERM_98.
    For subsets: we need to build a subset-specific flip permutation.

    Args:
        mean_shape: (N, 2) tensor of mean landmark positions in [0, 1].
        num_landmarks: Number of landmarks (len of mean_shape).
        landmark_indices: If subset is active, the original 98-pt indices.

    Returns:
        (N, 2) tensor of flipped mean shape positions.
    """
    flipped = mean_shape.clone()
    flipped[:, 0] = 1.0 - flipped[:, 0]  # Mirror x

    if landmark_indices is None or len(landmark_indices) == 98:
        # Full 98-point case: apply the standard flip permutation
        perm = torch.tensor(FLIP_PERM_98, dtype=torch.long, device=mean_shape.device)
        flipped = flipped[perm]
    else:
        # Subset case: build a permutation that swaps pairs within the subset.
        # For each flip pair (i, j) where BOTH i and j are in the subset,
        # swap their positions. Pairs where only one endpoint survives
        # are left in place (no swap partner).
        sorted_indices = sorted(landmark_indices)
        orig_to_pos = {orig: pos for pos, orig in enumerate(sorted_indices)}
        n = len(sorted_indices)
        perm = list(range(n))  # identity

        for i, j in WFLW_FLIP_PAIRS:
            if i in orig_to_pos and j in orig_to_pos:
                pos_i = orig_to_pos[i]
                pos_j = orig_to_pos[j]
                perm[pos_i] = pos_j
                perm[pos_j] = pos_i

        perm_tensor = torch.tensor(perm, dtype=torch.long, device=mean_shape.device)
        flipped = flipped[perm_tensor]

    return flipped
