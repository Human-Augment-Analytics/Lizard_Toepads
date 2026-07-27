"""Flip-aware mean shape utilities for GCN initialization.

When training with horizontal flip augmentation, the GCN's initial
coordinate estimate must match the flipped image. This module computes
the flipped mean shape (x-mirrored + landmark reordering) for use in
the training engine.
"""

import numpy as np
import torch

# Import the canonical flip pairs from the dataset topology module
# This ensures the mean shape flip uses the SAME permutation as the dataset
from ..datasets.wflw.topology import WFLW_FLIP_PAIRS, build_flip_permutation

# Pre-build the 98-point flip permutation using the canonical pairs
FLIP_PERM_98 = build_flip_permutation(WFLW_FLIP_PAIRS, 98)


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
