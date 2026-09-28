"""Lizard-specific topology configuration.

The Lizard landmarks are bilateral pairs running up the length of the digit
plus a single tip:

    0---1   base pair
    2---3
    4---5
    6---7   top pair
      8     tip

Landmark identities are LATERAL: even indices (0,2,4,6) lie on one side of the
digit, odd indices (1,3,5,7) on the other. A horizontal flip therefore SWAPS the
two members of every pair; mirroring x without relabeling would silently corrupt
the labels (a left-side point would carry a right-side identity). The flip pairs
below encode that swap so horizontal-flip augmentation is label-correct.
"""

import numpy as np

# Number of landmarks in the Lizard dataset
NUM_LANDMARKS = 9

# Graph topology name (used with get_edge_index). "lizard_ladder" encodes the
# bilateral pair (rung) + per-side (rail) structure above; "chain" remains valid
# for backward comparison.
TOPOLOGY_NAME = "lizard_ladder"

# Flip pairs for Lizard — bilateral partners swap on horizontal flip; the tip (8)
# maps to itself. Each tuple (i, j) means i<->j under a horizontal mirror.
LIZARD_FLIP_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def get_flip_permutation(num_landmarks: int = NUM_LANDMARKS) -> np.ndarray:
    """Return the horizontal-flip relabeling permutation for Lizard landmarks.

    Swaps each bilateral pair (0<->1, 2<->3, 4<->5, 6<->7); the tip is fixed.

    Args:
        num_landmarks: Number of landmarks (default 9).

    Returns:
        (num_landmarks,) int64 permutation array to reindex coords after a flip.
    """
    perm = np.arange(num_landmarks, dtype=np.int64)
    for i, j in LIZARD_FLIP_PAIRS:
        if i < num_landmarks and j < num_landmarks:
            perm[i] = j
            perm[j] = i
    return perm
