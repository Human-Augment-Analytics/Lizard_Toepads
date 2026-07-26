"""WFLW-specific topology configuration and flip pairs.

WFLW uses a 98-point facial landmark scheme with anatomically-defined
graph connectivity and horizontal flip pair mappings.
"""

import numpy as np

# Number of landmarks in WFLW
NUM_LANDMARKS = 98

# Graph topology name (used with get_edge_index)
TOPOLOGY_NAME = "wflw"

# Horizontal flip pairs for WFLW 98-point landmarks.
# Each tuple (i, j) means landmark i maps to j and j maps to i on horizontal flip.
WFLW_FLIP_PAIRS = [
    # Jaw contour: (0,32), (1,31), ..., (15,17); 16 is chin midline
    (0, 32), (1, 31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26),
    (7, 25), (8, 24), (9, 23), (10, 22), (11, 21), (12, 20), (13, 19),
    (14, 18), (15, 17), (16, 16),
    # Eyebrows: right (33-41) ↔ left (42-50)
    (33, 46), (34, 45), (35, 44), (36, 43), (37, 42),
    (38, 50), (39, 49), (40, 48), (41, 47),
    # Nose bridge: all midline
    (51, 51), (52, 52), (53, 53), (54, 54),
    # Nose base: (55,59), (56,58); 57 is midline
    (55, 59), (56, 58), (57, 57),
    # Eyes: right (60-67) ↔ left (68-75)
    (60, 68), (61, 69), (62, 70), (63, 71),
    (64, 72), (65, 73), (66, 74), (67, 75),
    # Outer mouth: (76,82), (77,81), (78,80); 79 is midline top
    #              (83,87), (84,86); 85 is midline bottom
    (76, 82), (77, 81), (78, 80), (79, 79),
    (83, 87), (84, 86), (85, 85),
    # Inner mouth: (88,92), (89,91); 90 is midline top
    #              (93,95); 94 is midline bottom
    (88, 92), (89, 91), (90, 90),
    (93, 95), (94, 94),
    # Pupils: right (96) ↔ left (97)
    (96, 97),
]


def build_flip_permutation(
    flip_pairs: list = None, num_landmarks: int = NUM_LANDMARKS
) -> np.ndarray:
    """Convert flip-pair list to a permutation index array.

    Returns perm such that flipped_coords = coords[perm] after mirroring x.

    Args:
        flip_pairs: List of (i, j) pairs. Defaults to WFLW_FLIP_PAIRS.
        num_landmarks: Total number of landmarks.

    Returns:
        (num_landmarks,) int64 array representing the permutation.
    """
    if flip_pairs is None:
        flip_pairs = WFLW_FLIP_PAIRS
    perm = np.arange(num_landmarks, dtype=np.int64)
    for i, j in flip_pairs:
        perm[i] = j
        perm[j] = i
    return perm


# Pre-built 98-point permutation
FLIP_PERM_98 = build_flip_permutation(WFLW_FLIP_PAIRS, 98)
