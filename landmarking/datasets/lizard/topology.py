"""Lizard-specific topology configuration.

Lizard uses a simple chain topology (0↔1↔2↔...↔8) with 9 landmarks.
Flip pairs are identity (no reordering needed for the chain).
"""

import numpy as np

# Number of landmarks in the Lizard dataset
NUM_LANDMARKS = 9

# Graph topology name (used with get_edge_index)
TOPOLOGY_NAME = "chain"

# Flip pairs for Lizard — identity permutation (no reordering on flip)
# Since the chain represents sequential toe-pad landmarks along a digit,
# horizontal flip just mirrors x coordinates without reordering.
LIZARD_FLIP_PAIRS = []


def get_flip_permutation(num_landmarks: int = NUM_LANDMARKS) -> np.ndarray:
    """Return identity permutation for Lizard landmarks.

    Lizard chain landmarks don't need reordering on horizontal flip.

    Returns:
        (num_landmarks,) identity permutation array.
    """
    return np.arange(num_landmarks, dtype=np.int64)
