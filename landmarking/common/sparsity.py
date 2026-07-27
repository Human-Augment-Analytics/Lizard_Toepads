"""Landmark sparsity subsetting utilities.

Provides functions to compute uniform landmark subsets from the full
WFLW 98-point annotation, always forcing inclusion of IOD anchor
landmarks (60, 72) for inter-ocular distance normalization.
"""

from typing import List, Optional


def make_subset_indices(
    total: int = 98,
    step: int = 4,
    forced: Optional[List[int]] = None,
) -> List[int]:
    """Compute a uniform landmark subset with forced inclusions.

    Args:
        total: Total number of landmarks in the full set.
        step: Step size for uniform sampling (range(0, total, step)).
        forced: Indices that must appear in the output (default: [60, 72]).

    Returns:
        Sorted list of unique integer indices.

    Raises:
        ValueError: If step < 1 or step >= total.
    """
    if step < 1 or step >= total:
        raise ValueError(
            f"step must be in [1, {total - 1}], got {step}"
        )
    if forced is None:
        forced = [60, 72]
    return sorted(set(range(0, total, step)) | set(forced))
