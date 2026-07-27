"""WFLW-specific evaluation metrics.

Implements NME (Normalized Mean Error), FR (Failure Rate), and AUC
(Area Under CED Curve) as used in the WFLW benchmark.
"""

from typing import List, Optional, Tuple

import numpy as np

# Inter-ocular distance landmarks (WFLW 98-point: outer eye corners)
IOD_LM_LEFT = 60
IOD_LM_RIGHT = 72

# Thresholds
FR_THRESHOLD = 0.10
AUC_THRESHOLD = 0.10
AUC_STEPS = 1000


def get_iod_indices_in_subset(landmark_indices: List[int]) -> Tuple[int, int]:
    """Find positions of IOD landmarks 60 and 72 within a sorted subset.

    Args:
        landmark_indices: Sorted list of original landmark indices in the subset.

    Returns:
        (pos_left, pos_right) — indices into the subset array where
        landmarks 60 and 72 are located.

    Raises:
        ValueError: If 60 or 72 is not in the subset.
    """
    sorted_indices = sorted(landmark_indices)
    if 60 not in sorted_indices:
        raise ValueError("IOD landmark 60 not found in subset")
    if 72 not in sorted_indices:
        raise ValueError("IOD landmark 72 not found in subset")
    return sorted_indices.index(60), sorted_indices.index(72)


def compute_nme(
    pred_px: np.ndarray,
    gt_px: np.ndarray,
    iod_left: int = IOD_LM_LEFT,
    iod_right: int = IOD_LM_RIGHT,
) -> Optional[float]:
    """Per-image NME normalized by inter-ocular distance.

    NME = mean(||pred_i - gt_i||) / IOD

    where IOD = ||gt[iod_left] - gt[iod_right]||.

    Args:
        pred_px: (N, 2) predicted landmarks in pixel space.
        gt_px: (N, 2) ground truth landmarks in pixel space.
        iod_left: Index of the left IOD landmark in the coordinate array.
        iod_right: Index of the right IOD landmark in the coordinate array.

    Returns:
        NME value, or None if inter-ocular distance is zero.
    """
    pred_px = np.asarray(pred_px, dtype=np.float64)
    gt_px = np.asarray(gt_px, dtype=np.float64)

    iod = float(np.linalg.norm(gt_px[iod_left] - gt_px[iod_right]))
    if iod <= 0:
        return None

    dists = np.linalg.norm(pred_px - gt_px, axis=1)
    return float(dists.mean() / iod)


def compute_fr(nme_list: list, threshold: float = FR_THRESHOLD) -> float:
    """Failure rate: fraction of samples with NME > threshold.

    Args:
        nme_list: List of per-sample NME values.
        threshold: NME threshold for failure.

    Returns:
        Failure rate in [0, 1].
    """
    if not nme_list:
        return 0.0
    nme_arr = np.array(nme_list)
    return float((nme_arr > threshold).mean())


def compute_auc(nme_list: list, threshold: float = AUC_THRESHOLD) -> float:
    """Area under the CED curve, normalized to [0, 1].

    CED(x) = fraction of samples with NME <= x.
    AUC = integral from 0 to threshold of CED(x) dx, divided by threshold.

    Args:
        nme_list: List of per-sample NME values.
        threshold: Maximum NME for AUC integration.

    Returns:
        AUC value in [0, 1].
    """
    if not nme_list:
        return 0.0
    nme_arr = np.array(nme_list)
    xs = np.linspace(0, threshold, AUC_STEPS + 1)
    ced = np.array([(nme_arr <= x).mean() for x in xs])
    # Use np.trapezoid (NumPy 2.x) with fallback to np.trapz (NumPy 1.x)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(ced, xs) / threshold)
