"""Lizard-specific evaluation metrics.

Provides pixel error computation, pixel-to-mm conversion, and
back-projection from 512px canvas to original image coordinates.
"""

import numpy as np


def compute_pixel_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-landmark Euclidean distance in pixel space.

    Args:
        pred: (num_landmarks, 2) predicted coordinates in pixel space.
        gt: (num_landmarks, 2) ground truth coordinates in pixel space.

    Returns:
        (num_landmarks,) array of per-landmark Euclidean distances.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return np.linalg.norm(pred - gt, axis=1)


def pixel_to_mm(
    pixel_error: np.ndarray, ruler_px: float, ruler_mm: float = 10.0
) -> np.ndarray:
    """Convert pixel error to millimeters using ruler calibration.

    Args:
        pixel_error: Array of pixel distances.
        ruler_px: Ruler distance in pixels (from TPS ruler landmarks).
        ruler_mm: Physical ruler length in mm (default 10mm).

    Returns:
        Array of distances in millimeters.

    Raises:
        ValueError: If ruler_px is zero or negative.
    """
    if ruler_px <= 0:
        raise ValueError(f"ruler_px must be positive, got {ruler_px}")
    mm_per_px = ruler_mm / ruler_px
    return np.asarray(pixel_error, dtype=np.float64) * mm_per_px


def back_project(
    coords_512: np.ndarray,
    M: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
) -> np.ndarray:
    """Undo letterbox + perspective transform to get original image coordinates.

    Reverses the preprocessing pipeline:
      1. Undo letterbox padding and scaling: (coord - pad) / scale
      2. Invert perspective transform M: homogeneous multiply + perspective divide

    Args:
        coords_512: (N, 2) coordinates in the 512×512 canvas space.
        M: (3, 3) perspective transform matrix (from getPerspectiveTransform).
        scale: Resize scale factor used during letterboxing.
        pad_x: Horizontal padding offset.
        pad_y: Vertical padding offset.

    Returns:
        (N, 2) coordinates in original image space.
    """
    coords_512 = np.asarray(coords_512, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Undo letterbox: (coord - pad) / scale → OBB crop space
    coords_raw = coords_512.copy()
    coords_raw[:, 0] = (coords_512[:, 0] - pad_x) / scale
    coords_raw[:, 1] = (coords_512[:, 1] - pad_y) / scale

    # Invert the perspective transform
    M_inv = np.linalg.inv(M)

    # Homogeneous multiply + perspective divide → original image space
    ones = np.ones((coords_raw.shape[0], 1), dtype=np.float64)
    coords_h = np.hstack([coords_raw, ones])  # (N, 3)
    proj = coords_h @ M_inv.T  # (N, 3)
    global_pts = proj[:, :2] / proj[:, 2:3]  # perspective divide

    return global_pts
