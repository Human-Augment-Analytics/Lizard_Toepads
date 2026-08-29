"""Cephalometric-specific evaluation metrics.

Provides normalized-to-pixel mapping, per-landmark radial error in
millimeters, and the field-standard Mean Radial Error (MRE) plus
Successful Detection Rate (SDR) at 2 / 2.5 / 3 / 4 mm thresholds.

All functions are pure numpy, mirroring ``metrics_lizard.py``.
"""

import numpy as np


def to_original_pixels(coords_norm: np.ndarray, orig_size) -> np.ndarray:
    """Map normalized [0,1] coordinates to original-image pixel space.

    Args:
        coords_norm: (N, 2) coordinates normalized to [0, 1] as (x, y).
        orig_size: iterable [H, W] giving the original image height and width
            in pixels (note the [H, W] order).

    Returns:
        (N, 2) pixel coordinates where x_px = x * W and y_px = y * H.
    """
    coords_norm = np.asarray(coords_norm, dtype=np.float64)
    orig_size = np.asarray(orig_size, dtype=np.float64).reshape(-1)
    H = float(orig_size[0])
    W = float(orig_size[1])
    px = np.empty_like(coords_norm)
    px[:, 0] = coords_norm[:, 0] * W
    px[:, 1] = coords_norm[:, 1] * H
    return px


def compute_radial_error_mm(
    pred_norm: np.ndarray,
    gt_norm: np.ndarray,
    orig_size,
    pixel_spacing: float,
) -> np.ndarray:
    """Per-landmark radial error in millimeters.

    Maps both predicted and ground-truth normalized coordinates back to
    original pixel space via ``orig_size``, computes the per-landmark
    Euclidean pixel distance, and converts to mm using ``pixel_spacing``.

    Args:
        pred_norm: (N, 2) predicted coordinates normalized to [0, 1].
        gt_norm: (N, 2) ground-truth coordinates normalized to [0, 1].
        orig_size: iterable [H, W] of the source image size in pixels.
        pixel_spacing: physical size of one pixel in millimeters.

    Returns:
        (N,) array of per-landmark radial errors in millimeters.
    """
    pred_px = to_original_pixels(pred_norm, orig_size)
    gt_px = to_original_pixels(gt_norm, orig_size)
    px_err = np.linalg.norm(pred_px - gt_px, axis=1)
    return px_err * float(pixel_spacing)


def compute_mre_sdr(radial_errors_mm, thresholds=(2.0, 2.5, 3.0, 4.0)) -> dict:
    """Mean Radial Error and Successful Detection Rate.

    Args:
        radial_errors_mm: 1-D array or list of per-landmark radial errors in
            millimeters, flattened across all landmarks and samples.
        thresholds: SDR thresholds in millimeters.

    Returns:
        Dict with keys:
            "mre": mean radial error in mm (float) or None if empty.
            "std": standard deviation of radial error in mm (float) or None.
            "sdr": dict mapping "<t>mm" -> percentage in [0, 100] of errors
                <= threshold (float), or None per-threshold if empty.
            "n": number of errors considered (int).
    """
    errors = np.asarray(radial_errors_mm, dtype=np.float64).reshape(-1)
    n = int(errors.size)

    if n == 0:
        return {
            "mre": None,
            "std": None,
            "sdr": {f"{t}mm": None for t in thresholds},
            "n": 0,
        }

    sdr = {}
    for t in thresholds:
        sdr[f"{t}mm"] = float(np.mean(errors <= t) * 100.0)

    return {
        "mre": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "sdr": sdr,
        "n": n,
    }
