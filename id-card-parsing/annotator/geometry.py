"""Affine geometry mapping original pixels <-> letterboxed preview space.

The forward transform is a uniform scale plus a letterbox translation:

    preview = scale * original + offset

represented as a 2x3 affine matrix ``[[s, 0, ox], [0, s, oy]]``.
"""

from __future__ import annotations

import numpy as np


def build_transform(scale: float, offset: tuple[float, float]) -> np.ndarray:
    """Return the 2x3 affine mapping original pixel coords to preview coords."""
    ox, oy = offset
    return np.array([[scale, 0.0, ox], [0.0, scale, oy]], dtype=np.float64)


def invert_affine(m: np.ndarray) -> np.ndarray:
    """Return the 2x3 inverse of a 2x3 affine matrix."""
    a = np.asarray(m, dtype=np.float64)
    linear = a[:, :2]
    translation = a[:, 2]
    inv_linear = np.linalg.inv(linear)
    inv_translation = -inv_linear @ translation
    return np.hstack([inv_linear, inv_translation.reshape(2, 1)])


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an (N, 2) array of points, returning (N, 2)."""
    a = np.asarray(m, dtype=np.float64)
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return p @ a[:, :2].T + a[:, 2]
