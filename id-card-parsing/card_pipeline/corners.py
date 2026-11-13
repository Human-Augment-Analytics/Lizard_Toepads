"""Reorder detector corners to the TL, TR, BR, BL convention rectification needs.

YOLO-OBB corner order depends on the box's rotation, not on image position.
``rectify_card`` requires TL, TR, BR, BL with positive oriented area (image
coordinates, y down), so corners are canonicalized before use.
"""

from __future__ import annotations

import cv2
import numpy as np

_MIN_AREA = 1.0


def canonicalize_corners(points, *, aspect: float | None = None) -> np.ndarray:
    """Return the quad as TL, TR, BR, BL with positive oriented area.

    ``aspect`` is the card's expected width / height (``None`` means landscape).
    The first edge (TL->TR) is the side matching that shape (long for a
    landscape card, short for a portrait one), running rightward, and of the
    two candidates the higher one on screen. Corners alone cannot distinguish a
    card from the same card rotated 180 degrees; the pipeline's form alignment
    resolves that. Degenerate, non-finite, or non-convex quads raise
    ``ValueError``.
    """
    quad = np.asarray(points, dtype=np.float64)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("corners must be a finite 4x2 array")
    if aspect is not None and not (np.isfinite(aspect) and aspect > 0):
        raise ValueError("aspect must be a positive finite width/height ratio")
    landscape = aspect is None or aspect >= 1

    # Walk the quad clockwise on screen (positive oriented area, y down).
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    ordered = quad[np.argsort(angles)]
    if cv2.contourArea(ordered.astype(np.float32), oriented=True) < 0:
        ordered = ordered[::-1]
    area = cv2.contourArea(ordered.astype(np.float32), oriented=True)
    if area <= _MIN_AREA or not cv2.isContourConvex(ordered.astype(np.float32)):
        raise ValueError("corners must form a convex quadrilateral with nonzero area")

    # Candidate starts: each rotation of the ring. Score the first edge: it
    # must be the longer pair of sides, tilted least from horizontal, and sit
    # above the opposite edge. Mildly rotated cards resolve unambiguously.
    best, best_score = None, None
    for shift in range(4):
        ring = np.roll(ordered, -shift, axis=0)
        top = ring[1] - ring[0]
        side = ring[3] - ring[0]
        top_len, side_len = np.hypot(*top), np.hypot(*side)
        tilt = abs(np.arctan2(top[1], top[0]))
        above = ring[:2].mean(axis=0)[1] <= ring[2:].mean(axis=0)[1]
        # The top edge has the card's shape: the long side for landscape, the
        # short side for portrait. Then prefer rightward, then higher.
        shape_ok = (top_len >= side_len - 1e-9) if landscape else (top_len <= side_len + 1e-9)
        score = (shape_ok, top[0] > 0, above, -tilt)
        if best_score is None or score > best_score:
            best, best_score = ring, score
    return best
