"""Rectify a hand-cropped card OBB and resize-pad it onto a square canvas.

The existing ``card_ocr.prepare.rectify_card`` warps four ordered corners
(TL, TR, BR, BL) to an upright image of a chosen size. Here we pick that size
from the card's own edge lengths (so the card keeps its true aspect ratio),
then resize-fit and center-pad onto a square ``canvas_size`` so every card
downstream shares one coordinate space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from card_ocr.prepare import rectify_card


@dataclass(frozen=True)
class NormalizeResult:
    """A normalized card plus the geometry used to produce it.

    ``scale`` maps rectified (upright, pre-pad) pixels to canvas pixels.
    ``pad_offset`` is the centered letterbox padding in canvas pixels.
    ``rect_size`` is the rectified card size before padding; ``rect_size_on_canvas``
    is that size after the resize-fit (i.e. the un-padded content size on the
    canvas).
    """

    image: Image.Image                      # Normalized_Card, canvas_size x canvas_size, RGB
    scale: float                            # rectified px -> canvas px (uniform)
    pad_offset: tuple[int, int]             # (ox, oy) centered padding, canvas px
    rect_size: tuple[int, int]              # rectified (pre-pad) card size, px
    rect_size_on_canvas: tuple[int, int]    # content size after resize-fit, canvas px


def _upright_size(corners: np.ndarray) -> tuple[int, int]:
    """Estimate the upright card size from OBB edge lengths (TL, TR, BR, BL)."""
    tl, tr, br, bl = corners
    width = (np.hypot(*(tr - tl)) + np.hypot(*(br - bl))) / 2.0
    height = (np.hypot(*(bl - tl)) + np.hypot(*(br - tr))) / 2.0
    return max(2, int(round(width))), max(2, int(round(height)))


def rectify_to_square(image: Image.Image, corners, canvas_size: int, *,
                      pad_mode: str = "solid", fill: tuple[int, int, int] = (255, 255, 255)) -> NormalizeResult:
    """Warp the card to upright, then resize-fit and center-pad to a square.

    ``corners`` is a 4x2 array of the card's corners in image pixels, ordered
    TL, TR, BR, BL. Degenerate quads (wrong count, non-finite, non-convex, or
    near-zero area) raise ``ValueError`` via ``rectify_card``.
    """
    if not isinstance(canvas_size, int) or canvas_size < 2:
        raise ValueError("canvas_size must be an integer >= 2")
    if pad_mode != "solid":
        raise ValueError("only pad_mode='solid' is supported")
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError("corners must be a 4x2 array ordered TL, TR, BR, BL")

    rect_w, rect_h = _upright_size(corners)
    # rectify_card validates convexity/area/finiteness and raises ValueError.
    upright, _inverse = rectify_card(image, corners, (rect_w, rect_h))

    scale = min(canvas_size / rect_w, canvas_size / rect_h)
    new_w = max(1, min(canvas_size, int(round(rect_w * scale))))
    new_h = max(1, min(canvas_size, int(round(rect_h * scale))))
    resized = upright.resize((new_w, new_h), Image.Resampling.LANCZOS)

    ox = (canvas_size - new_w) // 2
    oy = (canvas_size - new_h) // 2
    canvas = Image.new("RGB", (canvas_size, canvas_size), tuple(fill))
    canvas.paste(resized, (ox, oy))

    return NormalizeResult(
        image=canvas,
        scale=float(scale),
        pad_offset=(ox, oy),
        rect_size=(rect_w, rect_h),
        rect_size_on_canvas=(new_w, new_h),
    )
