"""Load a very large scan into a detection preview plus a bounded-size decode.

Scans can be ~200 megapixels, so neither detection nor rectification sees the
full-resolution buffer. Detection runs on a letterboxed square preview (the
same preview the annotator used for labeling, sized to the detector's training
resolution). Rectification warps from a larger but bounded decode, as
``card_ocr.prepare`` does. Corners found on the preview map to decode pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from annotator.geometry import apply_affine, build_transform, invert_affine
from annotator.image_loader import ImageLoader, ImageLoadError


@dataclass
class LoadedScan:
    preview: Image.Image                    # RGB, detect_size square, letterboxed
    decoded: Image.Image                    # RGB, at most decode_size on a side
    original_size: tuple[int, int]
    _preview_to_original: np.ndarray        # 2x3
    _decode_scale: tuple[float, float]      # original px -> decoded px

    def preview_to_decoded(self, points) -> np.ndarray:
        """Map Nx2 preview-space points to decoded-image pixels."""
        original = apply_affine(self._preview_to_original, points)
        return original * np.array(self._decode_scale)

    def preview_to_original(self, points) -> np.ndarray:
        """Map Nx2 preview-space points to original scan pixels."""
        return apply_affine(self._preview_to_original, points)


def load_scan(path: Path, *, detect_size: int, decode_size: int) -> LoadedScan:
    """Decode ``path`` into a detection preview and a bounded rectification image."""
    path = Path(path)
    preview = ImageLoader(detect_size).load(path)   # raises ImageLoadError
    try:
        with Image.open(path) as image:
            original_w, original_h = image.size
            image.draft("RGB", (decode_size, decode_size))   # cheap JPEG downscale
            image.thumbnail((decode_size, decode_size), Image.Resampling.LANCZOS)
            decoded = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError(f"cannot decode {path}: {exc}") from exc
    forward = build_transform(preview.scale, preview.offset)
    return LoadedScan(
        preview=preview.preview,
        decoded=decoded,
        original_size=(original_w, original_h),
        _preview_to_original=invert_affine(forward),
        _decode_scale=(decoded.width / original_w, decoded.height / original_h),
    )
