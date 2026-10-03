"""Memory-safe loading of very large ID card JPEGs.

The full-resolution pixel buffer is never materialized: we let the JPEG decoder
downscale during decode (``Image.draft``), then ``thumbnail`` to the exact fit,
and letterbox onto a square target canvas. Only the final preview and a few
scalars survive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# These are our own trusted, very large scans (~20k x 10k = ~200M px), well
# above Pillow's default decompression-bomb threshold. Disable the guard; we
# never materialize the full-resolution buffer anyway (draft + thumbnail).
Image.MAX_IMAGE_PIXELS = None


class ImageLoadError(Exception):
    """Raised when a source image cannot be decoded."""


@dataclass
class PreviewImage:
    preview: Image.Image            # RGB, target_res x target_res, letterboxed
    scale: float                    # uniform original -> preview scale
    offset: tuple[float, float]     # (ox, oy) letterbox padding, preview px
    orig_size: tuple[int, int]      # (orig_w, orig_h)


class ImageLoader:
    def __init__(self, target_resolution: int):
        if target_resolution <= 0:
            raise ValueError("target_resolution must be positive")
        self.target = int(target_resolution)

    def list_ids(self, data_dir: Path) -> list[str]:
        """Return sorted stems of ``*.jpg`` files (case-insensitive)."""
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            return []
        stems = {
            p.stem
            for p in data_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".jpg"
        }
        return sorted(stems)

    def load(self, path: Path) -> PreviewImage:
        path = Path(path)
        try:
            with Image.open(path) as img:
                orig_w, orig_h = img.size  # header only, no decode yet
                # Ask the JPEG decoder to downscale during decode.
                img.draft("RGB", (self.target, self.target))
                img = img.convert("RGB")
                img.thumbnail((self.target, self.target), Image.LANCZOS)
                thumb_w, thumb_h = img.size
                # Letterbox centered onto a square target canvas.
                canvas = Image.new("RGB", (self.target, self.target), (0, 0, 0))
                ox = (self.target - thumb_w) // 2
                oy = (self.target - thumb_h) // 2
                canvas.paste(img, (ox, oy))
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageLoadError(f"cannot decode {path}: {exc}") from exc

        scale = thumb_w / orig_w  # equals thumb_h / orig_h within rounding
        return PreviewImage(
            preview=canvas,
            scale=scale,
            offset=(float(ox), float(oy)),
            orig_size=(orig_w, orig_h),
        )
