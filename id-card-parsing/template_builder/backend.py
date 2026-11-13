"""GUI-free orchestration for the template builder.

This is the seam the Tkinter UI and the tests both use. It loads a card image
into a memory-safe preview, maps a preview-space Card_OBB back to original
pixels, rectify-pads the card to a square Normalized_Card, and assembles/saves
a Template_Record. No Tkinter import lives here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from annotator.geometry import apply_affine, build_transform, invert_affine
from annotator.image_loader import ImageLoader, PreviewImage
from annotator.obb import OBB

from .config import TemplateBuilderConfig
from .normalize import NormalizeResult, rectify_to_square
from .template_record import FieldSpec, TemplateRecord, load_record, save_record


class TemplateBuilderBackend:
    def __init__(self, config: TemplateBuilderConfig):
        self.config = config
        self._loader = ImageLoader(config.preview_resolution)
        self._preview: PreviewImage | None = None
        self._original: "Image.Image | None" = None

    # --- image loading -----------------------------------------------------

    def load_image(self, path: Path) -> PreviewImage:
        """Load a downscaled preview and keep the full-res handle for rectify.

        Decode failures propagate as ``ImageLoadError`` for the UI to surface.
        The original image is opened lazily via PIL and only read during
        ``normalize``; it is kept so changing canvas size re-warps from source.
        """
        from PIL import Image

        path = Path(path)
        self._preview = self._loader.load(path)
        # Keep a lazy handle to the original for full-detail rectification.
        self._original = Image.open(path)
        return self._preview

    def preview_transform(self) -> np.ndarray:
        """Return the 2x3 affine mapping original pixels to preview coordinates."""
        if self._preview is None:
            raise RuntimeError("No image loaded")
        return build_transform(self._preview.scale, self._preview.offset)

    # --- geometry ----------------------------------------------------------

    def obb_to_original(self, obb: OBB) -> np.ndarray:
        """Map a preview-space OBB to original Card_Image pixel corners (TL..BL)."""
        m_inv = invert_affine(self.preview_transform())
        return apply_affine(m_inv, obb.corners())

    def normalize(self, corners_original) -> NormalizeResult:
        """Rectify-pad the card to a square using the current config."""
        if self._original is None:
            raise RuntimeError("No image loaded")
        return rectify_to_square(
            self._original, corners_original, self.config.canvas_size,
            pad_mode=self.config.pad_mode, fill=self.config.fill,
        )

    # --- config changes (re-normalize triggers) ---------------------------

    def set_canvas_size(self, size: int) -> None:
        self.config = dataclasses.replace(self.config, canvas_size=size)

    def set_pad_mode(self, pad_mode: str) -> None:
        self.config = dataclasses.replace(self.config, pad_mode=pad_mode)

    # --- record assembly and persistence ----------------------------------

    def build_record(self, corners_original, fields: list[FieldSpec]) -> TemplateRecord:
        """Assemble a Template_Record from current config and the given fields."""
        corners = np.asarray(corners_original, dtype=float).reshape(4, 2)
        return TemplateRecord(
            canvas_size=self.config.canvas_size,
            pad_mode=self.config.pad_mode,
            fill=tuple(int(c) for c in self.config.fill),
            card_obb_corners=[[float(x), float(y)] for x, y in corners],
            fields=list(fields),
        )

    def save(self, record: TemplateRecord) -> Path:
        """Validate and atomically write the record to the configured output path."""
        return save_record(record, self.config.output_path, overwrite=True)

    def existing_record(self) -> TemplateRecord | None:
        """Load a previously saved template from the output path, if any."""
        return load_record(self.config.output_path)
