"""GUI-free orchestration: the seam the UI and later preprocessing both use."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AnnotatorConfig
from .geometry import build_transform, invert_affine
from .image_loader import ImageLoader, PreviewImage
from .obb import OBB
from .record import AnnotationRecord, load_record, save_record


class AnnotatorBackend:
    def __init__(self, config: AnnotatorConfig):
        self.config = config
        self._loader = ImageLoader(config.target_resolution)

    def image_ids(self) -> list[str]:
        return self._loader.list_ids(self.config.data_dir)

    def open(self, image_id: str) -> tuple[PreviewImage, np.ndarray]:
        """Load a preview and its original->preview transform."""
        preview = self._loader.load(self.config.data_dir / f"{image_id}.jpg")
        m = build_transform(preview.scale, preview.offset)
        return preview, m

    def existing_record(self, image_id: str) -> AnnotationRecord | None:
        return load_record(image_id, self.config.out_dir)

    def export(self, image_id: str, preview: PreviewImage, obb: OBB) -> Path:
        """Assemble and persist an AnnotationRecord for one image.

        Corner coordinates are clamped into ``[0, target_resolution]`` so the
        exported box never leaves the canvas; the stored ``obb_xywhr`` is then
        derived from the clamped corners so both representations agree. For an
        axis-aligned box this clamp is exact; for a rotated box the clamped
        quad may deform slightly at an edge that ran off-canvas.
        """
        target = self.config.target_resolution
        m = build_transform(preview.scale, preview.offset)
        m_inv = invert_affine(m)

        corners = np.clip(obb.corners(), 0.0, float(target))
        clamped = OBB.from_corners(corners)

        record = AnnotationRecord(
            image_id=image_id,
            target_resolution=target,
            transform=m.tolist(),
            transform_inverse=m_inv.tolist(),
            obb_xywhr=[clamped.cx, clamped.cy, clamped.w, clamped.h, clamped.angle],
            obb_corners=corners.tolist(),
        )
        return save_record(record, self.config.out_dir)
