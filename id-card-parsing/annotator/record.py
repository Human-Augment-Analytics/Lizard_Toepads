"""Annotation record: the exported, reconstruction-ready sample for one image.

Stores the image id, target resolution, the forward/inverse affine transforms
(original <-> preview space), and the OBB in target-space coordinates as both
(cx, cy, w, h, angle) and ordered corner points. Everything needed to map the
box back to original pixels is self-contained.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AnnotationRecord:
    image_id: str
    target_resolution: int
    transform: list[list[float]]          # 2x3 original -> preview
    transform_inverse: list[list[float]]  # 2x3 preview -> original
    obb_xywhr: list[float]                # [cx, cy, w, h, angle] in target space
    obb_corners: list[list[float]]        # 4x2 in target space

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AnnotationRecord":
        return cls(**json.loads(s))


def save_record(record: AnnotationRecord, out_dir: Path) -> Path:
    """Write ``{image_id}.json`` into ``out_dir`` (created if needed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record.image_id}.json"
    path.write_text(record.to_json(), encoding="utf-8")
    return path


def load_record(image_id: str, out_dir: Path) -> AnnotationRecord | None:
    """Load ``{image_id}.json`` from ``out_dir``, or None if it does not exist."""
    path = Path(out_dir) / f"{image_id}.json"
    if not path.is_file():
        return None
    return AnnotationRecord.from_json(path.read_text(encoding="utf-8"))
