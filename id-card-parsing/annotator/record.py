"""Annotation record: the exported, reconstruction-ready sample for one image.

Stores the image id, target resolution, the forward/inverse affine transforms
(original <-> preview space), and the OBB in target-space coordinates as both
(cx, cy, w, h, angle) and ordered corner points. Everything needed to map the
box back to original pixels is self-contained.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AnnotationRecord:
    """Store an image's annotation geometry and coordinate transforms.

    Includes the image ID, target resolution, forward and inverse affine
    transforms, and oriented box coordinates and corners.
    """
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


def save_record(record: AnnotationRecord, out_dir: Path, *, overwrite=True, cancel=None) -> Path:
    """Save a complete image_id JSON record without exposing a partially written file.

    Write and flush a temporary file in the destination directory, then save
    it atomically. Replace an existing annotation when overwrite=True; otherwise,
    raise FileExistsError if the destination already exists.

    If cancellation is observed before saving, raise InterruptedError.
    Always remove the temporary file and return the destination path on success.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record.image_id}.json"
    payload = record.to_json()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_dir,
                                         prefix=f".{record.image_id}.", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Save cancelled before committing the file")
        if overwrite:
            os.replace(temp_path, path)
        else:
            # Prevent overwrite
            os.link(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return path


def load_record(image_id: str, out_dir: Path) -> AnnotationRecord | None:
    """Load ``{image_id}.json`` from ``out_dir``, or None if it does not exist."""
    path = Path(out_dir) / f"{image_id}.json"
    if not path.is_file():
        return None
    return AnnotationRecord.from_json(path.read_text(encoding="utf-8"))
