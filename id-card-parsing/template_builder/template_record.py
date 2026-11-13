"""The Template_Record: one template's normalized geometry and field specs.

A ``FieldSpec`` mirrors ``card_ocr.fields.Field``'s name pattern and normalized
``[left, top, right, bottom]`` box convention so the exported layout binds to
the existing OCR pipeline without reshaping. Each field also carries a datatype
(the extraction engine's value kinds) and an optional recognition dictionary,
stored verbatim for a later OCR stage to resolve.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Subset of card_ocr.extraction value kinds (Req 6.2).
VALID_DATATYPES = ("text", "integer", "number", "time")

# How a crop is consumed downstream. "field" goes to text OCR; "checkbox" is a
# non-OCR crop (checkbox / pixel-intensity readout) handled separately.
VALID_CROP_TYPES = ("field", "checkbox")

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class FieldSpec:
    """One crop region in normalized canvas fractions, with downstream metadata.

    ``crop_type`` selects how the crop is used: ``"field"`` (default) routes to
    text OCR and carries a ``datatype``/``dictionary``; ``"checkbox"`` is a
    non-OCR crop where ``datatype``/``dictionary`` are not meaningful.
    """

    name: str
    box: tuple[float, float, float, float]
    datatype: str = "text"
    dictionary: str | None = None
    crop_type: str = "field"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"Invalid field name: {self.name!r}")
        box = tuple(self.box)
        if (len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= 1
                or not 0 <= box[1] < box[3] <= 1):
            raise ValueError(f"Invalid normalized box for {self.name}: {self.box!r}")
        # Freeze box as a float tuple regardless of input sequence type.
        object.__setattr__(self, "box", tuple(float(v) for v in box))
        if self.crop_type not in VALID_CROP_TYPES:
            raise ValueError(f"Invalid crop_type for {self.name}: {self.crop_type!r}")
        # A datatype only constrains OCR fields; checkbox crops ignore it.
        if self.crop_type == "field" and self.datatype not in VALID_DATATYPES:
            raise ValueError(f"Invalid datatype for {self.name}: {self.datatype!r}")
        if self.dictionary is not None and not isinstance(self.dictionary, str):
            raise ValueError(f"dictionary must be a string or None for {self.name}")

    @property
    def is_field(self) -> bool:
        return self.crop_type == "field"

    @property
    def is_checkbox(self) -> bool:
        return self.crop_type == "checkbox"


@dataclass(frozen=True)
class TemplateRecord:
    """A complete field template for one normalized card."""

    canvas_size: int
    pad_mode: str
    fill: tuple[int, int, int]
    card_obb_corners: list[list[float]]     # 4x2, original Card_Image pixels, TL,TR,BR,BL
    fields: list[FieldSpec] = field(default_factory=list)

    def validate(self) -> None:
        """Reject an unusable template before saving."""
        if not isinstance(self.canvas_size, int) or self.canvas_size < 2:
            raise ValueError("canvas_size must be an integer >= 2")
        if self.pad_mode != "solid":
            raise ValueError("pad_mode must be 'solid'")
        if (len(self.fill) != 3
                or any(not isinstance(c, int) or not 0 <= c <= 255 for c in self.fill)):
            raise ValueError("fill must be three integers in 0..255")
        corners = self.card_obb_corners
        if len(corners) != 4 or any(len(point) != 2 for point in corners):
            raise ValueError("card_obb_corners must be four [x, y] points")
        if not self.fields:
            raise ValueError("Template must contain at least one field")
        names = [spec.name for spec in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("Field names must be unique within the template")
        # FieldSpec.__post_init__ already validated each field individually.

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "TemplateRecord":
        payload = json.loads(text)
        fields = [FieldSpec(name=item["name"], box=tuple(item["box"]),
                            datatype=item.get("datatype", "text"),
                            dictionary=item.get("dictionary"),
                            crop_type=item.get("crop_type", "field"))  # backfill legacy
                  for item in payload["fields"]]
        return cls(
            canvas_size=int(payload["canvas_size"]),
            pad_mode=payload["pad_mode"],
            fill=tuple(payload["fill"]),
            card_obb_corners=[[float(x), float(y)] for x, y in payload["card_obb_corners"]],
            fields=fields,
        )


def save_record(record: TemplateRecord, path: Path, *, overwrite: bool = True) -> Path:
    """Validate then write the record atomically (temp file, then replace).

    Create parent directories as needed. With ``overwrite=False``, refuse to
    replace an existing file. The destination is never left partially written.
    """
    record.validate()
    path = Path(path)
    if not overwrite and path.exists():
        raise FileExistsError(f"Template already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.to_json()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.stem}.", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp_path, path)
        else:
            os.link(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return path


def load_record(path: Path) -> TemplateRecord | None:
    """Load a Template_Record from ``path``, or None if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    return TemplateRecord.from_json(path.read_text(encoding="utf-8"))
