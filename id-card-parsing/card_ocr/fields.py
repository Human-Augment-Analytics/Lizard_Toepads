"""Define editable field regions in normalized, upright template coordinates."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Field:
    """A text region as left, top, right, bottom fractions of the card size.

    These are expected writing areas, not detected handwriting boundaries.
    Margin expands horizontal/vertical card fractions before cropping.
    Pixel crops use Pillow's half-open rectangle convention.
    """

    name: str
    box: tuple[float, float, float, float]
    margin: tuple[float, float] = (0, 0)
    review_note: str | None = None

    def __post_init__(self):
        """Validate the field name, normalized bounds, margins, and review note."""
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"Invalid field name: {self.name}")
        if (len(self.box) != 4 or not all(math.isfinite(v) for v in self.box)
                or not 0 <= self.box[0] < self.box[2] <= 1
                or not 0 <= self.box[1] < self.box[3] <= 1):
            raise ValueError(f"Invalid normalized box for {self.name}: {self.box}")
        if len(self.margin) != 2 or any(not math.isfinite(v) or not 0 <= v <= .1 for v in self.margin):
            raise ValueError(f"Invalid horizontal/vertical margin for {self.name}: {self.margin}")
        if self.review_note is not None and not isinstance(self.review_note, str):
            raise ValueError(f'Invalid review note for {self.name}')

    def pixel_box(self, size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Round outward so a narrow region remains at least one pixel wide."""
        width, height = size
        left, top, right, bottom = self.box
        horizontal, vertical = self.margin
        return (math.floor(max(0, left-horizontal) * width), math.floor(max(0, top-vertical) * height),
                math.ceil(min(1, right+horizontal) * width), math.ceil(min(1, bottom+vertical) * height))


def load_layout(path: Path) -> list[Field]:
    """Read a layout, rejecting empty layouts, unsafe names, and duplicate fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = [Field(item["name"], tuple(float(v) for v in item["box"]),
                    tuple(float(v) for v in item.get("margin", (0, 0))), item.get('review_note'))
              for item in payload["fields"]]
    if not fields or len({field.name for field in fields}) != len(fields):
        raise ValueError("Layout must contain unique, nonempty fields")
    return fields


def load_checkboxes(path: Path) -> list[Field]:
    """Load optional checkbox regions without passing them to text OCR."""
    items = json.loads(Path(path).read_text(encoding='utf-8')).get('checkboxes', [])
    boxes = [Field(item['name'], tuple(item['box']), tuple(item.get('margin', (0, 0))))
             for item in items]
    if len({box.name for box in boxes}) != len(boxes):
        raise ValueError('Checkbox names must be unique')
    return boxes
