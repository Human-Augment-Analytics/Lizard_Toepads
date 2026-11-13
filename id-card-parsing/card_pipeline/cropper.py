"""Crop template regions from a normalized (square) card image.

Fractions follow the ``card_ocr.fields.Field.pixel_box`` convention: floor the
left/top edge and ceil the right/bottom edge so a thin box keeps at least one
pixel, using Pillow's half-open rectangle.
"""

from __future__ import annotations

import math

from PIL import Image

from template_builder.template_record import FieldSpec, TemplateRecord


def crop_region(card: Image.Image, box) -> Image.Image:
    """Crop ``[left, top, right, bottom]`` fractions from ``card``.

    Raises ``ValueError`` for malformed boxes, boxes outside 0..1, or a card
    too small to produce any pixel.
    """
    box = tuple(box)
    if (len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise ValueError(f"Invalid normalized box: {box!r}")
    width, height = card.size
    left, top = math.floor(box[0] * width), math.floor(box[1] * height)
    right, bottom = math.ceil(box[2] * width), math.ceil(box[3] * height)
    right, bottom = min(right, width), min(bottom, height)
    if right <= left or bottom <= top:
        raise ValueError(f"Box {box!r} has no pixels on a {width}x{height} card")
    return card.crop((left, top, right, bottom))


def crop_template(card: Image.Image, record: TemplateRecord
                  ) -> list[tuple[FieldSpec, Image.Image | None, str | None]]:
    """Crop every template region in template order.

    Each entry is ``(spec, crop, error)``: exactly one of crop/error is set, so
    one bad region never aborts the others. Field and checkbox regions are both
    cropped; routing is the caller's job.
    """
    results = []
    for spec in record.fields:
        try:
            results.append((spec, crop_region(card, spec.box), None))
        except ValueError as exc:
            results.append((spec, None, str(exc)))
    return results
