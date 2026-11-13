"""Register a normalized card to the blank template card.

Detector corners cannot tell a card from the same card rotated 180 degrees,
and they place the card only to within the box's accuracy. Registering the
printed form (SIFT + RANSAC homography, ``card_ocr.preprocessing.FormAligner``)
against the blank template resolves both: the homography carries any flip and
the residual misalignment, and warping by it puts the card in the template's
exact coordinate frame, where the template's field fractions apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from card_ocr.preprocessing import FormAligner
from template_builder.normalize import rectify_to_square

# FormAligner's thresholds are tuned around a ~1000 px card; register at a
# fixed working size regardless of the output canvas.
_WORK_SIZE = 1024


@dataclass(frozen=True)
class AlignResult:
    image: Image.Image              # card warped into the reference frame
    rotation_degrees: float         # in-plane rotation that was removed
    diagnostics: dict


def reference_from_template_image(path: Path, canvas_size: int, *,
                                  fill=(255, 255, 255), decode_size: int = 4096) -> Image.Image:
    """Normalize a blank card image (already cropped to the card) to the square canvas."""
    with Image.open(path) as image:
        image.draft("RGB", (decode_size, decode_size))
        image.thumbnail((decode_size, decode_size), Image.Resampling.LANCZOS)
        card = image.convert("RGB")
    width, height = card.size
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], float)
    return rectify_to_square(card, corners, canvas_size, fill=tuple(fill)).image


class CardAligner:
    """Warp normalized cards onto a reference normalized card of the same size."""

    def __init__(self, reference: Image.Image, *, fill=(255, 255, 255)):
        if reference.width != reference.height:
            raise ValueError("reference must be a square normalized card")
        self.size = reference.width
        self.fill = tuple(fill)
        self._scale = _WORK_SIZE / self.size
        self._aligner = FormAligner(reference.convert("RGB").resize(
            (_WORK_SIZE, _WORK_SIZE), Image.Resampling.LANCZOS))

    @classmethod
    def from_template_image(cls, path: Path, canvas_size: int, *, fill=(255, 255, 255)):
        return cls(reference_from_template_image(path, canvas_size, fill=fill), fill=fill)

    def align(self, card: Image.Image) -> AlignResult:
        """Register ``card`` to the reference; raises ValueError if registration fails."""
        if card.size != (self.size, self.size):
            raise ValueError(f"card must be {self.size}x{self.size} to match the reference")
        work = card.convert("RGB").resize((_WORK_SIZE, _WORK_SIZE), Image.Resampling.LANCZOS)
        registration = self._aligner.register(work)            # raises ValueError
        scale = np.diag([self._scale, self._scale, 1.0])
        matrix = np.linalg.inv(scale) @ registration.card_to_template @ scale
        warped = cv2.warpPerspective(np.asarray(card.convert("RGB")), matrix,
                                     (self.size, self.size), flags=cv2.INTER_CUBIC,
                                     borderValue=self.fill)
        # Rotation removed = direction the card's top edge maps to.
        top = cv2.perspectiveTransform(np.float32([[[0, 0], [self.size, 0]]]), matrix)[0]
        rotation = float(np.degrees(np.arctan2(*(top[1] - top[0])[::-1])))
        return AlignResult(Image.fromarray(warped), rotation, registration.diagnostics)
