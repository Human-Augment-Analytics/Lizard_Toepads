"""Measure the added ink in a checkbox using the aligned blank form."""

import cv2
import numpy as np
from PIL import Image

from .fields import Field

def checkbox_pixel_box(box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return the saved checkbox rectangle in aligned-card pixels."""
    return Field("checkbox", tuple(box)).pixel_box(size)


def inspect_checkbox(observed: Image.Image, template: Image.Image) -> dict:
    """Inspect the entire supplied crop and report marked, empty, or uncertain."""
    if observed.size != template.size or min(observed.size) < 1:
        raise ValueError('Checkbox images must have matching, nonempty sizes')
    printed = _ink_mask(template)
    observed_ink = _ink_mask(observed)
    # Allow for small registration errors around the printed brackets.
    tolerance = max(1, round(min(observed.size) * .06))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    support = cv2.dilate(printed.astype(np.uint8), kernel) > 0
    ink = np.uint8(observed_ink & ~support)

    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    area = ink.size
    fraction = float(ink.sum() / area)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    if fraction >= .015 and largest >= max(3, .008 * area):
        state, checked = 'marked', True
    elif largest <= max(1, .003 * area) and fraction < .015:
        state, checked = 'empty', False
    else:
        state, checked = 'uncertain', None
    return dict(state=state, checked=checked, added_ink_fraction=fraction,
                largest_component_pixels=largest, reviewed=False,
                warning='Heuristic checkbox state; confirm against the raw crop.')


def _ink_mask(image: Image.Image) -> np.ndarray:
    """Find pixels that differ from the local paper in lightness or color."""
    lab = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2LAB)
    lab = cv2.GaussianBlur(lab, (3, 3), .6).astype(np.float32)
    # Estimate paper from the lighter pixels to allow for tint and exposure.
    lightness = lab[:, :, 0]
    paper = lab[lightness >= np.percentile(lightness, 70)]
    relative = lab - np.median(paper, axis=0)
    color_difference = np.hypot(relative[:, :, 1], relative[:, :, 2])
    return (relative[:, :, 0] < -30) | (color_difference >= 18)
