"""Measure added checkbox ink separately from handwriting recognition.

States are heuristic review suggestions, not validated categorical labels or
calibrated probabilities. Printed brackets are excluded using the aligned form.
"""

import cv2
import numpy as np
from PIL import Image


def inspect_checkbox(observed: Image.Image, template: Image.Image) -> dict:
    """Report empty, marked, or uncertain using extra dark ink and its extent."""
    if observed.size != template.size:
        raise ValueError('Checkbox images must have matching sizes')
    reference = np.array(template.convert('L')) < 145
    tolerance = max(1, round(observed.width * .03))
    support = cv2.dilate(reference.astype('uint8'),
                         np.ones((2*tolerance+1, 2*tolerance+1), 'uint8')) > 0
    ink = np.uint8((np.array(observed.convert('L')) < 155) & ~support)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    area = ink.size
    fraction = float(ink.sum() / area)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    state = ('marked' if fraction >= .015 and largest / area >= .005 else
             'empty' if fraction <= .003 else 'uncertain')
    return dict(state=state, added_ink_fraction=fraction,
                largest_component_pixels=largest, reviewed=False,
                warning='Heuristic checkbox state; confirm against the raw crop.')
