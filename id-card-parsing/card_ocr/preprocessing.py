"""Align printed form features and mask printing while protecting added ink."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FormRegistration:
    """Map coarse card pixels into template coordinates, with fit diagnostics."""

    card_to_template: np.ndarray
    diagnostics: dict


class FormAligner:
    """Cache reference features and reject weak or implausible internal fits."""

    def __init__(self, template: Image.Image):
        """Extract reference form features once for reuse across cards."""
        self.size = template.size
        self.sift = cv2.SIFT_create(nfeatures=5000)
        gray = np.array(template.convert('L'))
        mask = np.zeros(gray.shape, np.uint8)
        height, width = gray.shape
        mask[round(height * .13):round(height * .96),
             round(width * .04):round(width * .96)] = 255
        self.keypoints, self.descriptors = self.sift.detectAndCompute(gray, mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def register(self, card: Image.Image) -> FormRegistration:
        """Estimate a card-to-template homography; reject weak or implausible fits."""
        if card.size != self.size:
            raise ValueError('Form registration requires matching card/template size')
        keypoints, descriptors = self.sift.detectAndCompute(np.array(card.convert('L')), None)
        if self.descriptors is None or descriptors is None or len(keypoints) < 20:
            raise ValueError('Insufficient printed form features for registration')
        matches = [pair[0] for pair in self.matcher.knnMatch(self.descriptors, descriptors, k=2)
                   if len(pair) == 2 and pair[0].distance < .70 * pair[1].distance]
        if len(matches) < 20:
            raise ValueError('Insufficient printed form matches for registration')
        source = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        target = np.float32([self.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        scale = self.size[0] / 1000
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, max(1, 3 * scale))
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            raise ValueError('Printed form registration failed')
        inliers = mask.ravel().astype(bool)
        count = int(inliers.sum())
        if count < 20 or count / len(matches) < .35:
            raise ValueError('Printed form registration has too few reliable inliers')
        coverage = cv2.contourArea(cv2.convexHull(target[inliers])) / np.prod(self.size)
        error = float(np.median(np.linalg.norm(
            cv2.perspectiveTransform(source[inliers], matrix) - target[inliers], axis=2)))
        width, height = self.size
        corners = np.float32([[[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]])
        projected = cv2.perspectiveTransform(corners, matrix)[0]
        area_ratio = cv2.contourArea(projected, oriented=True) / (width * height)
        if (coverage < .10 or error > max(1, 2 * scale)
                or not cv2.isContourConvex(projected)
                or not .65 <= area_ratio <= 1.50
                or np.linalg.norm(projected.mean(axis=0) - np.array(self.size)/2)
                > .15 * np.linalg.norm(self.size)):
            raise ValueError('Printed form registration has unreliable geometry or coverage')
        return FormRegistration(matrix, dict(
            matches=len(matches), inliers=count, inlier_ratio=count / len(matches),
            median_error_px=error, template_coverage=float(coverage),
            card_to_template=matrix.tolist()))


def clean_printed_form(card: Image.Image, template: Image.Image, *,
                       header_notes_box=None) -> tuple[Image.Image, Image.Image]:
    """Mask known print while preserving nearby added strokes and decimal points.

    Protect ink outside the print footprint and its immediate neighborhood.
    Overlapping print and handwriting may remain ambiguous; retain raw crops.
    The optional header box removes isolated ID fragments confined to the
    printed row, preserving connected handwriting extending below it.
    """
    if card.size != template.size:
        raise ValueError('Print removal requires matching card/template size')
    reference = np.array(template.convert('L'))
    observed = np.array(card.convert('L'))
    printed = np.uint8(reference < 145)
    printed[:round(card.height * .13)] = 0  # Keep specimen ID/header visible.
    tolerance = max(1, round(card.width * .002))
    support = cv2.dilate(printed, np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)) > 0
    added = np.uint8((observed < 155) & ~support)
    radius = max(2, round(card.width * .005))
    protected = cv2.dilate(added, np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
    remove = support & ~protected & (observed < 245)
    context_mask = remove.copy()
    if header_notes_box is not None:
        left, top, right, bottom = header_notes_box
        if not (0 <= left < right <= card.width and 0 <= top < bottom <= card.height):
            raise ValueError('Header notes box must be inside the card')
        # Only remove isolated ID fragments; notes may connect to the printed row.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.uint8(observed[:bottom] < 100))
        fragments = np.zeros(observed.shape, np.uint8)
        for label, (x, y, width, height, _area) in enumerate(stats[1:], 1):
            if (card.width * .30 < x < left < x + width < card.width * .68
                    and y < card.height * .07
                    and card.height * .03 < height
                    and y + height < card.height * .115):
                fragments[:bottom][labels == label] = 1
        fragments = cv2.dilate(fragments, np.ones((2*tolerance+1, 2*tolerance+1), np.uint8))
        context_mask |= fragments > 0
        remove[top:bottom, left:right] |= fragments[top:bottom, left:right] > 0
    source = np.array(card.convert('RGB'))
    cleaned = cv2.inpaint(source, np.uint8(context_mask | remove) * 255,
                          max(2, round(card.width * .003)), cv2.INPAINT_TELEA)
    cleaned[~remove] = source[~remove]
    return Image.fromarray(cleaned), Image.fromarray(np.uint8(remove) * 255)


def crop_warnings(crop: Image.Image) -> list[str]:
    """Flag dark ink near a crop boundary without deleting small marks."""
    ink = np.array(crop.convert('L')) < 120
    if (ink[:2].any() or ink[-2:].any() or ink[:, :2].any() or ink[:, -2:].any()):
        return ['Ink touches a field boundary; review the raw crop for clipping or neighboring text.']
    return []
