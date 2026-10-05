"""Locate ID cards in scan previews using a cropped reference template.

Match SIFT features and estimate a homography to predict the card boundary.
Refine the detected edges and constrain the oriented box to the visible scan
area. Weak or geometrically implausible detections return no match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .obb import OBB

# Template resize width in pixels and minimum detection-quality thresholds.
_TEMPLATE_WIDTH = 260
_MAX_SIFT_FEATURES = 2500
_DESCRIPTOR_RATIO_THRESHOLD = 0.72
_RANSAC_REPROJECTION_THRESHOLD = 3.0  # Preview pixels.
_MIN_GOOD_MATCHES = 16
_MIN_INLIERS = 12
_MIN_INLIER_RATIO = 0.30

# Accepted card area as a fraction of the full, letterboxed preview image.
_MIN_CARD_AREA_RATIO = 0.015
_MAX_CARD_AREA_RATIO = 0.80
_MIN_CARD_ASPECT_RATIO = 0.45
_MAX_CARD_ASPECT_RATIO = 0.85

# Local edge sampling and contrast requirements in grayscale preview pixels.
_EDGE_SAMPLE_COUNT = 80
_EDGE_SAMPLE_HALF_SPAN = 0.40
_MIN_VISIBLE_EDGE_SAMPLES = 24
_MIN_SAMPLE_GRADIENT = 3.0
_MIN_EDGE_SUPPORT_RATIO = 0.65
_MIN_MEDIAN_EDGE_GRADIENT = 5.0
_EDGE_SEARCH_SIZE_RATIO = 0.05
_MIN_EDGE_SEARCH_RADIUS = 2
_MAX_EDGE_SEARCH_RADIUS = 16


def _scan_bounds(scene: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return inclusive nonblack content bounds as xmin, xmax, ymin, ymax.

    Entirely black outer rows and columns are treated as padding. Return None
    for an entirely black scene; this estimates content bounds, not card edges.
    """
    rows = np.flatnonzero(np.any(scene != 0, axis=1))
    columns = np.flatnonzero(np.any(scene != 0, axis=0))
    if not len(rows) or not len(columns):
        return None
    return int(columns[0]), int(columns[-1]), int(rows[0]), int(rows[-1])


def _rectangle_from_corners(corners: np.ndarray) -> OBB:
    """Fit a rectangle to preview corners ordered TL, TR, BR, BL.

    Combine opposite edge directions to estimate rotation, then average the
    projected side positions. Perspective distortion is approximated by an OBB.
    """
    points = np.asarray(corners, dtype=np.float64)
    width_direction = (points[1] - points[0]) + (points[2] - points[3])
    height_direction = (points[3] - points[0]) + (points[2] - points[1])
    width_direction += np.array([height_direction[1], -height_direction[0]])
    angle = float(np.arctan2(width_direction[1], width_direction[0]))
    width_axis = np.array([np.cos(angle), np.sin(angle)])
    height_axis = np.array([-width_axis[1], width_axis[0]])
    projected_x, projected_y = points @ width_axis, points @ height_axis
    left, right = projected_x[[0, 3]].mean(), projected_x[[1, 2]].mean()
    top, bottom = projected_y[[0, 1]].mean(), projected_y[[2, 3]].mean()
    center = width_axis * (left + right) / 2 + height_axis * (top + bottom) / 2
    return OBB(float(center[0]), float(center[1]), float(right-left),
               float(bottom-top), angle)


def _refine_card_edges(scene: np.ndarray, obb: OBB) -> OBB:
    """Snap nearby sides to a sustained light-card/darker-background boundary.

    Sample the middle of each side to exclude rounded corners. Median contrast
    rejects text and isolated marks; weak or off-image edges retain the estimate.
    Prefer the nearest sustained edge, excluding unrelated black padding edges.
    The search radius is approximately five percent of the shorter box side,
    clamped to 2–16 preview pixels. A card touching padding can use that boundary.
    """
    gray = cv2.GaussianBlur(scene.astype(np.float32), (3, 3), 0.7)
    bounds = _scan_bounds(scene)
    if bounds is None:
        return obb
    x_min, x_max, y_min, y_max = bounds
    width_axis = np.array([np.cos(obb.angle), np.sin(obb.angle)])
    height_axis = np.array([-width_axis[1], width_axis[0]])
    center = np.array([obb.cx, obb.cy], dtype=np.float64)
    radius = max(
        _MIN_EDGE_SEARCH_RADIUS,
        min(_MAX_EDGE_SEARCH_RADIUS, int(round(min(obb.w, obb.h) * _EDGE_SEARCH_SIZE_RATIO))),
    )
    offsets = np.arange(-radius-1, radius+2, dtype=np.float32)
    shifts = []
    for normal, tangent, distance, length in (
        (-width_axis, height_axis, obb.w/2, obb.h), (width_axis, height_axis, obb.w/2, obb.h),
        (-height_axis, width_axis, obb.h/2, obb.w), (height_axis, width_axis, obb.h/2, obb.w),
    ):
        boundary = center + normal * distance
        positions = (boundary + normal * offsets[:, None, None]
                     + tangent * np.linspace(
                         -length*_EDGE_SAMPLE_HALF_SPAN,
                         length*_EDGE_SAMPLE_HALF_SPAN,
                         _EDGE_SAMPLE_COUNT,
                     )[None, :, None])
        # Exclude unrelated padding edges, but allow a card already touching a
        # black strip to snap to its boundary (instead of remaining in the strip).
        low_x = 1 if normal[0] < -0.5 and boundary[0] <= x_min+3 else x_min+3
        high_x = scene.shape[1]-2 if normal[0] > 0.5 and boundary[0] >= x_max-3 else x_max-3
        low_y = 1 if normal[1] < -0.5 and boundary[1] <= y_min+3 else y_min+3
        high_y = scene.shape[0]-2 if normal[1] > 0.5 and boundary[1] >= y_max-3 else y_max-3
        valid = ((positions[:, :, 0] >= low_x) & (positions[:, :, 0] <= high_x)
                 & (positions[:, :, 1] >= low_y) & (positions[:, :, 1] <= high_y))
        samples = cv2.remap(gray, positions[:, :, 0].astype(np.float32),
                            positions[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        gradients = (samples[:-2] - samples[2:]) / 2
        usable = valid[:-2] & valid[2:]
        scores = np.full(len(gradients), -np.inf)
        for index, row in enumerate(gradients):
            values = row[usable[index]]
            # A tilted card may have only part of a side inside the scan. Require
            # at least 30 percent of the samples, with consistent local contrast.
            if (len(values) >= _MIN_VISIBLE_EDGE_SAMPLES
                    and np.mean(values >= _MIN_SAMPLE_GRADIENT) >= _MIN_EDGE_SUPPORT_RATIO):
                scores[index] = np.median(values)
        candidates = np.flatnonzero(scores >= _MIN_MEDIAN_EDGE_GRADIENT)
        shift = 0.0
        if len(candidates):
            # A blurred edge spans several samples: find each band's peak, then
            # take the nearest band rather than the strongest unrelated object.
            bands = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1)+1)
            peaks = [int(band[np.argmax(scores[band])]) for band in bands]
            best = min(peaks, key=lambda index: abs(float(offsets[index+1])))
            shift = float(offsets[best+1])
        shifts.append(shift)
    left, right, top, bottom = shifts
    center += width_axis * (right-left)/2 + height_axis * (bottom-top)/2
    return OBB(float(center[0]), float(center[1]), obb.w+left+right,
               obb.h+top+bottom, obb.angle)


def _clip_to_scan(scene: np.ndarray, obb: OBB) -> OBB | None:
    """Keep a detected rectangle inside visible scan bounds, preserving rotation.

    Template features can extrapolate a card edge beyond a clipped source scan.
    Trim the corresponding side rather than independently clamping corners,
    which would deform the rectangle and change its angle.
    Return None for an empty scene or a box trimmed below two pixels per side.
    """
    bounds = _scan_bounds(scene)
    if bounds is None:
        return None
    x_min, x_max, y_min, y_max = bounds
    low = np.array([x_min, y_min], dtype=float)
    high = np.array([x_max, y_max], dtype=float)
    width_axis = np.array([np.cos(obb.angle), np.sin(obb.angle)])
    height_axis = np.array([-width_axis[1], width_axis[0]])
    center = np.array([obb.cx, obb.cy], dtype=float)
    width, height = obb.w, obb.h
    for normal, indices, is_width in (
        (-width_axis, [0, 3], True), (width_axis, [1, 2], True),
        (-height_axis, [0, 1], False), (height_axis, [2, 3], False),
    ):
        corners = OBB(*center, width, height, obb.angle).corners()[indices]
        axis = int(np.argmax(np.abs(normal)))
        violation = (corners[:, axis].max()-high[axis] if normal[axis] > 0
                     else low[axis]-corners[:, axis].min())
        trim = max(0.0, float(violation)) / abs(normal[axis])
        center -= normal*trim/2
        if is_width:
            width -= trim
        else:
            height -= trim
    if width < 2 or height < 2:
        return None
    return OBB(float(center[0]), float(center[1]), width, height, obb.angle)


@dataclass(frozen=True)
class TemplateMatch:
    """Store a preview-space OBB and descriptor-match quality counts.

    Box positions and sizes use preview pixels; the angle uses radians.
    Inliers are matches accepted by RANSAC. These counts are not confidence
    percentages; good_matches counts descriptor matches passing the ratio test.
    """

    obb: OBB
    inliers: int
    good_matches: int


class CardTemplateMatcher:
    """Locate a card using cached SIFT template features and a homography.

    Refine the predicted edges and constrain the oriented box to scan content.
    Instances are used sequentially by the app's image-loading worker.
    """

    def __init__(self, template_path: Path):
        """Cache features from a template cropped to the physical card boundary.

        Resize while preserving aspect ratio. Raise ValueError for too few usable
        features; image-reading errors propagate to the caller.
        """
        template_path = Path(template_path)
        with Image.open(template_path) as image:
            width = _TEMPLATE_WIDTH
            height = round(image.height * width / image.width)
            template = image.resize(
                (width, height), Image.Resampling.LANCZOS
            ).convert("L")

        self._template = np.asarray(template)
        self._sift = cv2.SIFT_create(nfeatures=_MAX_SIFT_FEATURES)
        self._template_keypoints, self._template_descriptors = (
            self._sift.detectAndCompute(self._template, None)
        )
        if self._template_descriptors is None or len(self._template_keypoints) < _MIN_GOOD_MATCHES:
            raise ValueError(f"template has too few usable features: {template_path}")
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match(self, preview: Image.Image) -> TemplateMatch | None:
        """Return a refined detection in preview coordinates without editing pixels.

        Return None for insufficient matches, implausible geometry, or a box too
        small after clipping. OpenCV processing errors propagate to the caller.
        """
        scene = np.asarray(preview.convert("L"))
        keypoints, descriptors = self._sift.detectAndCompute(scene, None)
        if descriptors is None or len(keypoints) < _MIN_GOOD_MATCHES:
            return None

        pairs = self._matcher.knnMatch(self._template_descriptors, descriptors, k=2)
        # The nearest descriptor must be clearly better than its runner-up.
        good_matches = [
            pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < _DESCRIPTOR_RATIO_THRESHOLD * pair[1].distance
        ]
        if len(good_matches) < _MIN_GOOD_MATCHES:
            return None

        template_points = np.float32(
            [self._template_keypoints[item.queryIdx].pt for item in good_matches]
        ).reshape(-1, 1, 2)
        scene_points = np.float32(
            [keypoints[item.trainIdx].pt for item in good_matches]
        ).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(
            template_points, scene_points, cv2.RANSAC, _RANSAC_REPROJECTION_THRESHOLD
        )
        if homography is None or mask is None:
            return None

        inliers = int(mask.sum())
        if inliers < _MIN_INLIERS or inliers / len(good_matches) < _MIN_INLIER_RATIO:
            return None

        height, width = self._template.shape
        # Preserve template-relative TL, TR, BR, BL order through the projection.
        template_corners = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        ).reshape(-1, 1, 2)
        corners = cv2.perspectiveTransform(template_corners, homography).reshape(4, 2)
        if not np.isfinite(corners).all() or not cv2.isContourConvex(corners.astype(np.float32)):
            return None

        area = abs(float(cv2.contourArea(corners.astype(np.float32))))
        image_area = float(preview.width * preview.height)
        if not _MIN_CARD_AREA_RATIO <= area / image_area <= _MAX_CARD_AREA_RATIO:
            return None

        obb = _rectangle_from_corners(corners)
        obb = _refine_card_edges(scene, obb)
        obb = _clip_to_scan(scene, obb)
        if obb is None:
            return None
        aspect = obb.w / max(obb.h, 1e-6)
        if not _MIN_CARD_ASPECT_RATIO <= aspect <= _MAX_CARD_ASPECT_RATIO:
            return None

        return TemplateMatch(obb=obb, inliers=inliers, good_matches=len(good_matches))
