"""OBB (Oriented Bounding Box) cropping utilities for Lizard dataset.

Handles perspective-corrected cropping of detected toe/finger regions
from YOLO OBB detections.
"""

import numpy as np
import cv2

# Class mapping for YOLO OBB detections
TRAIN_CLASSES = [0, 1]
CLASSMAP = {0: "finger", 1: "toe"}


def order_box_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as [top-left, top-right, bottom-right, bottom-left].

    Args:
        pts: (4, 2) array of corner points.

    Returns:
        (4, 2) array with points ordered consistently.
    """
    pts = np.array(pts, dtype=np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    tl, tr = top[0], top[1]
    bl, br = bottom[0], bottom[1]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_obb_from_corners(image: np.ndarray, corners: np.ndarray):
    """Crop an oriented bounding box region using perspective transform.

    Args:
        image: Input image (HWC).
        corners: (4, 2) array of OBB corner points.

    Returns:
        Tuple of (cropped_image, perspective_matrix_M).
    """
    box = order_box_points(corners)
    width = int(max(1, round(np.linalg.norm(box[1] - box[0]))))
    height = int(max(1, round(np.linalg.norm(box[2] - box[1]))))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(image, M, (width, height))
    return crop, M


def transform_keypoints(kps: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Transform keypoints through a perspective matrix.

    Args:
        kps: (N, 2) keypoint coordinates in source space.
        M: (3, 3) perspective transform matrix.

    Returns:
        (N, 2) transformed keypoint coordinates.
    """
    kps = np.array(kps, dtype=np.float32).reshape(-1, 1, 2)
    local_kps = cv2.perspectiveTransform(kps, M)
    return local_kps.reshape(-1, 2)


def resize_and_pad(crop: np.ndarray, keypoints=None, max_size: int = 512):
    """Resize crop to fit in max_size x max_size with center padding.

    Returns uint8 image and transformed keypoints, plus scale/pad info
    for reversibility.

    Args:
        crop: Input crop image (HWC).
        keypoints: Optional (N, 2) keypoint coordinates in crop space.
        max_size: Target canvas size.

    Returns:
        If keypoints provided: (padded_image, transformed_kps, scale, pad_x, pad_y)
        If no keypoints: (padded_image, scale, pad_x, pad_y)
    """
    h, w = crop.shape[:2]
    scale = min(max_size / h, max_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(crop, (new_w, new_h))

    pad_x = (max_size - new_w) // 2
    pad_y = (max_size - new_h) // 2
    padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    if keypoints is not None:
        kps = np.array(keypoints, dtype=np.float32).copy()
        kps[:, 0] = kps[:, 0] * scale + pad_x
        kps[:, 1] = kps[:, 1] * scale + pad_y
        return padded, kps, scale, pad_x, pad_y

    return padded, scale, pad_x, pad_y
