"""Reference HRNet transforms ported from HRNet-Facial-Landmark-Detection.

These functions are copied from `HRNet-Facial-Landmark-Detection/lib/utils/transforms.py`
(MIT License, Copyright (c) Microsoft, created by Tianheng Cheng and Yang Zhao).

Only the cv2-based `crop_v2` is included (NOT the scipy-dependent `crop()`),
since scipy.misc.imresize/imrotate were removed in SciPy 1.3 and np.math was
removed in NumPy 2.0.
"""

import cv2
import numpy as np
import torch


# ── Matched pairs for landmark flipping ──────────────────────────────────────

MATCHED_PARTS = {
    "WFLW": (
        [0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26],
        [7, 25], [8, 24], [9, 23], [10, 22], [11, 21], [12, 20], [13, 19],
        [14, 18], [15, 17],
        [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50],
        [39, 49], [40, 48], [41, 47],
        [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75],
        [66, 74], [67, 73],
        [55, 59], [56, 58],
        [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
        [88, 92], [89, 91], [95, 93], [96, 97],
    ),
}


def fliplr_joints(x, width, dataset='WFLW'):
    """Flip landmark coordinates horizontally and swap matched pairs.

    Args:
        x: (N, 2) numpy array of landmark coordinates in pixel space.
        width: Image width in pixels.
        dataset: Dataset name key into MATCHED_PARTS.

    Returns:
        (N, 2) numpy array with flipped coordinates.
    """
    matched_parts = MATCHED_PARTS[dataset]
    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    if dataset == 'WFLW':
        for pair in matched_parts:
            tmp = x[pair[0], :].copy()
            x[pair[0], :] = x[pair[1], :]
            x[pair[1], :] = tmp
    else:
        for pair in matched_parts:
            tmp = x[pair[0] - 1, :].copy()
            x[pair[0] - 1, :] = x[pair[1] - 1, :]
            x[pair[1] - 1, :] = tmp
    return x


def get_3rd_point(a, b):
    """Get the third point to define a unique affine transform from 2 points."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate a direction vector by rot_rad radians."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):
    """Compute the affine transformation matrix.

    Args:
        center: (2,) numpy array, center of the bounding box in pixel space.
        scale: scalar or (2,) array; scale factor(s) where face_size = scale * 200.
        rot: Rotation angle in degrees.
        output_size: [W, H] of output image.
        shift: (2,) shift vector (fraction of scale).
        inv: If 1, compute the inverse transform.

    Returns:
        (2, 3) affine transformation matrix.
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def crop_v2(img, center, scale, output_size, rot=0):
    """Crop and resize image using an affine transform (cv2-only).

    This is the cv2-only version that does NOT depend on scipy or np.math.

    Args:
        img: (H, W, C) or (H, W) numpy array.
        center: (2,) center point in pixel space.
        scale: Scalar or (2,) scale factor.
        output_size: [W, H] of output.
        rot: Rotation in degrees (default 0).

    Returns:
        Cropped/transformed image of shape (output_size[1], output_size[0], C).
    """
    trans = get_affine_transform(center, scale, rot, output_size)
    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR,
    )
    return dst_img


def get_transform(center, scale, output_size, rot=0):
    """Compute a 3x3 homogeneous transformation matrix for pixel mapping.

    Args:
        center: (2,) center of the face bounding box.
        scale: Scalar scale factor (face_size = scale * 200).
        output_size: [W, H] of the output space.
        rot: Rotation angle in degrees.

    Returns:
        (3, 3) numpy transformation matrix.
    """
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1] / 2
        t_mat[1, 2] = -output_size[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    """Transform a pixel coordinate between reference frames.

    Args:
        pt: (2,) point in source space.
        center: (2,) center of face box.
        scale: Scalar scale factor.
        output_size: [W, H] of target space.
        invert: If 1, apply inverse transform.
        rot: Rotation angle in degrees.

    Returns:
        (2,) transformed point (integer coordinates, 1-indexed).
    """
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def transform_preds(coords, center, scale, output_size):
    """Inverse-transform all predicted coordinates back to image space.

    Args:
        coords: (N, 2) tensor of coordinates in heatmap space.
        center: (2,) center of face box.
        scale: Scalar scale factor.
        output_size: [W, H] of the heatmap space.

    Returns:
        (N, 2) tensor of coordinates in image (512px) space.
    """
    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.tensor(
            transform_pixel(coords[p, 0:2], center, scale, output_size, 1, 0)
        )
    return coords


def generate_target(img, pt, sigma, label_type='Gaussian'):
    """Generate a truncated Gaussian heatmap target for one landmark.

    The Gaussian is truncated at 3*sigma. If the landmark falls outside the
    heatmap bounds, the heatmap is returned unchanged.

    Args:
        img: (H, W) numpy array — the heatmap to write into.
        pt: (2,) point coordinates in heatmap space (0-indexed).
        sigma: Standard deviation of the Gaussian.
        label_type: 'Gaussian' or 'Cauchy'.

    Returns:
        (H, W) numpy array with the Gaussian drawn.
    """
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    if label_type == 'Gaussian':
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img
