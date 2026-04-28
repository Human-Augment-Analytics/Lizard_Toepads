import sys
import os
import numpy as np
import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt

TRAIN_CLASSES = [0, 1]
CLASSMAP = {0: "finger", 1: "toe"}

DEBUG_CROP_TARGET = 10


def order_box_points(pts):
    pts = np.array(pts, dtype=np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    tl, tr = top[0], top[1]
    bl, br = bottom[0], bottom[1]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_obb_from_corners(image, corners):
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


def transform_keypoints(kps, M):
    kps = np.array(kps, dtype=np.float32).reshape(-1, 1, 2)
    local_kps = cv2.perspectiveTransform(kps, M)
    return local_kps.reshape(-1, 2)


def crop_toe_boxes_obb(result, image, g_coords, imgid, debug_state=None, target_classes=None, expected_points=9):
    if target_classes is None:
        target_classes = TRAIN_CLASSES

    crops, tps_local = [], []
    stats = {
        "total_target_boxes": 0,
        "kept": 0,
        "dropped_missing_tps": 0,
        "dropped_bad_count": 0,
        "dropped_out_of_bounds": 0,
    }

    if result.obb is None:
        return crops, tps_local, stats, []

    obb_corners = result.obb.xyxyxyxy.cpu().numpy()
    cls_ids = result.obb.cls.cpu().numpy()
    Ms = []
    for det_idx, (corners, cls_id) in enumerate(zip(obb_corners, cls_ids)):
        cls_id = int(cls_id)
        if cls_id not in target_classes:
            continue

        stats["total_target_boxes"] += 1
        class_name = CLASSMAP[cls_id]
        global_kps = g_coords.get(class_name, [])
        if len(global_kps) == 0:
            stats["dropped_missing_tps"] += 1
            continue
        if len(global_kps) != expected_points:
            stats["dropped_bad_count"] += 1
            continue

        crop, M = crop_obb_from_corners(image, corners)
        local_kps = transform_keypoints(global_kps, M)
        h_crop, w_crop = crop.shape[:2]
        in_bounds = (local_kps[:, 0] >= 0) & (local_kps[:, 0] < w_crop) & \
                    (local_kps[:, 1] >= 0) & (local_kps[:, 1] < h_crop)

        is_kept = bool(np.all(in_bounds))
        if not is_kept:
            stats["dropped_out_of_bounds"] += 1
        else:
            crops.append(crop)
            tps_local.append(local_kps.astype(np.float32))
            Ms.append(M)
            stats["kept"] += 1

        if debug_state is not None and debug_state["saved"] < DEBUG_CROP_TARGET:
            vis = crop.copy()
            for kp_idx, (x, y) in enumerate(local_kps):
                color = (0, 255, 0) if in_bounds[kp_idx] else (0, 0, 255)
                cv2.circle(vis, (int(round(x)), int(round(y))), 4, color, -1)
                cv2.putText(
                    vis,
                    str(kp_idx),
                    (int(round(x)) + 5, int(round(y)) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            status = "kept" if is_kept else "drop"
            debug_path = f"{debug_state.get('save_dir', '.')}/debug_{debug_state['saved']:02d}_{imgid}_{det_idx}_{status}.jpg"
            cv2.imwrite(debug_path, vis)

            if not debug_state.get("displayed", False):
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                plt.title(f"Debug OBB overlay: {status} ({imgid})")
                plt.axis("off")
                plt.show()
                debug_state["displayed"] = True

            debug_state["saved"] += 1

    return crops, tps_local, stats, Ms


def reversible_rescale(img, keypoints=None, max_size=512):
    h, w = img.shape[:2]
    scale = min(max_size / h, max_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_x = (max_size - new_w) // 2
    pad_y = (max_size - new_h) // 2
    padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    padded = padded.astype(np.float32) / 255.0
    padded = (padded - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    if keypoints is None:
        return padded, scale, pad_x, pad_y

    kps = np.array(keypoints, dtype=np.float32).copy()
    kps[:, 0] = kps[:, 0] * scale + pad_x
    kps[:, 1] = kps[:, 1] * scale + pad_y

    return padded, kps, scale, pad_x, pad_y
