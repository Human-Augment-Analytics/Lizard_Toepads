#!/usr/bin/env python3
"""Generate OBB labels for bot_finger, bot_toe, ruler, and id from TPS landmark files.

Uses cv2.minAreaRect on biological landmarks (skipping first 2 ruler points)
to compute the minimum-area oriented bounding box, then adds configurable padding.
Ruler uses first 2 points from finger TPS; ID uses 2-point _id.TPS file.

Output: YOLO OBB format (class x1 y1 x2 y2 x3 y3 x4 y4, normalized)
  Raw bottom-view class IDs: 0=finger, 1=toe, 2=ruler, 3=id
  (Remapped to 6-class scheme by create_merged_obb_dataset.py)

Usage:
    python scripts/preprocessing/obb/generate_obb_from_tps.py --config configs/H8_obb_noflip.yaml
"""
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Raw bottom-view class IDs (before 6-class remapping by merge script)
BOT_FINGER_CLS = 0
BOT_TOE_CLS = 1
RULER_CLS = 2
ID_CLS = 3


def read_tps_landmarks(tps_path):
    """Read landmarks from TPS file. Returns list of (x, y) in TPS coords (bottom-left origin)."""
    pts = []
    with open(tps_path) as f:
        for line in f:
            line = line.strip()
            if re.match(r'^-?\d', line):
                parts = line.split()
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
    return pts


def tps_to_image_coords(pts, img_height):
    """Convert TPS (bottom-left origin) to image (top-left origin) coordinates."""
    return [(x, img_height - y) for x, y in pts]


def compute_obb(landmarks_img, img_width, img_height,
                long_padding_ratio=0.15, short_padding_ratio=2.5):
    """Compute minimum-area OBB from image-coordinate landmarks with asymmetric padding.

    Returns 4 corners as normalized [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] or None.
    """
    if len(landmarks_img) < 3:
        return None

    pts = np.array(landmarks_img, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect

    if w >= h:
        w *= (1.0 + long_padding_ratio)
        h *= (1.0 + short_padding_ratio)
    else:
        w *= (1.0 + short_padding_ratio)
        h *= (1.0 + long_padding_ratio)

    padded_rect = ((cx, cy), (w, h), angle)
    corners = cv2.boxPoints(padded_rect)

    norm = []
    for x, y in corners:
        norm.append((
            max(0.0, min(1.0, x / img_width)),
            max(0.0, min(1.0, y / img_height)),
        ))
    return norm


def bbox_to_obb_corners(x_min, y_min, x_max, y_max, img_width, img_height):
    """Convert axis-aligned bbox to 4 normalized OBB corners (TL, TR, BR, BL)."""
    return [
        (max(0.0, min(1.0, x_min / img_width)), max(0.0, min(1.0, y_min / img_height))),
        (max(0.0, min(1.0, x_max / img_width)), max(0.0, min(1.0, y_min / img_height))),
        (max(0.0, min(1.0, x_max / img_width)), max(0.0, min(1.0, y_max / img_height))),
        (max(0.0, min(1.0, x_min / img_width)), max(0.0, min(1.0, y_max / img_height))),
    ]


def compute_ruler_obb(ruler_pts_img, img_width, img_height, padding_ratio=0.1):
    """Compute axis-aligned OBB for ruler from 2 ruler points (image coords).

    Returns 4 normalized corners or None.
    """
    if len(ruler_pts_img) < 2:
        return None
    x_coords = [p[0] for p in ruler_pts_img]
    y_coords = [p[1] for p in ruler_pts_img]
    rw = max(x_coords) - min(x_coords)
    rh = max(y_coords) - min(y_coords)
    pad_x = max(5, rw * padding_ratio)
    pad_y = max(5, rh * padding_ratio)
    x_min = max(0, min(x_coords) - pad_x)
    x_max = min(img_width, max(x_coords) + pad_x)
    y_min = max(0, min(y_coords) - pad_y)
    y_max = min(img_height, max(y_coords) + pad_y)
    return bbox_to_obb_corners(x_min, y_min, x_max, y_max, img_width, img_height)


def compute_id_obb(id_pts_tps, img_width, img_height, padding_ratio=0.05):
    """Compute axis-aligned OBB for ID card from 2 TPS points (top-left, bottom-right).

    Points are in TPS coords (bottom-left origin), converted internally.
    Returns 4 normalized corners or None.
    """
    if len(id_pts_tps) < 2:
        return None
    # TPS coords: bottom-left origin → image coords: top-left origin
    tl_x, tl_y = id_pts_tps[0]
    br_x, br_y = id_pts_tps[1]
    tl_y = img_height - tl_y
    br_y = img_height - br_y
    bw = abs(br_x - tl_x)
    bh = abs(tl_y - br_y)
    pad_x = max(5, bw * padding_ratio)
    pad_y = max(5, bh * padding_ratio)
    x_min = max(0, min(tl_x, br_x) - pad_x)
    x_max = min(img_width, max(tl_x, br_x) + pad_x)
    y_min = max(0, min(tl_y, br_y) - pad_y)
    y_max = min(img_height, max(tl_y, br_y) + pad_y)
    return bbox_to_obb_corners(x_min, y_min, x_max, y_max, img_width, img_height)


def process_image(img_stem, tps_dir, image_dir):
    """Process one image: read TPS files, compute OBBs for all 4 bottom classes."""
    finger_tps = os.path.join(tps_dir, f"{img_stem}_finger.TPS")
    toe_tps = os.path.join(tps_dir, f"{img_stem}_toe.TPS")
    id_tps = os.path.join(tps_dir, f"{img_stem}_id.TPS")
    img_path = None
    for ext in ('.jpg', '.jpeg', '.png'):
        candidate = os.path.join(image_dir, f"{img_stem}{ext}")
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        return []

    with Image.open(img_path) as img:
        img_width, img_height = img.size

    lines = []

    # bot_finger (class 2)
    if os.path.exists(finger_tps):
        pts_tps = read_tps_landmarks(finger_tps)
        bio_pts = pts_tps[2:]  # skip first 2 ruler points
        bio_img = tps_to_image_coords(bio_pts, img_height)
        corners = compute_obb(bio_img, img_width, img_height)
        if corners:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
            lines.append(f"{BOT_FINGER_CLS} {coords}")

        # ruler (class 4) — first 2 points of finger TPS
        ruler_pts = pts_tps[:2]
        ruler_img = tps_to_image_coords(ruler_pts, img_height)
        ruler_corners = compute_ruler_obb(ruler_img, img_width, img_height)
        if ruler_corners:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in ruler_corners)
            lines.append(f"{RULER_CLS} {coords}")

    # bot_toe (class 3)
    if os.path.exists(toe_tps):
        pts_tps = read_tps_landmarks(toe_tps)
        bio_pts = pts_tps[2:]  # skip first 2 ruler points
        bio_img = tps_to_image_coords(bio_pts, img_height)
        corners = compute_obb(bio_img, img_width, img_height)
        if corners:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
            lines.append(f"{BOT_TOE_CLS} {coords}")

    # id (class 5) — 2-point _id.TPS (top-left, bottom-right)
    if os.path.exists(id_tps):
        id_pts = read_tps_landmarks(id_tps)
        id_corners = compute_id_obb(id_pts, img_width, img_height)
        if id_corners:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in id_corners)
            lines.append(f"{ID_CLS} {coords}")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate OBB labels from TPS files")
    parser.add_argument("--config", required=True,
                        help="Path to project YAML config (e.g. configs/H7_obb_6class.yaml)")
    parser.add_argument("--tps-dir", help="Directory with TPS landmark files")
    parser.add_argument("--image-dir", help="Directory with source images")
    parser.add_argument("--output-dir", help="Output directory for OBB labels")
    parser.add_argument("--long-padding-ratio", type=float, default=0.15,
                        help="Padding ratio for long axis (default: 0.15)")
    parser.add_argument("--short-padding-ratio", type=float, default=2.5,
                        help="Padding ratio for short axis (default: 2.5)")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Number of images to visualize (0=none)")
    args = parser.parse_args()

    # Load config
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    prep = cfg.get("preprocessing", {})

    # CLI overrides > config > defaults
    tps_dir = args.tps_dir or prep.get("tps-dir", "/storage/ice-shared/cs8903onl/tps_files")
    image_dir = args.image_dir or prep.get("image-dir", "/storage/ice-shared/cs8903onl/miami_fall_24_jpgs")
    output_dir = args.output_dir or prep.get("obb-output-dir", "data/processed_obb/labels")

    print(f"Config:     {args.config}")
    print(f"  tps-dir:    {tps_dir}")
    print(f"  image-dir:  {image_dir}")
    print(f"  output-dir: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Find all image stems that have at least one TPS file
    tps_files = os.listdir(tps_dir)
    stems = set()
    for f in tps_files:
        if f.endswith('.TPS'):
            stem = f.replace('_finger.TPS', '').replace('_toe.TPS', '').replace('_id.TPS', '')
            stems.add(stem)

    stems = sorted(stems)
    print(f"Found {len(stems)} image stems with TPS files")

    count = 0
    for stem in stems:
        lines = process_image(stem, tps_dir, image_dir)
        if lines:
            label_path = os.path.join(output_dir, f"{stem}.txt")
            with open(label_path, 'w') as f:
                f.write("\n".join(lines) + "\n")
            count += 1

    print(f"Generated OBB labels for {count} images in {output_dir}")

    # Optional visualization
    if args.visualize > 0:
        import random
        random.seed(42)
        vis_stems = random.sample(stems[:count], min(args.visualize, count))
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        for stem in vis_stems:
            label_path = os.path.join(output_dir, f"{stem}.txt")
            img_path = os.path.join(image_dir, f"{stem}.jpg")
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue

            img_bgr = cv2.imread(img_path)
            h, w = img_bgr.shape[:2]

            with open(label_path) as lf:
                for line in lf:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    corners = np.array([
                        [coords[0] * w, coords[1] * h],
                        [coords[2] * w, coords[3] * h],
                        [coords[4] * w, coords[5] * h],
                        [coords[6] * w, coords[7] * h],
                    ], dtype=np.int32)
                    cls_colors = {
                        BOT_FINGER_CLS: ((255, 0, 255), "bot_finger"),
                        BOT_TOE_CLS: ((0, 255, 0), "bot_toe"),
                        RULER_CLS: ((255, 0, 0), "ruler"),
                        ID_CLS: ((0, 165, 255), "id"),
                    }
                    color, label = cls_colors.get(cls_id, ((255, 255, 255), f"cls_{cls_id}"))
                    cv2.polylines(img_bgr, [corners], True, color, thickness=4)
                    cx_vis = int(np.mean(corners[:, 0]))
                    cy_vis = int(np.min(corners[:, 1])) - 15
                    cv2.putText(img_bgr, label, (cx_vis - 60, cy_vis),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            scale = 2000 / max(h, w)
            out = cv2.resize(img_bgr, None, fx=scale, fy=scale)
            vis_path = os.path.join(vis_dir, f"{stem}_obb.jpg")
            cv2.imwrite(vis_path, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Visualization: {vis_path}")


if __name__ == "__main__":
    main()
