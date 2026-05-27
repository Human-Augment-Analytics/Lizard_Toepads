#!/usr/bin/env python3
"""Visualize OBB annotations from a YOLO OBB dataset.

Draws oriented bounding boxes on images for specified classes.
Useful for sanity-checking dataset labels before training.

Usage:
    python scripts/visualization/visualize_obb_labels.py
    python scripts/visualization/visualize_obb_labels.py --dataset data/dataset_obb_6class_active --stems 1001 1003 --classes 2 3 4 5
    python scripts/visualization/visualize_obb_labels.py --stems 1001 --classes 2 3 4 5  # all classes on one image
"""
import argparse
import os
import cv2
import numpy as np
from pathlib import Path

CLASSES = {0: 'up_finger', 1: 'up_toe', 2: 'bot_finger', 3: 'bot_toe', 4: 'ruler', 5: 'id'}
COLORS  = {0: (255,165,0), 1: (0,255,255), 2: (255,0,255), 3: (0,255,0), 4: (0,165,255), 5: (255,255,0)}


def draw_obb_labels(img, label_path, target_classes=None):
    """Draw OBB boxes for the specified classes onto img. Returns annotated copy."""
    h, w = img.shape[:2]
    vis = img.copy()
    thickness = max(4, w // 500)
    font_scale = max(1.2, w / 1500)

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if target_classes is not None and cls_id not in target_classes:
                continue
            coords = [float(x) for x in parts[1:]]
            if len(coords) != 8:
                continue
            corners = np.array([
                [coords[0] * w, coords[1] * h],
                [coords[2] * w, coords[3] * h],
                [coords[4] * w, coords[5] * h],
                [coords[6] * w, coords[7] * h],
            ], dtype=np.int32)
            color = COLORS.get(cls_id, (255, 255, 255))
            cv2.polylines(vis, [corners], True, color, thickness=thickness)
            label = CLASSES.get(cls_id, str(cls_id))
            cx = int(np.mean(corners[:, 0]))
            cy = int(np.min(corners[:, 1])) - 20
            cv2.putText(vis, label, (cx - 60, max(cy, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
    return vis


def main():
    project_root = os.environ.get("PROJECT_ROOT", "/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads")
    parser = argparse.ArgumentParser(description="Visualize YOLO OBB labels on images")
    parser.add_argument("--dataset", default=f"{project_root}/data/dataset_obb_6class_active",
                        help="Path to YOLO dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Dataset split to use")
    parser.add_argument("--stems", nargs="+", default=["1001", "1003"],
                        help="Image stems to visualize")
    parser.add_argument("--classes", nargs="+", type=int, default=None,
                        help="Class IDs to draw (default: all)")
    parser.add_argument("--output-dir", default=f"{project_root}/data/visualizations/obb_labels",
                        help="Output directory for visualizations")
    parser.add_argument("--width", type=int, default=1500,
                        help="Output image width in pixels")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    img_dir = dataset / "images" / args.split
    lbl_dir = dataset / "labels" / args.split
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_classes = set(args.classes) if args.classes else None

    for stem in args.stems:
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"  Image not found for stem: {stem}")
            continue

        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            print(f"  Label not found: {lbl_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read image: {img_path}")
            continue

        vis = draw_obb_labels(img, lbl_path, target_classes)

        scale = args.width / img.shape[1]
        out = cv2.resize(vis, None, fx=scale, fy=scale)

        cls_suffix = "_".join(CLASSES.get(c, str(c)) for c in sorted(target_classes)) if target_classes else "all"
        out_path = out_dir / f"{stem}_{cls_suffix}.jpg"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
