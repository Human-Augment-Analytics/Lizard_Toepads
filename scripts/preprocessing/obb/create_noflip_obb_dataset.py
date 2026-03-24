#!/usr/bin/env python3
"""
Create a 6-class OBB dataset without upper-view (flip) annotations.

Keeps the 6-class structure for consistency with H5 detect model, but:
  - Class 0 (up_finger): empty (no annotations)
  - Class 1 (up_toe):    empty (no annotations)
  - Class 2 (bot_finger): from merged 6-class OBB dataset
  - Class 3 (bot_toe):    from merged 6-class OBB dataset
  - Class 4 (ruler):      from merged 6-class OBB dataset
  - Class 5 (id):         from merged 6-class OBB dataset

This dataset is used to train YOLO-OBB for fair comparison against the
H5 baseline detect model (which also doesn't use flip strategy).

Usage:
    python scripts/preprocessing/obb/create_noflip_obb_dataset.py --config configs/H8_obb_botonly.yaml
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


# Keep only bottom-view classes (drop upper-view classes 0, 1)
KEEP_CLASSES = {2, 3, 4, 5}


def filter_labels(src_label_dir: Path, dst_label_dir: Path):
    """Filter 6-class labels to keep only bottom-view classes."""
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0
    total_with_labels = 0

    for label_file in sorted(src_label_dir.glob("*.txt")):
        kept_lines = []

        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in KEEP_CLASSES:
                    kept_lines.append(line.strip())

        total_files += 1
        if kept_lines:
            total_with_labels += 1

        # Always write label file (empty if no annotations)
        with open(dst_label_dir / label_file.name, "w") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")

    print(f"  Processed {total_files} files, {total_with_labels} had annotations")


def symlink_images(src_img_dir: Path, dst_img_dir: Path):
    """Symlink images from source to destination."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_img_dir.iterdir()):
        dst = dst_img_dir / img.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # Resolve symlinks so we point to the actual file
        os.symlink(img.resolve(), dst)
        count += 1
    print(f"  Symlinked {count} images")


def main():
    parser = argparse.ArgumentParser(
        description="Create 6-class OBB dataset without upper-view annotations"
    )
    parser.add_argument("--config", required=True,
                        help="Path to project YAML config (e.g. configs/H8_obb_botonly.yaml)")
    parser.add_argument("--source-obb-dataset",
                        help="Path to 6-class OBB dataset (input)")
    parser.add_argument("--output-dir",
                        help="Output directory for noflip dataset")
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
    src_6class = Path(args.source_obb_dataset or prep.get("source-obb-dataset", "data/obb/dataset_6class"))
    dst_dataset = Path(args.output_dir or prep.get("output-dir", "data/obb/dataset_noflip"))

    print(f"Config: {args.config}")
    print(f"  source-obb-dataset: {src_6class}")
    print(f"  output-dir:         {dst_dataset}")

    if dst_dataset.exists():
        print(f"Removing existing dataset at {dst_dataset}")
        shutil.rmtree(dst_dataset)

    # Flat directory structure (train/val split happens later)
    print("\nFiltering labels:")
    filter_labels(src_6class / "labels", dst_dataset / "labels")
    print("Symlinking images:")
    symlink_images(src_6class / "images", dst_dataset / "images")

    print(f"\nDataset created at {dst_dataset}")
    print("Classes: 0=up_finger (empty), 1=up_toe (empty), 2=bot_finger, 3=bot_toe, 4=ruler, 5=id")

    # Optional visualization
    if args.visualize > 0:
        vis_dir = dst_dataset / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        cls_colors = {
            2: ((255, 0, 255), "bot_finger"),
            3: ((0, 255, 0), "bot_toe"),
            4: ((255, 0, 0), "ruler"),
            5: ((0, 165, 255), "id"),
        }

        # Collect label files that have annotations
        label_files = [f for f in sorted((dst_dataset / "labels").glob("*.txt")) if f.stat().st_size > 0]
        random.seed(42)
        samples = random.sample(label_files, min(args.visualize, len(label_files)))

        for lf in samples:
            stem = lf.stem
            img_path = dst_dataset / "images" / f"{stem}.jpg"
            if not img_path.exists():
                continue

            img_bgr = cv2.imread(str(img_path.resolve()))
            h, w = img_bgr.shape[:2]
            scale_factor = max(h, w) / 1000
            line_thick = max(2, int(4 * scale_factor))
            font_scale = max(0.8, 1.2 * scale_factor)
            font_thick = max(2, int(3 * scale_factor))

            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    corners = np.array([
                        [coords[0] * w, coords[1] * h],
                        [coords[2] * w, coords[3] * h],
                        [coords[4] * w, coords[5] * h],
                        [coords[6] * w, coords[7] * h],
                    ], dtype=np.int32)
                    color, label = cls_colors.get(cls_id, ((255, 255, 255), f"cls_{cls_id}"))
                    cv2.polylines(img_bgr, [corners], True, color, thickness=line_thick)
                    cx_vis = int(np.mean(corners[:, 0]))
                    cy_vis = int(np.min(corners[:, 1])) - int(20 * scale_factor)
                    cv2.putText(img_bgr, label, (cx_vis - int(60 * scale_factor), cy_vis),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick)

            vis_path = vis_dir / f"{stem}_botonly.jpg"
            cv2.imwrite(str(vis_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Visualization: {vis_path}")


if __name__ == "__main__":
    main()
