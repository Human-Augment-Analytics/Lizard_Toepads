#!/usr/bin/env python3
"""
Create a 2-class OBB dataset (bot_finger, bot_toe) from the 6-class dataset.

Filters labels to keep only class 2 (bot_finger) and 3 (bot_toe) from the
6-class dataset, remapping them to class 0 and 1 respectively.
Images are symlinked from the 6-class dataset.

Usage:
    python scripts/preprocessing/create_2class_obb_dataset.py
"""

import os
import shutil
from pathlib import Path

# Paths
SRC_DATASET = Path("/home/hice1/yloh30/scratch/Lizard_Toepads/data/dataset_obb_6class")
DST_DATASET = Path("/home/hice1/yloh30/scratch/Lizard_Toepads/data/dataset_obb_2class")

# 6-class mapping: 0=up_finger, 1=up_toe, 2=bot_finger, 3=bot_toe, 4=ruler, 5=id
# 2-class mapping: 0=bot_finger, 1=bot_toe
CLASS_REMAP = {2: 0, 3: 1}


def filter_labels(src_label_dir: Path, dst_label_dir: Path):
    """Filter label files to keep only bot_finger and bot_toe, remapping classes."""
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0
    total_kept = 0

    for label_file in sorted(src_label_dir.glob("*.txt")):
        kept_lines = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in CLASS_REMAP:
                    new_cls = CLASS_REMAP[cls_id]
                    kept_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        total_files += 1
        if kept_lines:
            total_kept += 1
            with open(dst_label_dir / label_file.name, "w") as f:
                f.write("\n".join(kept_lines) + "\n")
        else:
            # Write empty label file so YOLO doesn't complain about missing labels
            with open(dst_label_dir / label_file.name, "w") as f:
                pass

    print(f"  Processed {total_files} files, {total_kept} had bot_finger/bot_toe labels")


def symlink_images(src_img_dir: Path, dst_img_dir: Path):
    """Symlink images from source to destination."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_img_dir.iterdir()):
        dst = dst_img_dir / img.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(img.resolve(), dst)
        count += 1
    print(f"  Symlinked {count} images")


def main():
    if DST_DATASET.exists():
        print(f"Removing existing dataset at {DST_DATASET}")
        shutil.rmtree(DST_DATASET)

    for split in ["train", "val"]:
        print(f"\nProcessing {split}:")
        filter_labels(SRC_DATASET / "labels" / split, DST_DATASET / "labels" / split)
        symlink_images(SRC_DATASET / "images" / split, DST_DATASET / "images" / split)

    print(f"\nDataset created at {DST_DATASET}")
    print("Classes: 0=bot_finger, 1=bot_toe")


if __name__ == "__main__":
    main()
