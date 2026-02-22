#!/usr/bin/env python3
"""
Create a 6-class OBB dataset without upper-view (flip) annotations.

Keeps the 6-class structure for consistency with H5 detect model, but:
  - Class 0 (up_finger): empty (no annotations)
  - Class 1 (up_toe):    empty (no annotations)
  - Class 2 (bot_finger): OBB from dataset_obb_6class
  - Class 3 (bot_toe):    OBB from dataset_obb_6class
  - Class 4 (ruler):      OBB from processed_obb (bottom class 2 -> remapped to 4)
  - Class 5 (id):         empty (no annotations, same as original)

This dataset is used to train YOLO-OBB for fair comparison against the
H5 baseline detect model (which also doesn't use flip strategy).

Usage:
    python scripts/preprocessing/create_noflip_obb_dataset.py
"""

import os
import shutil
from pathlib import Path


# Paths
project_root = os.environ.get("PROJECT_ROOT", "/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads")
SRC_6CLASS = Path(f"{project_root}/data/dataset_obb_6class")
SRC_PROCESSED_OBB = Path(f"{project_root}/data/processed_obb/labels")
DST_DATASET = Path(f"{project_root}/data/dataset_obb_noflip")

# From dataset_obb_6class: keep only bot_finger (2) and bot_toe (3)
KEEP_CLASSES = {2, 3}

# From processed_obb: class 2 (ruler) -> remap to class 4
RULER_SRC_CLASS = 2
RULER_DST_CLASS = 4


def get_ruler_obb_line(processed_label_path: Path) -> str | None:
    """Extract ruler OBB line from processed_obb label file.

    processed_obb has both 5-value (bbox) and 9-value (OBB) lines.
    We want the 9-value OBB line for class 2 (ruler), remapped to class 4.
    """
    if not processed_label_path.exists():
        return None

    with open(processed_label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls_id = int(parts[0])
            if cls_id == RULER_SRC_CLASS:
                return f"{RULER_DST_CLASS} {' '.join(parts[1:])}"
    return None


def filter_labels(src_label_dir: Path, dst_label_dir: Path):
    """Filter 6-class labels to keep only bot_finger, bot_toe, and add ruler from processed_obb."""
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0
    total_with_labels = 0
    ruler_added = 0

    for label_file in sorted(src_label_dir.glob("*.txt")):
        kept_lines = []

        # Keep bot_finger and bot_toe from 6-class OBB labels
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in KEEP_CLASSES:
                    kept_lines.append(line.strip())

        # Add ruler OBB from processed_obb
        processed_label = SRC_PROCESSED_OBB / label_file.name
        ruler_line = get_ruler_obb_line(processed_label)
        if ruler_line:
            kept_lines.append(ruler_line)
            ruler_added += 1

        total_files += 1
        if kept_lines:
            total_with_labels += 1

        # Always write label file (empty if no annotations)
        with open(dst_label_dir / label_file.name, "w") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")

    print(f"  Processed {total_files} files, {total_with_labels} had annotations")
    print(f"  Ruler OBB added: {ruler_added}/{total_files}")


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
    if DST_DATASET.exists():
        print(f"Removing existing dataset at {DST_DATASET}")
        shutil.rmtree(DST_DATASET)

    for split in ["train", "val"]:
        print(f"\nProcessing {split}:")
        filter_labels(SRC_6CLASS / "labels" / split, DST_DATASET / "labels" / split)
        symlink_images(SRC_6CLASS / "images" / split, DST_DATASET / "images" / split)

    print(f"\nDataset created at {DST_DATASET}")
    print("Classes: 0=up_finger (empty), 1=up_toe (empty), 2=bot_finger, 3=bot_toe, 4=ruler, 5=id (empty)")


if __name__ == "__main__":
    main()
