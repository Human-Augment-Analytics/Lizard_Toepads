#!/usr/bin/env python3
"""
Create the H5_obb_6class dataset — bot_finger, bot_toe, ruler, and id all annotated.

Class scheme:
  - Class 0 (up_finger): empty (handled via flip at inference)
  - Class 1 (up_toe):    empty (handled via flip at inference)
  - Class 2 (bot_finger): OBB from dataset_obb_6class
  - Class 3 (bot_toe):    OBB from dataset_obb_6class
  - Class 4 (ruler):      OBB from processed_obb (class 2 -> remapped to 4)
  - Class 5 (id):         OBB from id_obb_from_tps

Usage:
    python scripts/preprocessing/create_6class_active_obb_dataset.py
"""

import os
import shutil
from pathlib import Path

project_root = os.environ.get("PROJECT_ROOT", "/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads")
SRC_6CLASS = Path(f"{project_root}/data/dataset_obb_6class")
SRC_PROCESSED_OBB = Path(f"{project_root}/data/processed_obb/labels")
SRC_ID_OBB = Path(f"{project_root}/data/id_obb_from_tps")
DST_DATASET = Path(f"{project_root}/data/dataset_obb_6class_active")

KEEP_CLASSES = {2, 3}
RULER_SRC_CLASS = 2
RULER_DST_CLASS = 4


def get_ruler_obb_line(processed_label_path: Path):
    """Extract ruler OBB line from processed_obb label (9-value lines only)."""
    if not processed_label_path.exists():
        return None
    with open(processed_label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            if int(parts[0]) == RULER_SRC_CLASS:
                return f"{RULER_DST_CLASS} {' '.join(parts[1:])}"
    return None


def get_id_obb_line(id_label_path: Path):
    """Read id OBB line from id_obb_from_tps label."""
    if not id_label_path.exists():
        return None
    with open(id_label_path) as f:
        line = f.readline().strip()
    return line if line else None


def filter_labels(src_label_dir: Path, dst_label_dir: Path):
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    total = ruler_added = id_added = 0

    for label_file in sorted(src_label_dir.glob("*.txt")):
        kept_lines = []

        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) in KEEP_CLASSES:
                    kept_lines.append(line.strip())

        ruler_line = get_ruler_obb_line(SRC_PROCESSED_OBB / label_file.name)
        if ruler_line:
            kept_lines.append(ruler_line)
            ruler_added += 1

        id_line = get_id_obb_line(SRC_ID_OBB / label_file.name)
        if id_line:
            kept_lines.append(id_line)
            id_added += 1

        with open(dst_label_dir / label_file.name, "w") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")

        total += 1

    print(f"  Processed {total} files | ruler: {ruler_added} | id: {id_added}")


def symlink_images(src_img_dir: Path, dst_img_dir: Path):
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
        filter_labels(SRC_6CLASS / "labels" / split, DST_DATASET / "labels" / split)
        symlink_images(SRC_6CLASS / "images" / split, DST_DATASET / "images" / split)

    print(f"\nDataset created at {DST_DATASET}")
    print("Classes: 0=up_finger (empty), 1=up_toe (empty), 2=bot_finger, 3=bot_toe, 4=ruler, 5=id")


if __name__ == "__main__":
    main()
