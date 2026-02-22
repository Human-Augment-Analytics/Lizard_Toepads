#!/usr/bin/env python3
"""Create a 6-class YOLO-OBB dataset by merging bottom-view OBB labels
with upper-view standard bbox labels (converted to axis-aligned OBB format).

6-class scheme:
  0: up_finger   (from upper bbox class 0)
  1: up_toe      (from upper bbox class 1)
  2: bot_finger  (from bottom OBB class 0)
  3: bot_toe     (from bottom OBB class 1)
  4: ruler       (from bottom OBB class 2)
  5: id          (no annotations)

Bottom OBB labels: 9-value lines (class x1 y1 x2 y2 x3 y3 x4 y4)
Upper bbox labels: 5-value lines (class cx cy w h) -> converted to 9-value OBB
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import math


# Class remapping
BOTTOM_REMAP = {0: 2, 1: 3, 2: 4}  # finger->bot_finger, toe->bot_toe, ruler->ruler
UPPER_REMAP = {0: 0, 1: 1}          # up_finger->up_finger, up_toe->up_toe


def rotate_bbox_to_obb(cx, cy, w, h, angle_deg):
    """Convert bbox (cx, cy, w, h) rotated by angle_deg to 4 corners.
    
    Args:
        cx, cy: Center coordinates
        w, h: Width and height (unrotated dimensions)
        angle_deg: Rotation angle in degrees (clockwise positive? standard cv2 convention)
    
    Returns:
        8 floats: x1 y1 ... x4 y4
    """
    # Create the unrotated rectangle corners relative to center (0,0)
    # BL, TL, TR, BR order to match cv2 structure usually? 
    # Actually, let's just use simple rotation matrix.
    
    # 4 corners relative to center:
    # (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)  <-- TL, TR, BR, BL order?
    # TL: -w/2, -h/2
    # TR: w/2, -h/2
    # BR: w/2, h/2
    # BL: -w/2, h/2
    
    corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ])
    
    # Rotation matrix
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    # Rotate
    rotated_corners = corners @ R.T
    
    # Shift to center
    rotated_corners[:, 0] += cx
    rotated_corners[:, 1] += cy
    
    # Flatten
    return tuple(rotated_corners.flatten())


def bbox_to_obb(cx, cy, w, h):
    """Convert standard bbox (cx, cy, w, h) to 4 axis-aligned corners.

    Returns 8 floats: x1 y1 x2 y2 x3 y3 x4 y4 (TL, TR, BR, BL).
    """
    return rotate_bbox_to_obb(cx, cy, w, h, 0.0)


def extract_bot_info(label_path):
    """Extract class, angle, width, and height from bottom-view OBB labels.
    
    Returns:
        dict: {target_cls_id: {'angle': float, 'w': float, 'h': float}}
        Target class ID is the UPPER class that corresponds to the found BOTTOM class.
        e.g. found bot_finger (0) -> store under up_finger (0)
    """
    info = {}
    if not label_path.exists():
        return info

    # Mapping from bot class to upper class for storage
    # bot_finger (0) -> up_finger (0)
    # bot_toe (1) -> up_toe (1)
    # ruler (2) -> X
    BOT_TO_UP_MAP = {0: 0, 1: 1} # Reverse of BOTTOM_REMAP but using raw class IDs

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            
            cls = int(parts[0])
            if cls not in BOT_TO_UP_MAP:
                continue
            
            upper_cls = BOT_TO_UP_MAP[cls]
            
            # Extract corners
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(4, 2).astype(np.float32)
            
            # Get MinAreaRect
            # (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
            rect = cv2.minAreaRect(pts)
            (cx, cy), (w, h), angle = rect
            
            # Use the larger dimension as length? 
            # Roboflow bbox w/h are usually width/height relative to image axis.
            # But here we want the physical dimensions of the limb.
            
            info[upper_cls] = {'angle': angle, 'w': w, 'h': h}
            
    return info


def parse_bottom_obb_labels(label_path):
    """Read bottom-view label file, keep only 9-value OBB lines, remap classes."""
    lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue  # skip 5-value standard bbox lines
            cls = int(parts[0])
            if cls not in BOTTOM_REMAP:
                continue
            new_cls = BOTTOM_REMAP[cls]
            coords = parts[1:]
            lines.append(f"{new_cls} {' '.join(coords)}")
    return lines


def parse_upper_bbox_labels(label_path, bot_info=None):
    """Read upper-view label file, convert standard bbox to OBB, remap classes.
    
    Args:
        label_path: Path to upper label file
        bot_info: dict containing angle/dim info from bottom view, keyed by upper class ID
    """
    lines = []
    if bot_info is None:
        bot_info = {}

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls not in UPPER_REMAP:
                continue
            new_cls = UPPER_REMAP[cls]
            cx, cy, w, h = map(float, parts[1:])
            
            # Check if we have transfer info
            angle = 0.0
            
            # For Dim transfer, if available, use bot's w/h
            if new_cls in bot_info:
                angle = bot_info[new_cls]['angle']
                
                # Use Bot dimensions if they exist
                # But notice: MinAreaRect returns (w,h) which might be rotated 90 deg relative to what we expect.
                # However, for an OBB, (w,h) and angle define the box.
                # If we just use bot's w, h, angle centered at upper's cx, cy, 
                # we get a box of identical shape/orientation to bot, but at upper position.
                # This assumes upper limb size ~= bottom limb size (plausible)
                
                # Override w, h with bot's
                w = bot_info[new_cls]['w']
                h = bot_info[new_cls]['h']
            
            corners = rotate_bbox_to_obb(cx, cy, w, h, angle)
            coords_str = ' '.join(f"{v:.6f}" for v in corners)
            lines.append(f"{new_cls} {coords_str}")
    return lines


def parse_split_file(split_path):
    """Parse a split file to extract image stem names."""
    stems = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Lines are full paths like /path/to/images/1234.jpg
            stem = Path(line).stem
            stems.append(stem)
    return stems


def create_dataset(args):
    bottom_labels_dir = Path(args.bottom_labels_dir)
    upper_labels_dir = Path(args.upper_labels_dir)
    images_dir = Path(args.images_dir)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    # Parse train/val splits
    train_stems = parse_split_file(splits_dir / "train.txt")
    val_stems = parse_split_file(splits_dir / "val.txt")

    print(f"Train images: {len(train_stems)}, Val images: {len(val_stems)}")

    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        skipped = 0
        written = 0

        for stem in stems:
            # Find image file (try common extensions)
            img_src = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_src = candidate
                    break

            if img_src is None:
                skipped += 1
                continue

            # Symlink image
            img_dst = img_out / img_src.name
            if not img_dst.exists():
                img_dst.symlink_to(img_src.resolve())

            # Merge labels
            merged_lines = []

            # Bottom-view OBB labels
            bottom_file = bottom_labels_dir / f"{stem}.txt"
            
            # Extract info for transfer FIRST
            bot_info = {}
            if bottom_file.exists():
                bot_info = extract_bot_info(bottom_file)
                # Also append bottom labels to merged
                merged_lines.extend(parse_bottom_obb_labels(bottom_file))

            # Upper-view labels (converted bbox -> OBB using bot info)
            upper_file = upper_labels_dir / f"{stem}.txt"
            if upper_file.exists():
                merged_lines.extend(parse_upper_bbox_labels(upper_file, bot_info))

            # Write merged label file
            lbl_dst = lbl_out / f"{stem}.txt"
            with open(lbl_dst, "w") as f:
                f.write("\n".join(merged_lines) + "\n" if merged_lines else "")

            written += 1

        print(f"  {split_name}: {written} written, {skipped} skipped (image not found)")

    # Summary: check a sample label
    sample_stem = train_stems[0] if train_stems else val_stems[0]
    sample_file = output_dir / "labels" / "train" / f"{sample_stem}.txt"
    if sample_file.exists():
        print(f"\nSample label ({sample_file.name}):")
        with open(sample_file) as f:
            for line in f:
                print(f"  {line.rstrip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 6-class merged OBB dataset from bottom OBB + upper bbox labels"
    )
    parser.add_argument(
        "--bottom-labels-dir",
        default="data/processed_obb/labels",
        help="Directory with bottom-view OBB label files (default: data/processed_obb/labels)",
    )
    parser.add_argument(
        "--upper-labels-dir",
        default="/storage/ice-shared/cs8903onl/miami_fall_24_upper_dataset_roboflow/train/labels",
        help="Directory with upper-view standard bbox label files",
    )
    parser.add_argument(
        "--images-dir",
        default="data/processed_obb/images",
        help="Directory with source images (default: data/processed_obb/images)",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/dataset_obb/splits",
        help="Directory with train.txt and val.txt split files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dataset_obb_6class",
        help="Output directory for merged dataset (default: data/dataset_obb_6class)",
    )
    args = parser.parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
