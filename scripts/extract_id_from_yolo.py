"""
ID OCR Pipeline with YOLO Landmarks + EasyOCR
---------------------------------------------
Reads YOLO label files, extracts the ID region, and runs OCR
(using EasyOCR) with orientation correction.

Input:
    images/ and labels/ directories
Output:
    text_results/ containing detected 4-digit IDs
"""

import os
import cv2
import easyocr
import argparse
import yaml
from pathlib import Path
reader = easyocr.Reader(["en"], gpu=False)

# -----------------------------
# Load YOLO Landmarks
# -----------------------------
def get_last_landmark(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
        last_line = lines[-1].strip()
        parts = last_line.split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])
    return class_id, x_center, y_center, box_width, box_height

# -----------------------------
# Crop Region from Image
# -----------------------------
def crop_from_yolo(image, x_center, y_center, box_width, box_height):
    """Convert YOLO coords to pixel coords and crop image with padding."""
    h, w = image.shape[:2]
    x_center_px = x_center * w
    y_center_px = y_center * h
    box_w_px = box_width * w
    box_h_px = box_height * h
    # Get top-left and bottom-right coordinates
    x_min = int(x_center_px - box_w_px / 2)
    x_max = int(x_center_px + box_w_px / 2)
    y_min = int(y_center_px - box_h_px / 2)
    y_max = int(y_center_px + box_h_px / 2)

    padding_ratio_x = 0.0
    padding_ratio_y = 0.0

    pad_x = box_w_px * padding_ratio_x
    pad_y = box_h_px * padding_ratio_y
    x_min = max(0, int(x_min - pad_x))
    x_max = min(w, int(x_max + pad_x))
    y_min = max(0, int(y_min - pad_y))
    y_max = min(h, int(y_max + pad_y))

    crop = image[y_min:y_max, x_min:x_max]

    return crop

# -----------------------------
# Run OCR (with Rotation Check)
# -----------------------------
def detect_digits(image):
    """Run EasyOCR on both normal and rotated image, choose 4-digit result."""
    result = reader.readtext(image, detail=0, allowlist='0123456789')
    result = [r for r in result if r.isdigit()]  # filter digits only

    # If we already have a 4-digit result, return it
    for r in result:
        if len(r) == 4:
            return r

    # Try 180-degree rotation
    crop_rot = cv2.rotate(image, cv2.ROTATE_180)
    result = reader.readtext(crop_rot, detail=0, allowlist='0123456789')
    result = [r for r in result if r.isdigit()]
    for r in result:
        if len(r) == 4:
            return r

    # If still no 4-digit, return the first numeric result or empty string
    return result[0] if result else ""

# -----------------------------
# Batch Processing
# -----------------------------
def batch_process_directory(images_dir, labels_dir, output_dir):
    """Process all image-label pairs in a directory."""
    os.makedirs(output_dir, exist_ok=True)


    for img_path in Path(images_dir).glob("*.jpg"):
        label_path = Path(labels_dir) / f"{img_path.stem.replace('marked_','')}.txt"

        if not label_path.exists():
            print(f"No label for {img_path.name}")
            continue

        class_id, x_center, y_center, box_width, box_height = get_last_landmark(label_path)
        if not box_width or not box_height:
            print(f"No valid landmark in {label_path.name}")
            continue

        image = cv2.imread(img_path)
        id_crop = crop_from_yolo(image, x_center, y_center, box_width, box_height)
        if id_crop.size == 0:
            raise ValueError(f"Cropped image is empty! Check coordinates or padding for {img_path.name}")

        cv2.imwrite(os.path.join(output_dir, f"{img_path.stem}_id.jpg"), id_crop)

        detected_id = detect_digits(id_crop)

        # Save cropped ID and OCR result

        with open(os.path.join(output_dir, f"{img_path.stem}_id.txt"), "w") as f:
            f.write(detected_id)

        print(f"{img_path.name} → Detected ID: {detected_id}")

# -----------------------------
# Performance
# -----------------------------
def calculate_accuracy(output_dir):
    """
    Compare detected IDs with expected IDs and compute accuracy.
    Assumes file naming like: marked_1008.jpg and marked_1008_id.txt
    """
    correct = 0
    total = 0

    for txt_file in Path(output_dir).glob("*_id.txt"):
        detected_id = txt_file.read_text().strip()  # Detected by OCR
        # Extract expected ID from filename
        expected_id = txt_file.stem.replace("_id", "").replace("marked_", "")

        # Only count if expected ID is numeric
        if expected_id.isdigit():
            total += 1
            if detected_id == expected_id:
                correct += 1
            else:
                print(f"❌ Mismatch: {txt_file.stem} → Detected: {detected_id}, Expected: {expected_id}")

    accuracy = correct / total * 100 if total else 0
    print(f"\nTotal labeled images: {total}")
    print(f"Correctly detected: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

# -----------------------------
# Run
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract ID from YOLO images.')

    parser.add_argument('--config', default='configs/H1.yaml', help='Path to YAML config file (default: configs/H1.yaml). CLI flags override config.')

    # Batch mode
    parser.add_argument('--image-dir', help='Directory containing YOLO images')
    parser.add_argument('--labels-dir', help='Directory containing YOLO labels')

    # Shared
    parser.add_argument('--output-dir', help='Output directory')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Load YAML config if provided
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}


    def get_opt(name: str, default):
        # CLI has highest precedence
        value = getattr(args, name.replace('-', '_'), None)
        if value not in (None, ""):
            return value
        # then config YAML
        # support nested under 'extracting' section
        if name in cfg and cfg[name] not in (None, ""):
            return cfg[name]
        if isinstance(cfg.get('extracting'), dict) and cfg['extracting'].get(name) not in (None, ""):
            return cfg['extracting'][name]
        return default

    image_dir = get_opt('image-dir', None)
    label_dir = get_opt('label-dir', None)
    output_dir = get_opt('output-dir', 'data/processed')

    print(f"Using output_dir: {output_dir}")

    if not image_dir or not label_dir:
        raise ValueError("image-dir and labels_dir are required. Provide via --image-dir/--label_dir or config file.")

    batch_process_directory(
        images_dir=image_dir,
        labels_dir=label_dir,
        output_dir=output_dir
    )

    # Then calculate accuracy
    calculate_accuracy(output_dir)