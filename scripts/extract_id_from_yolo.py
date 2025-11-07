import os
import cv2
import easyocr
import argparse

import numpy as np
import yaml
import re
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
def crop_from_yolo(image, x_center, y_center, box_width, box_height,enhance=True, target_size=(128,64)):
    """Convert YOLO coords to pixel coords and crop image with padding."""
    # Denoise and sharpen
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    crop = cv2.filter2D(denoised, -1, sharp_kernel)
    h, w = crop.shape[:2]
    x_center_px = x_center * w
    y_center_px = y_center * h
    box_w_px = box_width * w
    box_h_px = box_height * h
    # Get top-left and bottom-right coordinates
    x_min = int(x_center_px - box_w_px / 2)
    x_max = int(x_center_px + box_w_px / 2)
    y_min = int(y_center_px - box_h_px / 2)
    y_max = int(y_center_px + box_h_px / 2)

    padding_ratio_x = -0.01
    padding_ratio_y = -0.01

    pad_x = box_w_px * padding_ratio_x
    pad_y = box_h_px * padding_ratio_y
    x_min = max(0, int(x_min - pad_x))
    x_max = min(w, int(x_max + pad_x))
    y_min = max(0, int(y_min - pad_y))
    y_max = min(h, int(y_max + pad_y))

    # Shrink bottom by, say, 10% of box height
    shrink_ratio = 0.10  # 10%
    y_max = int(y_max - box_h_px * shrink_ratio)

    crop = image[y_min:y_max, x_min:x_max]

    # Resize while maintaining aspect ratio
    target_w, target_h = target_size
    cropped_h, cropped_w = crop.shape[:2]

    scale_w = target_w / cropped_w
    scale_h = target_h / cropped_h
    scale = min(scale_w, scale_h)

    new_w = int(cropped_w * scale)
    new_h = int(cropped_h * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    crop = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[1,1,1])


    if enhance and crop.size > 0:
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Increase contrast and brightness
        alpha = 2.0  # Contrast control (1.0–3.0)
        beta = 10  # Brightness control (0–100)
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # Apply adaptive threshold to make digits stand out
        thresh = cv2.adaptiveThreshold(enhanced, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 27, 10)

        crop = thresh

    return crop

def detect_digits(image,conf_threshold=0.5):
    # Run EasyOCR at 0° first
    # Optional morphological enhancement
    cnts, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])  # left → right

    digit_imgs = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 80:  # skip noise
            digit_imgs.append(image[y:y + h, x:x + w])

    # results_0 = reader.readtext(image,detail=1, allowlist="0123456789")

    results_0 = []
    for d in digit_imgs:
        result = reader.readtext(d, detail=1, allowlist='0123456789')
        results_0.append(result[0]) if result else ""

    best_text = ""
    best_conf = 0
    if results_0:
        for (bbox, text, conf) in results_0:
            text = re.sub(r"\D", "", text)
            if not text:
                continue
            candidates = [text[i:i + 4] for i in range(len(text) - 3)] if len(text) > 4 else [text]

            valid_candidates = [c for c in candidates if 000 <= int(c) <= 2000]
            for c in valid_candidates[::-1]:
                print(c, conf)
                if 3 <= len(c) <= 4 and conf > best_conf:
                    best_conf = conf
                    best_text = c

    # Only try 180° rotation if confidence is below threshold
    if best_conf < conf_threshold:
        rotated_img = cv2.rotate(image, cv2.ROTATE_180)
        results_180 = reader.readtext(rotated_img, detail=1, allowlist="0123456789")
        if results_180:
            for (bbox, text, conf) in results_180:
                text = re.sub(r"\D", "", text)
                if not text:
                    continue
                candidates = [text[i:i + 4] for i in range(len(text) - 3)] if len(text) > 4 else [text]
                valid_candidates = [c for c in candidates if 0 <= int(c) <= 2000]
                for c in valid_candidates:
                    if 3 <= len(c) <= 4 and conf > best_conf:
                        best_conf = conf
                        best_text = c
                        print(f"↩️ Rotation 180° → {best_text} (conf {best_conf:.2f})")
    else:
        print(f"↩️ Rotation 0° → {best_text} (conf {best_conf:.2f})")

    return best_text


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