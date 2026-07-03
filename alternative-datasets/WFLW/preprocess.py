"""
WFLW dataset preprocessor.

Converts raw WFLW JPEG images and annotation text files into .pt crop files
compatible with the existing LizardDataset format.

Each output .pt file contains:
  - "image": (3, 512, 512) uint8 tensor — letterbox-resized RGB crop
  - "tps":   (98, 2) float32 tensor — landmark coordinates normalized to [0, 1]
  - "attrs": (6,) uint8 tensor — binary attribute flags:
             [pose, expression, illumination, makeup, occlusion, blur]
  - "orig_size": (2,) int tensor — [H, W] of the crop before letterbox

Usage:
    # Process training split
    python preprocess.py \\
        --annotation-file WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt \\
        --image-root WFLW_images/ \\
        --output-dir pt_crops/train/

    # Process test split
    python preprocess.py \\
        --annotation-file WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt \\
        --image-root WFLW_images/ \\
        --output-dir pt_crops/test/
"""
import argparse
import logging
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def parse_annotation_line(line: str) -> dict:
    """Parse one WFLW annotation line into its component fields.

    Format (207 whitespace-separated tokens):
        x0 y0 x1 y1 ... x97 y97         (196 floats — 98 landmark x,y pairs)
        x_min y_min x_max y_max          (4 floats  — bounding box)
        pose expr illum makeup occ blur  (6 ints    — binary attributes)
        image/relative/path.jpg          (1 string  — relative image path)

    Args:
        line: A single non-empty annotation line.

    Returns:
        dict with keys:
            "landmarks": ndarray (98, 2) float64
            "bbox":      ndarray (4,)  float64  [x_min, y_min, x_max, y_max]
            "attrs":     ndarray (6,)  uint8
            "image_path": str

    Raises:
        ValueError: If the line does not have exactly 207 tokens.
    """
    tokens = line.strip().split()
    if len(tokens) != 207:
        raise ValueError(
            f"Expected 207 tokens per annotation line, got {len(tokens)}"
        )

    landmarks = np.array(tokens[:196], dtype=np.float64).reshape(98, 2)
    bbox = np.array(tokens[196:200], dtype=np.float64)
    attrs = np.array(tokens[200:206], dtype=np.uint8)
    image_path = tokens[206]

    return {
        "landmarks": landmarks,
        "bbox": bbox,
        "attrs": attrs,
        "image_path": image_path,
    }


def letterbox_crop(
    img: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray,
    padding: float = 0.10,
    target_size: int = 512,
) -> tuple:
    """Crop the face region and letterbox-resize to target_size × target_size.

    Args:
        img:        HWC uint8 BGR image (as read by cv2).
        bbox:       (4,) float array [x_min, y_min, x_max, y_max] in pixel space.
        landmarks:  (98, 2) float array of landmark coordinates in pixel space.
        padding:    Fractional padding to add around the bounding box on each side.
        target_size: Output image size (square).

    Returns:
        Tuple of:
            img_chw:         (3, target_size, target_size) uint8 numpy array (CHW RGB)
            landmarks_norm:  (98, 2) float32 array with coordinates in [0, 1]
    """
    H, W = img.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    # Expand bbox by padding fraction
    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = bw * padding
    pad_y = bh * padding

    x_min_pad = max(0, x_min - pad_x)
    y_min_pad = max(0, y_min - pad_y)
    x_max_pad = min(W, x_max + pad_x)
    y_max_pad = min(H, y_max + pad_y)

    # Crop
    x1, y1 = int(x_min_pad), int(y_min_pad)
    x2, y2 = int(x_max_pad), int(y_max_pad)
    crop = img[y1:y2, x1:x2]

    # Convert BGR → RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_h, crop_w = crop_rgb.shape[:2]

    # Remap landmark coords to crop-local space
    lm_crop = landmarks.copy()
    lm_crop[:, 0] -= x1
    lm_crop[:, 1] -= y1

    # Letterbox resize via albumentations (preserves aspect ratio)
    transform = A.Compose(
        [
            A.LongestMaxSize(max_size=target_size),
            A.PadIfNeeded(
                min_height=target_size,
                min_width=target_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    keypoints = lm_crop.tolist()
    augmented = transform(image=crop_rgb, keypoints=keypoints)
    img_resized = augmented["image"]
    kps_resized = np.array(augmented["keypoints"], dtype=np.float32)

    # Normalize to [0, 1]
    landmarks_norm = kps_resized / target_size
    landmarks_norm = np.clip(landmarks_norm, 0.0, 1.0)

    # HWC → CHW
    img_chw = img_resized.transpose(2, 0, 1)  # (3, H, W) uint8

    return img_chw, landmarks_norm.astype(np.float32)


def preprocess_wflw(
    annotation_file: str,
    image_root: str,
    output_dir: str,
) -> None:
    """Convert a WFLW annotation file + images into .pt crop files.

    Skips individual samples that have malformed annotation lines or missing
    image files, logging a warning for each. Never raises on a per-sample basis.

    Args:
        annotation_file: Path to the WFLW annotation .txt file.
        image_root:      Root directory containing WFLW images (subdirs by category).
        output_dir:      Output directory for .pt files. Created if it does not exist.
    """
    annotation_file = Path(annotation_file)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    logging.info(f"Processing {len(lines)} samples from {annotation_file.name} → {output_dir}")
    saved = 0
    skipped = 0

    for line_idx, line in enumerate(lines):
        # Parse annotation
        try:
            ann = parse_annotation_line(line)
        except ValueError as e:
            logging.warning(f"Line {line_idx}: malformed annotation — {e}")
            skipped += 1
            continue

        # Resolve image path
        img_path = image_root / ann["image_path"]
        if not img_path.exists():
            logging.warning(f"Line {line_idx}: image not found — {img_path}")
            skipped += 1
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning(f"Line {line_idx}: cv2 failed to read — {img_path}")
            skipped += 1
            continue

        # Crop and resize
        try:
            img_chw, landmarks_norm = letterbox_crop(
                img, ann["bbox"], ann["landmarks"]
            )
        except Exception as e:
            logging.warning(f"Line {line_idx}: letterbox_crop failed — {e}")
            skipped += 1
            continue

        # Build stem — include line index to handle multiple faces per image
        # (WFLW images can contain multiple annotated faces from WIDER Face)
        stem = f"{Path(ann['image_path']).stem}_{line_idx:05d}"

        # Save .pt file
        pt_data = {
            "image": torch.from_numpy(img_chw),                          # (3,512,512) uint8
            "tps":   torch.from_numpy(landmarks_norm),                    # (98,2) float32
            "attrs": torch.from_numpy(ann["attrs"]),                      # (6,) uint8
            "orig_size": torch.tensor(
                [img_chw.shape[1], img_chw.shape[2]], dtype=torch.int32  # [H, W]
            ),
        }

        out_path = output_dir / f"{stem}.pt"
        torch.save(pt_data, str(out_path))
        saved += 1

        if saved % 500 == 0:
            logging.info(f"  {saved}/{len(lines)} saved...")

    logging.info(
        f"Done. Saved: {saved}, Skipped: {skipped}, Total: {len(lines)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert WFLW annotations and images to .pt crop files"
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        required=True,
        help=(
            "Path to WFLW annotation txt file, e.g. "
            "WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
        ),
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory containing WFLW images (e.g. WFLW_images/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for .pt files (e.g. pt_crops/train/)",
    )
    args = parser.parse_args()

    if not Path(args.annotation_file).exists():
        print(f"ERROR: annotation file not found: {args.annotation_file}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.image_root).exists():
        print(f"ERROR: image root not found: {args.image_root}", file=sys.stderr)
        sys.exit(1)

    preprocess_wflw(args.annotation_file, args.image_root, args.output_dir)
