"""
WFLW dataset preprocessor.

Converts raw WFLW JPEG images and annotation text files into .pt crop files
compatible with the existing LizardDataset format.

Uses the paper-faithful affine crop (Wang et al. CVPR 2019) rather than
letterboxing.  The affine crop warps a square region centred on the detection
bbox to fill the full target_size × target_size canvas.  Landmarks are mapped
through the same affine matrix, guaranteeing pixel-perfect alignment.

Each output .pt file contains:
  - "image": (3, target_size, target_size) uint8 tensor — RGB, affine-cropped
  - "tps":   (98, 2) float32 tensor — landmark coordinates normalized to [0, 1]
  - "attrs": (6,) uint8 tensor — binary attribute flags:
             [pose, expression, illumination, makeup, occlusion, blur]
  - "orig_size": (2,) int tensor — [H, W] of the source bbox region

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
    return {"landmarks": landmarks, "bbox": bbox, "attrs": attrs,
            "image_path": image_path}


# ── Paper-faithful affine crop utilities ─────────────────────────────────────
# Ported from HRNet-Facial-Landmark-Detection/lib/utils/transforms.py
# (Wang et al., CVPR 2019)

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad: float):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn,
            src_point[0] * sn + src_point[1] * cs]


def _get_affine_transform(
    center: np.ndarray,
    scale: float,
    rot: float,
    output_size,
    inv: int = 0,
) -> np.ndarray:
    """Compute 2×3 affine matrix that maps the face crop to output_size.

    Args:
        center:      (2,) pixel coordinates of the face centre.
        scale:       Scalar — face size = scale * 200 pixels.
        rot:         Rotation angle in degrees (0 for preprocessing).
        output_size: (W, H) of the output canvas.
        inv:         0 = forward (src→dst), 1 = inverse (dst→src).

    Returns:
        (2, 3) float32 affine matrix for cv2.warpAffine.
    """
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size[0], output_size[1]

    rot_rad = np.pi * rot / 180.0
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + np.array(src_dir, dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = _get_3rd_point(src[0], src[1])
    dst[2, :] = _get_3rd_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _affine_transform_point(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a 3×3 (homogeneous) affine transform to a 2-D point."""
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0], dtype=np.float64)
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(np.float32) + 1


def _get_transform_matrix(
    center: np.ndarray,
    scale: float,
    output_size,
    rot: float = 0.0,
) -> np.ndarray:
    """Build the 3×3 transform matrix used to map landmark coordinates.

    Equivalent to HRNet's get_transform(), used with invert=1 in transform_pixel
    to map original image coords → output canvas coords.
    """
    h = 200.0 * scale
    t = np.zeros((3, 3), dtype=np.float64)
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1.0
    return t


def affine_crop(
    img: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray,
    target_size: int = 512,
    scale_padding: float = 1.25,
) -> tuple:
    """Crop the face region using a paper-faithful affine transform.

    Replicates the HRNet preprocessing pipeline:
      1. Compute face centre and scale from the detection bbox.
      2. Apply a 2×3 affine matrix (cv2.warpAffine) to produce a tight
         square crop that fills the full target_size × target_size canvas.
      3. Map landmark coordinates through the same transformation matrix
         to guarantee pixel-perfect alignment.

    Args:
        img:           HWC uint8 BGR image (as read by cv2).
        bbox:          (4,) float array [x_min, y_min, x_max, y_max].
        landmarks:     (98, 2) float array in original image pixel space.
        target_size:   Output canvas size (square).
        scale_padding: Multiply the raw bbox scale by this factor (paper: 1.25).

    Returns:
        img_chw:        (3, target_size, target_size) uint8 ndarray — CHW RGB.
        landmarks_norm: (98, 2) float32 ndarray in [0, 1] on the output canvas.
    """
    x_min, y_min, x_max, y_max = bbox

    # Face centre and scale (paper convention: scale * 200 = face width in px)
    center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0],
                      dtype=np.float32)
    face_size = max(x_max - x_min, y_max - y_min)
    scale = (face_size / 200.0) * scale_padding   # scalar

    # Forward affine: original image → output canvas
    output_size = (target_size, target_size)
    trans = _get_affine_transform(center, scale, rot=0, output_size=output_size)

    # Warp image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_warped = cv2.warpAffine(
        img_rgb, trans, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )  # HWC uint8 RGB

    # Map landmarks through the same affine matrix used for the image warp.
    # The 2×3 affine matrix T maps [x, y, 1]^T → [x', y']^T in output space.
    # This is guaranteed to be consistent with cv2.warpAffine.
    lm_h = np.hstack([landmarks.astype(np.float32), np.ones((len(landmarks), 1), dtype=np.float32)])
    lm_out = (trans @ lm_h.T).T   # (98, 2) in output pixel space

    # Normalise to [0, 1]
    landmarks_norm = lm_out / float(target_size)
    landmarks_norm = np.clip(landmarks_norm, 0.0, 1.0)

    # HWC → CHW
    img_chw = img_warped.transpose(2, 0, 1)

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

        # Compute pre-crop face dimensions for orig_size
        H_img, W_img = img.shape[:2]
        x_min, y_min, x_max, y_max = ann["bbox"]
        face_size = max(x_max - x_min, y_max - y_min)
        crop_h = int(face_size * 1.25)
        crop_w = int(face_size * 1.25)

        # Affine crop
        try:
            img_chw, landmarks_norm = affine_crop(
                img, ann["bbox"], ann["landmarks"]
            )
        except Exception as e:
            logging.warning(f"Line {line_idx}: affine_crop failed — {e}")
            skipped += 1
            continue

        # Build stem — include line index to handle multiple faces per image
        # (WFLW images can contain multiple annotated faces from WIDER Face)
        stem = f"{Path(ann['image_path']).stem}_{line_idx:05d}"

        # Save .pt file
        # orig_size is the approximate face crop dimensions before affine warp,
        # used by compute_rescaled_pixel_error to report error in original pixel space.
        pt_data = {
            "image": torch.from_numpy(img_chw),                               # (3,512,512) uint8
            "tps":   torch.from_numpy(landmarks_norm),                        # (98,2) float32
            "attrs": torch.from_numpy(ann["attrs"]),                          # (6,) uint8
            "orig_size": torch.tensor([crop_h, crop_w], dtype=torch.int32),  # [H, W] pre-affine
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
