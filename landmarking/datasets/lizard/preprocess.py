"""Lizard dataset preprocessor.

Converts raw lizard images + TPS annotation files + YOLO OBB detections
into .pt crop files compatible with LizardDataset.

All paths are parameterized via config or CLI arguments — no hardcoded
absolute paths.

Each output .pt file contains:
  - "image":     (3, 512, 512) uint8 tensor — RGB, letterbox-padded
  - "tps":       (9, 2) float32 tensor — keypoints in 512px space
  - "heatmap":   (9, 512, 512) float32 tensor — Gaussian heatmaps
  - "orig_size": (2,) float32 tensor — [H, W] of OBB crop before resize
  - "M":         (3, 3) float64 tensor — perspective transform matrix
  - "scale":     scalar float32 — resize scale factor
  - "pad":       (2,) float32 tensor — [pad_x, pad_y]
  - "class_name": str — "finger" or "toe"
  - "ruler_px":  scalar float32 — ruler pixel distance (for mm conversion)

Usage:
    python -m landmarking.datasets.lizard.preprocess \\
        --config path/to/preprocess_config.json

    Or programmatically:
        from landmarking.datasets.lizard.preprocess import run_preprocessing
        run_preprocessing(config_dict)
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .tps_utils import get_tps_coords, get_ruler_distance
from .obb_utils import (
    CLASSMAP,
    TRAIN_CLASSES,
    crop_obb_from_corners,
    resize_and_pad,
    transform_keypoints,
)
from ...common.heatmap_utils import generate_gaussian_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_config(config_path: str) -> dict:
    """Load and validate preprocessing config JSON.

    Required keys: yolo_obb_path, imgdir, tps_data_dir, data_dir
    """
    p = Path(config_path)
    if not p.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(p) as f:
        config = json.load(f)
    required = ["yolo_obb_path", "imgdir", "tps_data_dir", "data_dir"]
    for key in required:
        if key not in config:
            logging.error(f"Required key '{key}' missing from config")
            sys.exit(1)
    return config


def collect_image_ids(imgdir: str) -> list:
    """Collect valid image IDs from the image directory.

    Filters for .jpg files with numeric stems > 1000.
    """
    dir_path = Path(imgdir)
    ids = []
    for f in dir_path.iterdir():
        if f.suffix == ".jpg":
            try:
                imgid = int(f.stem)
                if imgid > 1000:
                    ids.append(str(imgid))
            except ValueError:
                continue
    return sorted(ids)


def process_image_annotated(imgid, model, config, output_dir):
    """Process LHS crops from original image (annotated with TPS ground truth).

    Args:
        imgid: Image ID string.
        model: YOLO model instance.
        config: Config dict with imgdir, tps_data_dir keys.
        output_dir: Directory to write .pt files.

    Returns:
        (written_count, skipped_count)
    """
    img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
    if img is None:
        logging.warning(f"[{imgid}] Could not read image, skipping")
        return 0, 0

    tps = get_tps_coords(imgid, img, config["tps_data_dir"])
    ruler_distances = get_ruler_distance(imgid, img, config["tps_data_dir"])

    results = model(img, verbose=False)
    if results[0].obb is None:
        return 0, 0

    obb_corners = results[0].obb.xyxyxyxy.cpu().numpy()
    cls_ids = results[0].obb.cls.cpu().numpy()

    # Identify valid crops: must have correct TPS count and in-bounds keypoints
    crop_classes = []
    for corners, cls_id in zip(obb_corners, cls_ids):
        cls_id = int(cls_id)
        if cls_id not in TRAIN_CLASSES:
            continue
        class_name = CLASSMAP[cls_id]
        global_kps = tps.get(class_name, [])
        if len(global_kps) == 0 or len(global_kps) != 9:
            continue
        crop_check, M_check = crop_obb_from_corners(img, corners)
        local_kps = transform_keypoints(np.array(global_kps), M_check)
        h_c, w_c = crop_check.shape[:2]
        in_bounds = (
            (local_kps[:, 0] >= 0)
            & (local_kps[:, 0] < w_c)
            & (local_kps[:, 1] >= 0)
            & (local_kps[:, 1] < h_c)
        )
        if not np.all(in_bounds):
            continue
        crop_classes.append(class_name)

    # Process valid detections
    written = 0
    skipped = 0
    crop_idx = 0

    for corners, cls_id in zip(obb_corners, cls_ids):
        cls_id = int(cls_id)
        if cls_id not in TRAIN_CLASSES:
            continue
        if crop_idx >= len(crop_classes):
            break
        if crop_idx > 1:
            break

        class_name = crop_classes[crop_idx]
        crop, M = crop_obb_from_corners(img, corners)
        global_kps = tps.get(class_name, [])
        local_kps = transform_keypoints(np.array(global_kps), M)

        orig_h, orig_w = crop.shape[:2]
        img_aug, kps_aug, scale, pad_x, pad_y = resize_and_pad(crop, local_kps)

        if kps_aug.shape[0] != 9:
            logging.warning(
                f"[{imgid}_{crop_idx}] Keypoint count {kps_aug.shape[0]} != 9, skipping"
            )
            skipped += 1
            crop_idx += 1
            continue

        # Generate heatmap
        heatmap = generate_gaussian_heatmap(kps_aug, 512, sigma=6)

        # Get ruler pixel distance for this class
        ruler_px = ruler_distances.get(class_name, None)

        out_path = os.path.join(output_dir, f"{imgid}_{crop_idx}.pt")
        torch.save(
            {
                "image": torch.from_numpy(img_aug).permute(2, 0, 1).to(torch.uint8),
                "tps": torch.from_numpy(kps_aug).to(torch.float32),
                "heatmap": torch.from_numpy(heatmap).to(torch.float32),
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
                "M": torch.from_numpy(M).to(torch.float64),
                "scale": torch.tensor(scale, dtype=torch.float32),
                "pad": torch.tensor([pad_x, pad_y], dtype=torch.float32),
                "class_name": class_name,
                "ruler_px": torch.tensor(
                    ruler_px if ruler_px is not None else 0.0,
                    dtype=torch.float32,
                ),
            },
            out_path,
        )
        written += 1
        crop_idx += 1

    return written, skipped


def run_preprocessing(config: dict, test_ratio: float = 0.05, seed: int = 42):
    """Run the full Lizard preprocessing pipeline.

    Args:
        config: Dict with keys: yolo_obb_path, imgdir, tps_data_dir, data_dir.
        test_ratio: Fraction of images to use as test set.
        seed: Random seed for train/test split.
    """
    from sklearn.model_selection import train_test_split
    from ultralytics import YOLO

    data_dir = config["data_dir"]
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Wipe existing data
    for d in [train_dir, test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    image_ids = collect_image_ids(config["imgdir"])
    logging.info(f"Found {len(image_ids)} valid images (id > 1000)")

    train_ids, test_ids = train_test_split(
        image_ids, test_size=test_ratio, random_state=seed
    )
    logging.info(f"Split: {len(train_ids)} train, {len(test_ids)} test")

    model = YOLO(config["yolo_obb_path"])

    total_written = 0
    total_skipped = 0

    for imgid in train_ids:
        w, s = process_image_annotated(imgid, model, config, train_dir)
        total_written += w
        total_skipped += s

    for imgid in test_ids:
        w, s = process_image_annotated(imgid, model, config, test_dir)
        total_written += w
        total_skipped += s

    logging.info(f"Done. Written: {total_written}, Skipped: {total_skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Lizard preprocessing: raw images + TPS + YOLO OBB → .pt crops"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    run_preprocessing(config, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
