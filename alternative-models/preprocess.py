import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import shutil
import numpy as np
import cv2
import torch
from pathlib import Path
import albumentations as A
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

from common.tps_utils import get_tps_coords
from common.obb_utils import crop_toe_boxes_obb
from common.heatmap_utils import tps_to_heatmap


def load_config(config_path):
    p = Path(config_path)
    if not p.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        config = json.load(f)
    required = ["yolo_obb_path", "imgdir", "tps_data_dir", "data_dir"]
    for key in required:
        if key not in config:
            print(f"ERROR: required key '{key}' missing from config", file=sys.stderr)
            sys.exit(1)
    return config


def collect_image_ids(imgdir):
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


def resize_and_pad(crop, keypoints=None, max_size=512):
    """Resize crop to fit in max_size x max_size with center padding.
    
    Returns uint8 image and transformed keypoints, plus scale/pad info
    for reversibility.
    """
    h, w = crop.shape[:2]
    scale = min(max_size / h, max_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(crop, (new_w, new_h))

    pad_x = (max_size - new_w) // 2
    pad_y = (max_size - new_h) // 2
    padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    if keypoints is not None:
        kps = np.array(keypoints, dtype=np.float32).copy()
        kps[:, 0] = kps[:, 0] * scale + pad_x
        kps[:, 1] = kps[:, 1] * scale + pad_y
        return padded, kps, scale, pad_x, pad_y

    return padded, scale, pad_x, pad_y


def process_image_annotated(imgid, model, config, output_dir):
    """Process LHS crops from original image (annotated with TPS ground truth)."""
    img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
    if img is None:
        print(f"[{imgid}] WARNING: could not read image, skipping")
        return 0, 0

    tps = get_tps_coords(imgid, img, config["tps_data_dir"])

    results = model(img, verbose=False)
    if results[0].obb is None:
        return 0, 0

    crops, tps_local, stats, Ms = crop_toe_boxes_obb(results[0], img, tps, imgid)

    from common.obb_utils import CLASSMAP, TRAIN_CLASSES, crop_obb_from_corners, transform_keypoints
    crop_classes = []
    if results[0].obb is not None:
        cls_ids = results[0].obb.cls.cpu().numpy()
        obb_corners = results[0].obb.xyxyxyxy.cpu().numpy()
        for corners, cls_id in zip(obb_corners, cls_ids):
            cls_id = int(cls_id)
            if cls_id not in TRAIN_CLASSES:
                continue
            class_name = CLASSMAP[cls_id]
            global_kps = tps.get(class_name, [])
            if len(global_kps) == 0 or len(global_kps) != 9:
                continue
            crop_check, M_check = crop_obb_from_corners(img, corners)
            local_kps = transform_keypoints(global_kps, M_check)
            h_c, w_c = crop_check.shape[:2]
            in_bounds = (local_kps[:, 0] >= 0) & (local_kps[:, 0] < w_c) & \
                        (local_kps[:, 1] >= 0) & (local_kps[:, 1] < h_c)
            if not np.all(in_bounds):
                continue
            crop_classes.append(class_name)

    written = 0
    skipped = 0

    for i, (crop, kps, M) in enumerate(zip(crops, tps_local, Ms)):
        if i > 1:
            break

        class_name = crop_classes[i] if i < len(crop_classes) else "unknown"
        orig_h, orig_w = crop.shape[:2]

        img_aug, kps_aug, scale, pad_x, pad_y = resize_and_pad(crop, kps)

        if kps_aug.shape[0] != 9:
            print(f"\n[{imgid}_{i}] WARNING: keypoint count {kps_aug.shape[0]} != 9, skipping")
            skipped += 1
            continue

        heatmap = tps_to_heatmap(kps_aug, img_aug, sigma=6)
        heatmap_chw = np.transpose(heatmap, (2, 0, 1))

        out_path = os.path.join(output_dir, f"{imgid}_{i}.pt")
        torch.save(
            {
                "image": torch.from_numpy(img_aug).permute(2, 0, 1).to(torch.uint8),
                "tps": torch.from_numpy(kps_aug).to(torch.float32),
                "heatmap": torch.from_numpy(heatmap_chw).to(torch.float32),
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
                "M": torch.from_numpy(M).to(torch.float64),
                "scale": torch.tensor(scale, dtype=torch.float32),
                "pad": torch.tensor([pad_x, pad_y], dtype=torch.float32),
                "class_name": class_name,
            },
            out_path,
        )
        written += 1

    return written, skipped


def process_image_unannotated(imgid, model, config, output_dir):
    """Flip image horizontally, run YOLO OBB to get RHS crops (no annotation).
    
    Crops are stored in flipped orientation as uint8.
    During evaluation, predictions are flipped back for visualization.
    """
    img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
    if img is None:
        return 0, 0

    img_flipped = cv2.flip(img, 0)  # vertical flip — puts RHS (top) on bottom for YOLO detection

    results = model(img_flipped, verbose=False)
    if results[0].obb is None:
        return 0, 0

    if results[0].obb.xyxyxyxy is None:
        return 0, 0

    obb_corners = results[0].obb.xyxyxyxy.cpu().numpy()
    cls_ids = results[0].obb.cls.cpu().numpy()

    from common.obb_utils import crop_obb_from_corners, TRAIN_CLASSES

    written = 0
    crop_idx = 0

    for corners, cls_id in zip(obb_corners, cls_ids):
        cls_id = int(cls_id)
        if cls_id not in TRAIN_CLASSES:
            continue
        if crop_idx > 1:
            break

        crop, M = crop_obb_from_corners(img_flipped, corners)
        orig_h, orig_w = crop.shape[:2]

        img_aug, scale, pad_x, pad_y = resize_and_pad(crop)

        out_path = os.path.join(output_dir, f"{imgid}_flip_{crop_idx}.pt")
        torch.save(
            {
                "image": torch.from_numpy(img_aug).permute(2, 0, 1).to(torch.uint8),
                "tps": torch.zeros(9, 2, dtype=torch.float32),
                "heatmap": torch.zeros(9, 512, 512, dtype=torch.float32),
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
                "M": torch.from_numpy(M).to(torch.float64),
                "scale": torch.tensor(scale, dtype=torch.float32),
                "pad": torch.tensor([pad_x, pad_y], dtype=torch.float32),
                "flipped": torch.tensor(True),
            },
            out_path,
        )
        written += 1
        crop_idx += 1

    return written, 0


def main():
    parser = argparse.ArgumentParser(description="Unified preprocessing for all alternative models")
    parser.add_argument("--config", type=str, default="preprocess_config.json")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config["data_dir"]

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    unannotated_dir = os.path.join(data_dir, "unannotated")

    # Wipe existing data in output directories
    for d in [train_dir, test_dir, unannotated_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Cleared existing directory: {d}")

    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    Path(unannotated_dir).mkdir(parents=True, exist_ok=True)

    image_ids = collect_image_ids(config["imgdir"])
    print(f"Found {len(image_ids)} valid images (id > 1000)")

    train_ids, test_ids = train_test_split(
        image_ids, test_size=args.test_ratio, random_state=args.seed
    )
    print(f"Split: {len(train_ids)} train images, {len(test_ids)} test images")

    model = YOLO(config["yolo_obb_path"])

    total_written = 0
    total_skipped = 0
    train_count = 0
    test_count = 0
    unannotated_count = 0

    print("\n--- Processing LHS (annotated) crops ---")
    for imgid in train_ids:
        print(f"Processing train image {imgid}", end="\r", flush=True)
        w, s = process_image_annotated(imgid, model, config, train_dir)
        total_written += w
        total_skipped += s
        train_count += w

    for imgid in test_ids:
        print(f"Processing test image {imgid}", end="\r", flush=True)
        w, s = process_image_annotated(imgid, model, config, test_dir)
        total_written += w
        total_skipped += s
        test_count += w

    print("\n--- Processing RHS (unannotated) crops via flip ---")
    all_ids = train_ids + test_ids
    for imgid in all_ids:
        print(f"Processing unannotated image {imgid}", end="\r", flush=True)
        w, _ = process_image_unannotated(imgid, model, config, unannotated_dir)
        total_written += w
        unannotated_count += w

    print()
    print("=" * 50)
    print(f"Total crops written:  {total_written}")
    print(f"Total crops skipped:  {total_skipped}")
    print(f"Train crops:          {train_count}")
    print(f"Test crops:           {test_count}")
    print(f"Unannotated crops:    {unannotated_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
