import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
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


def process_image(imgid, model, config, output_dir):
    img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
    if img is None:
        print(f"[{imgid}] WARNING: could not read image, skipping")
        return 0, 0

    tps = get_tps_coords(imgid, img, config["tps_data_dir"])

    results = model(img, verbose=False)
    if results[0].obb is None:
        return 0, 0

    crops, tps_local, stats, Ms = crop_toe_boxes_obb(results[0], img, tps, imgid)

    base_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    written = 0
    skipped = 0

    for i, (crop, kps, M) in enumerate(zip(crops, tps_local, Ms)):
        if i > 1:
            break

        orig_h, orig_w = crop.shape[:2]

        aug = base_transform(image=crop, keypoints=kps.tolist())
        img_aug = aug["image"]
        kps_aug = np.array(aug["keypoints"], dtype=np.float32)

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
            },
            out_path,
        )
        written += 1

    return written, skipped


def main():
    parser = argparse.ArgumentParser(description="Unified preprocessing for all alternative models")
    parser.add_argument("--config", type=str, default="preprocess_config.json")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config["data_dir"]

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

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

    for imgid in train_ids:
        print(f"Processing train image {imgid}", end="\r", flush=True)
        w, s = process_image(imgid, model, config, train_dir)
        total_written += w
        total_skipped += s
        train_count += w

    for imgid in test_ids:
        print(f"Processing test image {imgid}", end="\r", flush=True)
        w, s = process_image(imgid, model, config, test_dir)
        total_written += w
        total_skipped += s
        test_count += w

    print()
    print("=" * 50)
    print(f"Total crops written: {total_written}")
    print(f"Total crops skipped: {total_skipped}")
    print(f"Train crops: {train_count}")
    print(f"Test crops:  {test_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
