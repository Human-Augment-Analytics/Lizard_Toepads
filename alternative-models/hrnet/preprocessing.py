import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultralytics import YOLO
import numpy as np
import cv2
import torch
from pathlib import Path
import albumentations as A
import argparse
import json

from common.tps_utils import get_tps_coords
from common.yolo_utils import crop_toe_boxes
from common.heatmap_utils import tps_to_heatmap, generate_overlay
from common.obb_utils import crop_toe_boxes_obb

MAX_SIZE = 512


def process_image(imgid, model, config, save_crop=False):
    img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
    tps = get_tps_coords(imgid, img, config["tps_data_dir"])

    results = model(img, verbose=False)

    crops, box_coords, local_tps_coords = crop_toe_boxes(
        results, img, tps, output_name=imgid
    )

    for i, crop in enumerate(crops):
        if i > 1:
            break

        base_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=MAX_SIZE),
                A.PadIfNeeded(
                    MAX_SIZE,
                    MAX_SIZE,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False
            ),
        )

        aug = base_transform(
            image=crop,
            keypoints=local_tps_coords[i]
        )

        img_aug = aug["image"]
        kps_aug = np.array(aug["keypoints"])

        heatmaps = tps_to_heatmap(kps_aug, img_aug, sigma=6)

        if heatmaps.shape[2] != 9:
            print()
            print(f"Mismatch found: {heatmaps.shape}")
            continue

        ipath = f"{config['training_data_dir']}/crops/{imgid}_{i}_b.jpg"
        hpath = f"{config['training_data_dir']}/heatmaps/{imgid}_{i}_b.pt"

        if save_crop:
            overlay = generate_overlay(img_aug, heatmaps)
            cv2.imwrite(ipath, overlay)

        torch.save(
            {
                "image": torch.from_numpy(img_aug).permute(2, 0, 1).to(torch.uint8),
                "heatmap": torch.from_numpy(heatmaps).permute(2, 0, 1).to(torch.float32),
                "tps": torch.from_numpy(kps_aug).to(torch.float32),
            },
            hpath,
        )


def process_images(config):
    model = YOLO(config["yolo_path"])
    dir_path = Path(config["imgdir"])
    Path(f"{config['training_data_dir']}/crops").mkdir(parents=True, exist_ok=True)
    Path(f"{config['training_data_dir']}/heatmaps").mkdir(parents=True, exist_ok=True)
    count = 0
    for file in dir_path.iterdir():
        print(f"Processing file {count}", end="\r", flush=True)
        try:
            if ".jpg" in file.name:
                imgid = file.name.replace(".jpg", "")
                if int(imgid) > 1000:
                    process_image(imgid, model, config, count < 10)
            count += 1
        except Exception as e:
            count += 1
            print()
            print(f"Failed to process file {file}: {e}")
            break


def process_image_obb(imgid, model, config, save_crop=False):
    img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
    tps = get_tps_coords(imgid, img, config["tps_data_dir"])

    results = model(img, verbose=False)

    crops, local_tps_coords, stats, Ms = crop_toe_boxes_obb(
        results[0], img, tps, imgid
    )

    for i, crop in enumerate(crops):
        if i > 1:
            break

        base_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=MAX_SIZE),
                A.PadIfNeeded(
                    MAX_SIZE,
                    MAX_SIZE,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False
            ),
        )

        aug = base_transform(
            image=crop,
            keypoints=local_tps_coords[i].tolist()
        )

        img_aug = aug["image"]
        kps_aug = np.array(aug["keypoints"], dtype=np.float32)

        heatmaps = tps_to_heatmap(kps_aug, img_aug, sigma=6)

        if heatmaps.shape[2] != 9:
            print()
            print(f"Mismatch found: {heatmaps.shape}")
            continue

        ipath = f"{config['training_data_dir']}/crops/{imgid}_{i}_b.jpg"
        hpath = f"{config['training_data_dir']}/heatmaps/{imgid}_{i}_b.pt"

        if save_crop:
            overlay = generate_overlay(img_aug, heatmaps)
            cv2.imwrite(ipath, overlay)

        torch.save(
            {
                "image": torch.from_numpy(img_aug).permute(2, 0, 1).to(torch.uint8),
                "heatmap": torch.from_numpy(heatmaps).permute(2, 0, 1).to(torch.float32),
                "tps": torch.from_numpy(kps_aug).to(torch.float32),
            },
            hpath,
        )


def process_images_obb(config):
    model = YOLO(config["yolo_obb_path"])
    dir_path = Path(config["imgdir"])
    Path(f"{config['training_data_dir']}/crops").mkdir(parents=True, exist_ok=True)
    Path(f"{config['training_data_dir']}/heatmaps").mkdir(parents=True, exist_ok=True)
    count = 0
    for file in dir_path.iterdir():
        print(f"Processing file {count}", end="\r", flush=True)
        try:
            if ".jpg" in file.name:
                imgid = file.name.replace(".jpg", "")
                if int(imgid) > 1000:
                    process_image_obb(imgid, model, config, count < 10)
            count += 1
        except Exception as e:
            count += 1
            print()
            print(f"Failed to process file {file}: {e}")
            break


def load_config(config_name):
    if config_name is not None:
        p = Path(f"configs/{config_name}.json")
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    p = Path("configs/default.json")
    with open(p, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for HRNet")
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--mode", type=str, required=False, default="axis", choices=["axis", "obb"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "obb":
        process_images_obb(config)
    else:
        process_images(config)
