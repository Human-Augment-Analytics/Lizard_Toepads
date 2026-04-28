import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultralytics import YOLO
import numpy as np
import cv2
import torch
from pathlib import Path
import albumentations as A
import argparse
import json

from common.obb_utils import crop_toe_boxes_obb
from common.tps_utils import get_tps_coords


def process_image(imgid, model, config):
    img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
    if img is None:
        print(f"[{imgid}] Missing image, skipping")
        return

    tps = get_tps_coords(imgid, img, config["tps_data_dir"])

    results = model(img, verbose=False)

    crops, local_tps_coords, stats, Ms = crop_toe_boxes_obb(
        results[0],
        img,
        tps,
        imgid,
    )

    for i, crop in enumerate(crops):
        if i > 1:
            break

        base_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=512),
                A.PadIfNeeded(
                    512,
                    512,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            ),
        )

        aug = base_transform(
            image=crop,
            keypoints=local_tps_coords[i].tolist(),
        )

        img_aug = aug["image"]
        kps_aug = np.array(aug["keypoints"], dtype=np.float32)

        if kps_aug.shape[0] != 9:
            print()
            print(f"[{imgid}] Keypoint mismatch after aug: {kps_aug.shape}")
            continue

        h_crop, w_crop = crop.shape[:2]
        pt_path = os.path.join(config["training_data_path"], f"{imgid}_{i}_b.pt")

        torch.save(
            {
                "image": torch.from_numpy(img_aug).permute(2, 0, 1).to(torch.uint8),
                "tps": torch.from_numpy(kps_aug).to(torch.float32),
                "orig_size": torch.tensor([h_crop, w_crop], dtype=torch.float32),
            },
            pt_path,
        )


def process_images(config):
    model = YOLO(config["yolo_obb_path"])
    dir_path = Path(config["imgdir"])
    Path(config["training_data_path"]).mkdir(parents=True, exist_ok=True)
    count = 0
    for file in dir_path.iterdir():
        print(f"Processing file {count}", end="\r", flush=True)
        try:
            if ".jpg" in file.name:
                imgid = file.name.replace(".jpg", "")
                if int(imgid) > 1000:
                    process_image(imgid, model, config)
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
    p = Path("default-config.json")
    with open(p, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for HRNet-GCN")
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()
    config = load_config(args.config)
    process_images(config)
