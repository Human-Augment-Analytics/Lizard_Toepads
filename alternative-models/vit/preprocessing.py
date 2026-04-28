import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultralytics import YOLO
import numpy as np
import cv2
import torch
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import json

from common.tps_utils import get_tps_coords
from common.yolo_utils import crop_toe_boxes
from common.obb_utils import crop_toe_boxes_obb

IMAGENORMALIZE = A.Compose(
    [
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(
            min_height=224,
            min_width=224,
            border_mode=0,
            value=0,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=False
    )
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def preprocess_image(crop, tps):
    aug = IMAGENORMALIZE(image=crop, keypoints=tps)
    return aug["image"], aug["keypoints"]


def produce_training(r, img, tps_coords, training_data_dir, name="sample"):
    crops, box_coords, local_tps_coords = crop_toe_boxes(r, img, tps_coords, False)
    for i in range(len(crops)):
        if len(local_tps_coords[i]) != 9:
            continue
        rescaled, r_coords = preprocess_image(crops[i], local_tps_coords[i])
        data = {
            "image": rescaled,
            "keypoints": torch.tensor(r_coords, dtype=torch.float32)
        }
        torch.save(data, f"{training_data_dir}/vit/{name}_{i}.pt")


def produce_training_obb(r, img, tps_coords, training_data_dir, name="sample"):
    crops, local_tps_coords, stats, Ms = crop_toe_boxes_obb(r[0], img, tps_coords, name)
    for i in range(len(crops)):
        if len(local_tps_coords[i]) != 9:
            continue
        rescaled, r_coords = preprocess_image(crops[i], local_tps_coords[i].tolist())
        data = {
            "image": rescaled,
            "keypoints": torch.tensor(r_coords, dtype=torch.float32)
        }
        torch.save(data, f"{training_data_dir}/vit/{name}_{i}.pt")


def process_image(imgid, model, config):
    img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
    tps = get_tps_coords(imgid, img, config["tps_data_dir"])
    r = model(img, verbose=False)
    produce_training(r, img, tps, config["training_data_dir"], imgid)


def process_images(config):
    model = YOLO(config["yolo_path"])
    dir_path = Path(config["imgdir"])
    Path(f"{config['training_data_dir']}/vit").mkdir(parents=True, exist_ok=True)
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
    p = Path("configs/default.json")
    with open(p, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for ViT")
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--mode", type=str, required=False, default="axis", choices=["axis", "obb"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "obb":
        model = YOLO(config["yolo_obb_path"])
        dir_path = Path(config["imgdir"])
        Path(f"{config['training_data_dir']}/vit").mkdir(parents=True, exist_ok=True)
        count = 0
        for file in dir_path.iterdir():
            print(f"Processing file {count}", end="\r", flush=True)
            try:
                if ".jpg" in file.name:
                    imgid = file.name.replace(".jpg", "")
                    if int(imgid) > 1000:
                        img = cv2.imread(f"{config['imgdir']}/{imgid}.jpg")
                        tps = get_tps_coords(imgid, img, config["tps_data_dir"])
                        r = model(img, verbose=False)
                        produce_training_obb(r, img, tps, config["training_data_dir"], imgid)
                count += 1
            except Exception as e:
                count += 1
                print()
                print(f"Failed to process file {file}: {e}")
                break
    else:
        process_images(config)
