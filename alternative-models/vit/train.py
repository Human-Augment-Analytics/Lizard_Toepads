import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import logging
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from pathlib import Path
import argparse
import json

from model import ViTLandmark
from dataset import ViTDataset

MODEL_NAME = "vit"

def setup_logging():
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{MODEL_NAME}.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )


def load_config(config_name):
    if config_name is not None:
        p = Path(f"configs/{config_name}.json")
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    p = Path("configs/default.json")
    with open(p, "r") as f:
        return json.load(f)


def main(args):
    config = load_config(args.config)
    setup_logging()

    data_dir = args.data if args.data is not None else config.get("training_data_dir", "../data/training_data")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ViTLandmark(pretrained=True).to(device)
    b_params = model.backbone.parameters()
    h_params = model.head.parameters()
    optimizer = optim.SGD([
        {'params': b_params, 'lr': config["backbone_lr"]},
        {'params': h_params, 'lr': config["head_lr"]},
    ], momentum=config["momentum"])

    criterion = nn.SmoothL1Loss()
    dataset = ViTDataset(f"{data_dir}/train")
    val_fraction = config["val_fraction"]
    val_len = int(len(dataset) * val_fraction)
    train_len = len(dataset) - val_len

    if args.split:
        if not Path(args.split).exists():
            print(f"ERROR: split file not found: {args.split}", file=sys.stderr)
            sys.exit(1)
        with open(args.split) as f:
            split_data = json.load(f)
        train_dataset = ViTDataset(split_data["train"])
        val_dataset = ViTDataset(split_data["val"])
    else:
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

    for p in model.backbone.parameters():
        p.requires_grad = False

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    best_pixel_error = float('inf')

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        bc = 1
        for images, keypoints in train_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            k_flat = keypoints.view(keypoints.size(0), -1)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, k_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_max_norm"])
            optimizer.step()
            total_loss += loss.item()
            print(f"\rBatch {bc}/{len(train_loader)}, Loss {loss}", end='')
            bc += 1

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch}/{config['epochs']} - Loss: {avg_loss:.6f}")

        model.eval()
        val_loss = 0
        total_pixel_error = 0
        val_batches = 0
        with torch.no_grad():
            for images, keypoints in val_loader:
                images = images.to(device)
                keypoints = keypoints.to(device)
                k_flat = keypoints.view(keypoints.size(0), -1)

                preds = model(images)
                loss = criterion(preds, k_flat)
                val_loss += loss.item()

                preds_px = preds * 224
                k_px = k_flat * 224
                pixel_error = torch.mean(torch.abs(preds_px - k_px)).item()
                total_pixel_error += pixel_error
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_pixel_error = total_pixel_error / val_batches
        logging.info(f"Epoch {epoch}/{config['epochs']} - Val Loss: {avg_val_loss:.6f}, Avg Pixel Error: {avg_pixel_error:.2f}")

        if avg_pixel_error < best_pixel_error:
            best_pixel_error = avg_pixel_error
            torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}_best.pth")
            logging.info(f"New best model saved with pixel error {best_pixel_error:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT Landmark Model")
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--data", type=str, required=False)
    parser.add_argument("--split", type=str, required=False, default=None)
    args = parser.parse_args()
    main(args)
