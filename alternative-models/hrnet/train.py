import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import torch
import logging
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import json

from model import HRNetLandmarkModel
from dataset import LizardDataset

MODEL_NAME = "hrnet"

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


def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_landmarks(image_tensor, coords, save_path):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = denormalize(img)
    H, W, _ = img.shape
    coords = coords.cpu().numpy()
    for i, (x, y) in enumerate(coords):
        px = int(x * W)
        py = int(y * H)
        cv2.circle(img, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(
            img,
            str(i),
            (px + 4, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1
        )
    cv2.imwrite(save_path, img)


def main(args):
    config = load_config(args.config)
    setup_logging()

    DATA_PATH = args.data if args.data is not None else config["training_data_dir"]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = config["epochs"]
    BATCH_SIZE = config["batch_size"]
    VALIDATION_FRACTION = config["val_fraction"]
    LR_BACKBONE = config["lr_backbone"]
    LR_HEAD = config["lr_head"]
    INPUT_SIZE = config["input_size"]

    OVERLAY_DIR = os.path.join(DATA_PATH, "overlays")
    Path(OVERLAY_DIR).mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    heatmaps_dir = os.path.join(DATA_PATH, "heatmaps")
    if not os.path.isdir(heatmaps_dir):
        raise FileNotFoundError(f"Heatmaps directory not found: {heatmaps_dir}")
    pt_files = [os.path.join(heatmaps_dir, f) for f in os.listdir(heatmaps_dir) if f.endswith(".pt")]

    dataset = LizardDataset(pt_files, input_size=INPUT_SIZE)

    val_len = int(len(dataset) * VALIDATION_FRACTION)
    train_len = len(dataset) - val_len

    if args.split:
        if not Path(args.split).exists():
            print(f"ERROR: split file not found: {args.split}", file=sys.stderr)
            sys.exit(1)
        with open(args.split) as f:
            split_data = json.load(f)
        train_dataset = LizardDataset(split_data["train"], input_size=INPUT_SIZE)
        val_dataset = LizardDataset(split_data["val"], input_size=INPUT_SIZE)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    model = HRNetLandmarkModel(num_heads=config.get("num_heads", 8)).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': model.cross_attn.parameters(), 'lr': LR_HEAD},
        {'params': model.self_attn.parameters(), 'lr': LR_HEAD},
        {'params': model.coord_head.parameters(), 'lr': LR_HEAD},
        {'params': [model.landmark_queries], 'lr': LR_HEAD}
    ], weight_decay=config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"]
    )

    criterion = torch.nn.SmoothL1Loss()

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        for imgs, coords_gt in train_loader:
            imgs = imgs.to(DEVICE)
            coords_gt = coords_gt.to(DEVICE)

            optimizer.zero_grad()

            coords_pred = model(imgs)

            loss = criterion(coords_pred, coords_gt)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_max_norm"])

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, (imgs, coords_gt) in enumerate(val_loader):
                imgs = imgs.to(DEVICE)
                coords_gt = coords_gt.to(DEVICE)

                coords_pred = model(imgs)

                loss = criterion(coords_pred, coords_gt)

                val_loss += loss.item()

                if batch_idx == 0:
                    overlay_landmarks(
                        imgs[0],
                        coords_pred[0],
                        os.path.join(OVERLAY_DIR, f"epoch{epoch}.png")
                    )

        avg_val_loss = val_loss / len(val_loader)

        logging.info(
            f"Epoch {epoch} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {avg_val_loss:.4f}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRNet Landmark Model")
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--data", type=str, required=False)
    parser.add_argument("--split", type=str, required=False, default=None)
    args = parser.parse_args()
    main(args)
