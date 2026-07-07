"""
Training script for HRNet heatmap regression (paper-faithful baseline).

Follows the same CLI conventions as the other alternative-models train.py scripts:
    --config  path to JSON config (default: configs/heatmap_default.json)
    --split   path to shared split JSON (train/val keys with .pt file lists)
    --data    path to data dir (fallback if no --split provided)

Checkpoint: checkpoints/hrnet_heatmap_best.pth
Log:        logs/hrnet_heatmap.log
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from hrnet_heatmap import HRNetHeatmap, make_gaussian_heatmaps
from dataset import LizardDataset

MODEL_NAME = "hrnet_heatmap"
SCRIPT_DIR = Path(__file__).parent.resolve()

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def setup_logging():
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(str(log_dir / f"{MODEL_NAME}.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_arg: str | None) -> dict:
    """Load config from path or name. Falls back to heatmap_default.json."""
    if config_arg is not None:
        # Try as a direct path first, then as a name under configs/
        candidates = [
            Path(config_arg),
            SCRIPT_DIR / "configs" / config_arg,
            SCRIPT_DIR / "configs" / f"{config_arg}.json",
        ]
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        print(f"WARNING: config '{config_arg}' not found, using defaults", file=sys.stderr)

    default = SCRIPT_DIR / "configs" / "heatmap_default.json"
    if default.exists():
        with open(default) as f:
            return json.load(f)

    # Hardcoded fallback so the script always runs
    return {
        "num_landmarks": 9,
        "input_size": 512,
        "heatmap_size": 128,
        "sigma": 2.0,
        "lr_backbone": 1e-5,
        "lr_head": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 120,
        "val_fraction": 0.2,
        "num_workers": 4,
        "lr_milestones": [60, 90],
        "lr_gamma": 0.1,
        "grad_clip": 1.0,
    }


def save_overlay(img_tensor, coords_pred, coords_gt, save_path, input_size=512):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for x, y in (coords_gt.cpu().numpy() * input_size):
        cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 255, 0), -1)
    for x, y in (coords_pred.cpu().numpy() * input_size):
        cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.imwrite(str(save_path), img_bgr)


def main():
    parser = argparse.ArgumentParser(
        description="Train HRNet heatmap regression (paper-faithful)"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Config name or path (default: configs/heatmap_default.json)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data directory containing .pt files")
    parser.add_argument("--split", type=str, default=None,
                        help="Path to split JSON with 'train'/'val' keys")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    num_landmarks = cfg["num_landmarks"]
    input_size    = cfg["input_size"]
    heatmap_size  = cfg["heatmap_size"]
    sigma         = cfg["sigma"]
    epochs        = cfg["epochs"]
    batch_size    = cfg["batch_size"]
    num_workers   = cfg.get("num_workers", 4)

    # ── Datasets ──────────────────────────────────────────────────────────
    if args.split:
        split_path = Path(args.split)
        if not split_path.exists():
            logging.error(f"Split file not found: {split_path}")
            sys.exit(1)
        with open(split_path) as f:
            split_data = json.load(f)
        train_dataset = LizardDataset(split_data["train"], input_size=input_size)
        val_dataset   = LizardDataset(split_data["val"],   input_size=input_size)
    elif args.data:
        data_dir = Path(args.data)
        pt_files = sorted(data_dir.rglob("*.pt"))
        if not pt_files:
            logging.error(f"No .pt files found in {data_dir}")
            sys.exit(1)
        dataset  = LizardDataset([str(p) for p in pt_files], input_size=input_size)
        val_len  = int(len(dataset) * cfg["val_fraction"])
        train_dataset, val_dataset = random_split(
            dataset, [len(dataset) - val_len, val_len]
        )
    else:
        # Try the config's training_data_path
        data_path = cfg.get("training_data_path") or cfg.get("training_data_dir")
        if not data_path:
            logging.error("No --split, --data, or training_data_path in config")
            sys.exit(1)
        pt_files = sorted(Path(data_path).rglob("*.pt"))
        dataset  = LizardDataset([str(p) for p in pt_files], input_size=input_size)
        val_len  = int(len(dataset) * cfg["val_fraction"])
        train_dataset, val_dataset = random_split(
            dataset, [len(dataset) - val_len, val_len]
        )

    logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = HRNetHeatmap(
        num_landmarks=num_landmarks,
        pretrained=True,
        heatmap_size=heatmap_size,
    ).to(device)
    logging.info(
        f"HRNetHeatmap | landmarks={num_landmarks} | "
        f"heatmap_size={heatmap_size} | sigma={sigma}"
    )

    # ── Optimiser — lower LR for pretrained backbone ───────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["lr_backbone"]},
        {"params": model.head.parameters(),     "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["lr_milestones"],
        gamma=cfg["lr_gamma"],
    )

    criterion = torch.nn.MSELoss()

    ckpt_dir = SCRIPT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = SCRIPT_DIR / "visualizations" / MODEL_NAME
    vis_dir.mkdir(parents=True, exist_ok=True)

    best_px_err = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for imgs, coords_gt in train_loader:
            imgs      = imgs.to(device)
            coords_gt = coords_gt.to(device)

            target_hm = make_gaussian_heatmaps(
                coords_gt, heatmap_size=heatmap_size, sigma=sigma
            )
            pred_hm, _ = model(imgs)
            loss = criterion(pred_hm, target_hm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_dataset)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        px_err   = 0.0

        with torch.no_grad():
            for imgs, coords_gt in val_loader:
                imgs      = imgs.to(device)
                coords_gt = coords_gt.to(device)

                target_hm = make_gaussian_heatmaps(
                    coords_gt, heatmap_size=heatmap_size, sigma=sigma
                )
                pred_hm, coords_pred = model(imgs)

                val_loss += criterion(pred_hm, target_hm).item() * imgs.size(0)
                px_err   += (
                    (coords_pred - coords_gt).norm(dim=-1).sum().item() * input_size
                )

        val_loss /= len(val_dataset)
        px_err   /= len(val_dataset)

        logging.info(
            f"Epoch {epoch}/{epochs}, "
            f"Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, "
            f"Avg Pixel Error: {px_err:.2f}"
        )

        scheduler.step()

        if px_err < best_px_err:
            best_px_err = px_err
            torch.save(
                model.state_dict(),
                str(ckpt_dir / f"{MODEL_NAME}_best.pth"),
            )

        # Overlays: every epoch for first 20, then every 10
        if epoch <= 20 or epoch % 10 == 0:
            save_overlay(
                imgs[0], coords_pred[0], coords_gt[0],
                save_path=vis_dir / f"overlay_epoch{epoch:04d}.jpg",
                input_size=input_size,
            )

    logging.info(f"Training complete. Best val pixel error: {best_px_err:.2f}px")


if __name__ == "__main__":
    main()
