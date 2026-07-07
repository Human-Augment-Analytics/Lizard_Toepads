import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import StackedHourGlass
from dataset import LizardDataset
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import json

MODEL_NAME = "stacked_hourglass"
SCRIPT_DIR = Path(__file__).parent.resolve()

def setup_logging():
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(str(log_dir / f"{MODEL_NAME}.log")),
            logging.StreamHandler(sys.stdout),
        ]
    )

def main(args):
    setup_logging()
    training_data_dir = args.data
    if training_data_dir is None:
        training_data_dir = str(SCRIPT_DIR / "data")

    configName = args.config
    config = loadConfig(configName)
    validateConfig(config)
    logging.info(config)
    ckpt_dir = initEnvironment()

    npz_dir = Path(f"{training_data_dir}/train")
    npz_paths = list(npz_dir.glob("*.pt"))
    if not npz_paths:
        npz_dir = Path(training_data_dir)
        npz_paths = list(npz_dir.glob("*.pt"))
    logging.info(f"Found {len(npz_paths)} training samples at location {npz_dir}")

    if args.split:
        if not Path(args.split).exists():
            print(f"ERROR: split file not found: {args.split}", file=sys.stderr)
            sys.exit(1)
        with open(args.split) as f:
            split_data = json.load(f)
        train_paths = [Path(p) for p in split_data["train"]]
        val_paths = [Path(p) for p in split_data["val"]]
    else:
        tsize = 1 - config["trainTestSplit"]
        train_paths, val_paths = train_test_split(npz_paths, test_size=tsize, random_state=config["randomState"])

    batch_size = config["batchSize"]

    dataset = LizardDataset(train_paths, aug_factor=config["augmentationFactor"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    valid_dataset = LizardDataset(val_paths, aug_factor=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shg = StackedHourGlass()
    shg.to(device)

    optimizer = torch.optim.SGD(
        shg.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"]
    )
    num_epochs = config["epochs"]
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        shg.train()
        running_loss = 0.0

        for imgs, gt_heatmaps in dataloader:
            imgs, gt_heatmaps = imgs.to(device), gt_heatmaps.to(device)
            optimizer.zero_grad()
            combined_hm_preds = shg(imgs)
            pred_list = [combined_hm_preds[:, i, :, :, :] for i in range(combined_hm_preds.shape[1])]
            loss = shg.calc_loss(pred_list, gt_heatmaps).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(dataloader)

        shg.eval()
        val_loss = 0.0
        stack_losses = [0.0] * shg.nstack
        with torch.no_grad():
            for imgs, gt_heatmaps in valid_dataloader:
                imgs, gt_heatmaps = imgs.to(device), gt_heatmaps.to(device)
                preds = shg(imgs)
                pred_list = [preds[:, i, :, :, :] for i in range(preds.shape[1])]
                per_stack = shg.calc_loss(pred_list, gt_heatmaps)  # (B, nstack)
                val_loss += per_stack.mean().item()
                for s in range(shg.nstack):
                    stack_losses[s] += per_stack[:, s].mean().item()
        avg_val_loss = val_loss / len(valid_dataloader)
        stack_loss_str = " | ".join(
            f"Stack{s+1}: {stack_losses[s]/len(valid_dataloader):.4f}"
            for s in range(shg.nstack)
        )
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(shg.state_dict(), str(ckpt_dir / f"{MODEL_NAME}_best.pth"))
        logging.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {stack_loss_str}")

def initEnvironment():
    ckpt_dir = SCRIPT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir

def loadConfig(cname):
    if cname != None:
        p = SCRIPT_DIR / "configs" / f"{cname}.json"
        if p.exists():
            try:
                with open(p, "r") as f:
                    config = json.load(f)
                    return config
            except Exception as e:
                print(f"Unable to load config {cname} at path {p}")
    else:
        return loadDefaultConfig()
    return None

def loadDefaultConfig():
    p = SCRIPT_DIR / "configs" / "default.json"
    with open(p, "r") as f:
        config = json.load(f)
        return config

def validateConfig(config):
    default = loadDefaultConfig()
    for key in default:
        if key not in config:
            config[key] = default[key]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stacked Hourglass")

    parser.add_argument("--config", type=str, required=False, help="Name of config file to use in config directory")
    parser.add_argument("--data", type=str, required=False, help="Path to training data directory")
    parser.add_argument("--split", type=str, required=False, default=None, help="Path to shared split JSON file")

    args = parser.parse_args()
    main(args)
