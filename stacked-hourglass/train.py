from model.stackedhourglass import StackedHourGlass
from lizarddataset import LizardDataset
from lizarddataset_pt import LizardDatasetPT
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader, random_split
import logging
import sys
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
        ],
    )


def main(args):
    setup_logging()
    configName = args.config
    config = loadConfig(configName)
    validateConfig(config)
    logging.info(config)
    initEnvironment()

    sigma = args.sigma if args.sigma is not None else config.get("sigma", 5.0)
    input_size = config.get("inputSize", 512)
    heatmap_size = config.get("heatmapSize", 128)

    # ── Dataset loading ───────────────────────────────────────────────────
    # Mode A: --split JSON (same format as HRNet pipeline, .pt files)
    # Mode B: --data pointing at a dir of .pt files
    # Mode C: --data pointing at legacy training_data dir with heatmaps/*.npz
    if args.split:
        split_path = Path(args.split)
        with open(split_path) as f:
            split_data = json.load(f)
        train_dataset = LizardDatasetPT(
            split_data["train"], input_size=input_size,
            heatmap_size=heatmap_size, sigma=sigma
        )
        val_dataset = LizardDatasetPT(
            split_data["val"], input_size=input_size,
            heatmap_size=heatmap_size, sigma=sigma
        )
        logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    elif args.data and Path(args.data).is_dir():
        data_dir = Path(args.data)
        pt_files = sorted(data_dir.rglob("*.pt"))

        if pt_files:
            val_frac = 1.0 - config["trainTestSplit"]
            train_paths, val_paths = train_test_split(
                pt_files, test_size=val_frac, random_state=config["randomState"]
            )
            train_dataset = LizardDatasetPT(
                [str(p) for p in train_paths], input_size=input_size,
                heatmap_size=heatmap_size, sigma=sigma
            )
            val_dataset = LizardDatasetPT(
                [str(p) for p in val_paths], input_size=input_size,
                heatmap_size=heatmap_size, sigma=sigma
            )
            logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        else:
            # Legacy .npz path
            npz_dir = data_dir / "heatmaps"
            npz_paths = list(npz_dir.glob("*.npz"))
            if not npz_paths:
                logging.error(f"No .pt or .npz files found under {data_dir}")
                return
            val_frac = 1.0 - config["trainTestSplit"]
            train_paths, val_paths = train_test_split(
                npz_paths, test_size=val_frac, random_state=config["randomState"]
            )
            train_dataset = LizardDataset(train_paths, aug_factor=config["augmentationFactor"])
            val_dataset   = LizardDataset(val_paths,   aug_factor=1)
            logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    else:
        training_data_dir = args.data if args.data else "./data/training_data"
        npz_dir = Path(training_data_dir) / "heatmaps"
        npz_paths = list(npz_dir.glob("*.npz"))
        if not npz_paths:
            logging.error(f"No .npz files found at {npz_dir}")
            return
        val_frac = 1.0 - config["trainTestSplit"]
        train_paths, val_paths = train_test_split(
            npz_paths, test_size=val_frac, random_state=config["randomState"]
        )
        train_dataset = LizardDataset(train_paths, aug_factor=config["augmentationFactor"])
        val_dataset   = LizardDataset(val_paths,   aug_factor=1)
        logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    batch_size = config["batchSize"]
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    valid_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    shg = StackedHourGlass()
    shg.to(device)
    logging.info(f"StackedHourGlass | nstack={shg.nstack} | sigma={sigma}")

    optimizer = torch.optim.SGD(
        shg.parameters(), lr=config["initialLR"], momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config["scheduler"]["factor"],
        patience=config["scheduler"]["patience"]
    )
    num_epochs = config["epochs"]
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
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

        # ── Validation ────────────────────────────────────────────────────
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
            torch.save(shg.state_dict(), str(SCRIPT_DIR / "checkpoints" / f"{MODEL_NAME}_best.pth"))

        logging.info(
            f"Epoch {epoch}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f} | {stack_loss_str}"
        )

    logging.info(f"Training complete. Best val loss: {best_val_loss:.6f}")


def initEnvironment():
    (SCRIPT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)


def loadConfig(cname):
    if cname is not None:
        p = Path(f"./configs/{cname}.json")
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Unable to load config {cname}: {e}")
    return loadDefaultConfig()


def loadDefaultConfig():
    p = Path(f"./configs/default.json")
    with open(p) as f:
        return json.load(f)


def validateConfig(config):
    default = loadDefaultConfig()
    for key in default:
        if key not in config:
            config[key] = default[key]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stacked Hourglass")
    parser.add_argument("--config", type=str, required=False,
                        help="Name of config file to use in config directory")
    parser.add_argument("--data", type=str, required=False,
                        help="Path to directory containing .pt files or training_data dir with heatmaps/")
    parser.add_argument("--split", type=str, required=False,
                        help="Path to split JSON with 'train'/'val' keys (same format as HRNet pipeline)")
    parser.add_argument("--sigma", type=float, required=False, default=None,
                        help="Gaussian sigma for heatmap targets (overrides config)")
    args = parser.parse_args()
    main(args)
