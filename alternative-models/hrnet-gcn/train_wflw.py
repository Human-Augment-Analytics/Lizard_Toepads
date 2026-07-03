"""
Model A training script — HRNet-GCN with mean-shape initialization on WFLW.

Implements its own training loop (does NOT call utils.train()) because:
  - The frozen utils.train() hardcodes a 9-point Lizard mean shape
  - WFLW requires a 98-point mean shape loaded from a .pt file
  - WFLW training uses noise-augmented initialization for pose robustness
  - WFLW uses the facial topology graph instead of a simple chain

Does NOT modify any frozen files:
  hrnet_gcn.py, train.py, utils.py, lizard_dataset.py, default-config.json

Usage:
    python train_wflw.py \\
        --config wflw-config.json \\
        --split /path/to/wflw_split.json
"""
import sys
import os

SCRIPT_DIR = __import__("pathlib").Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "alternative-datasets"))

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import HRNetGCNTrainingConfig
from hrnet_gcn import HRNetGNN
from lizard_dataset import LizardDataset
from utils import landmark_loss, compute_rescaled_pixel_error, visualize_landmarks
from common.graph_topologies import get_edge_index

MODEL_NAME = "hrnet_gcn_wflw"


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


def main():
    parser = argparse.ArgumentParser(
        description="Train HRNet-GCN (mean-init) on WFLW"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "wflw-config.json"),
        help="Path to WFLW training config JSON",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Path to split JSON file (must have train/val/test keys)",
    )
    args = parser.parse_args()

    config = HRNetGCNTrainingConfig(args.config)
    setup_logging()

    split_path = Path(args.split)
    if not split_path.exists():
        logging.error(f"Split file not found: {split_path}")
        sys.exit(1)

    with open(split_path) as f:
        split_data = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load mean shape
    if not config.mean_shape_path or not Path(config.mean_shape_path).exists():
        logging.error(
            f"mean_shape_path not set or not found: {config.mean_shape_path}\n"
            "Run compute_mean_shape.py first and update wflw-config.json."
        )
        sys.exit(1)

    mean_shape = torch.load(config.mean_shape_path, map_location=device)
    logging.info(f"Loaded mean shape: {mean_shape.shape} from {config.mean_shape_path}")

    # Build edge index from topology registry
    edge_index = get_edge_index(config.graph_topology, config.num_landmarks).to(device)
    logging.info(
        f"Graph topology: '{config.graph_topology}', "
        f"edge_index shape: {edge_index.shape}"
    )

    # Datasets
    train_dataset = LizardDataset(split_data["train"], input_size=config.input_size)
    val_dataset = LizardDataset(split_data["val"], input_size=config.input_size)
    logging.info(
        f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Model
    model = HRNetGNN(
        hrnet_backbone="hrnet_w18",
        feat_dim=config.feat_dim,
        gnn_hidden=config.gnn_hidden,
        num_layers=config.num_layers,
        num_landmarks=config.num_landmarks,
        num_iters=config.num_iters,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    ckpt_dir = SCRIPT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = SCRIPT_DIR / "visualizations" / MODEL_NAME
    vis_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, config.epochs + 1):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for imgs, coords, _ in train_loader:
            imgs = imgs.to(device)
            coords = coords.to(device)
            B = imgs.shape[0]

            # Noise-augmented mean shape initialization (train only)
            noise = torch.randn(B, config.num_landmarks, 2, device=device) * config.init_noise_sigma
            initial_coords = mean_shape.unsqueeze(0).repeat(B, 1, 1) + noise

            pred_coords = model(imgs, initial_coords, edge_index)
            loss = landmark_loss(pred_coords, coords)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        epoch_loss /= len(train_dataset)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        pxerr_total = 0.0

        with torch.no_grad():
            for imgs, coords, orig_size in val_loader:
                imgs = imgs.to(device)
                coords = coords.to(device)
                B = imgs.shape[0]

                # Clean mean shape — no noise during validation
                initial_coords = mean_shape.unsqueeze(0).repeat(B, 1, 1)
                pred_coords = model(imgs, initial_coords, edge_index)

                val_loss_total += landmark_loss(pred_coords, coords).item() * imgs.size(0)
                pxerr_total += compute_rescaled_pixel_error(pred_coords, coords, orig_size, device)

        val_loss = val_loss_total / len(val_dataset)
        pix_err = pxerr_total / len(val_dataset)

        logging.info(
            f"Epoch {epoch}/{config.epochs}, "
            f"Train Loss: {epoch_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, "
            f"Avg Pixel Error: {pix_err:.2f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                str(ckpt_dir / f"{MODEL_NAME}_best.pth"),
            )

        if epoch % 10 == 0:
            visualize_landmarks(
                imgs[0],
                pred_coords[0],
                coords[0],
                save_path=str(vis_dir / f"epoch{epoch}.jpg"),
            )

        model.train()


if __name__ == "__main__":
    main()
