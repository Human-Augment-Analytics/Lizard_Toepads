"""
Model B training script — HRNet-GCN with image-conditioned initialization.

Trains HRNetGNNWithInit using dual-loss supervision:
    loss = loss_init_weight * landmark_loss(initial_coords, gt)
         + loss_final_weight * landmark_loss(final_coords, gt)

Checkpoint selection is based on final_coords validation loss only.

Does NOT modify any frozen files:
  hrnet_gcn.py, train.py, utils.py, lizard_dataset.py, default-config.json

Validates on Lizard first (hinit-config.json, num_landmarks=9) before WFLW.

Usage:
    python train_hinit.py \\
        --config hinit-config.json \\
        --split /path/to/lizard_split.json
"""
import sys
import os
from pathlib import Path as _Path

_SCRIPT_DIR = _Path(__file__).resolve().parent
_ALT_DATASETS = _SCRIPT_DIR.parent.parent / "alternative-datasets"
if str(_ALT_DATASETS) not in sys.path:
    sys.path.insert(0, str(_ALT_DATASETS))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import HRNetGCNTrainingConfig
from hrnet_gcn_hinit import HRNetGNNWithInit
from lizard_dataset import LizardDataset
from utils import landmark_loss, compute_rescaled_pixel_error, visualize_landmarks

# Import directly by absolute path to avoid shadowing by alternative-models/common/
import importlib.util as _ilu
_gt_spec = _ilu.spec_from_file_location(
    "alt_datasets_graph_topologies",
    str(_ALT_DATASETS / "common" / "graph_topologies.py")
)
_gt_mod = _ilu.module_from_spec(_gt_spec)
_gt_spec.loader.exec_module(_gt_mod)
get_edge_index = _gt_mod.get_edge_index
make_chain_edge_index = _gt_mod.make_chain_edge_index

MODEL_NAME = "hrnet_gcn_hinit"
SCRIPT_DIR = _SCRIPT_DIR


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
        description="Train HRNet-GCN with image-conditioned initialization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "hinit-config.json"),
        help="Path to training config JSON",
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

    # Edge index from topology registry or fallback to chain (Lizard case)
    if config.graph_topology is not None:
        edge_index = get_edge_index(config.graph_topology, config.num_landmarks).to(device)
        logging.info(f"Graph topology: '{config.graph_topology}'")
    else:
        edge_index = make_chain_edge_index(config.num_landmarks).to(device)
        logging.info(f"Graph topology: chain (fallback), num_landmarks={config.num_landmarks}")

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
    model = HRNetGNNWithInit(
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

            initial_coords, final_coords = model(imgs, edge_index)

            loss = (
                config.loss_init_weight * landmark_loss(initial_coords, coords)
                + config.loss_final_weight * landmark_loss(final_coords, coords)
            )

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

                _, final_coords = model(imgs, edge_index)

                # Checkpoint selection based on final_coords val loss only
                val_loss_total += landmark_loss(final_coords, coords).item() * imgs.size(0)
                pxerr_total += compute_rescaled_pixel_error(final_coords, coords, orig_size, device)

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
                final_coords[0],
                coords[0],
                save_path=str(vis_dir / f"epoch{epoch}.jpg"),
            )

        model.train()


if __name__ == "__main__":
    main()
