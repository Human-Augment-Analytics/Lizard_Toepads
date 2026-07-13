"""
HRNet heatmap regression training script for WFLW.

Parallel to alternative-models/hrnet-gcn/train_wflw.py — same conventions,
same split JSON format, same WFLWDataset, same evaluation metrics.

Uses the paper-faithful HRNetHeatmap model:
  - HRNet-W18 backbone (ImageNet pretrained)
  - Highest-resolution branch (128×128) → 1×1 conv → 98 heatmaps
  - Gaussian MSE loss
  - Soft-argmax coordinate extraction
  - Step LR decay at epochs 60 and 90

This is the primary comparison baseline for the sample complexity study.
Train at multiple data fractions (10%, 25%, 50%, 80%, 100%) using the
same splits as the GCN model and compare NME curves.

Usage:
    python train_heatmap_wflw.py \\
        --split /path/to/splits/wflw_0.8_seed42.json \\
        --config configs/wflw-config.json

Checkpoint: checkpoints/hrnet_heatmap_wflw_best.pth
Log:        logs/hrnet_heatmap_wflw.log
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

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from hrnet_heatmap import HRNetHeatmap, make_gaussian_heatmaps, hard_argmax

# WFLWHeatmapDataset: paper-faithful augmentation (flip + rotation + scale)
# Used instead of WFLWDataset which lacks rotation.
import importlib.util as _ilu2
_hm_ds_path = Path(_SCRIPT_DIR) / "wflw_heatmap_dataset.py"
_hm_spec = _ilu2.spec_from_file_location("wflw_heatmap_dataset", str(_hm_ds_path))
_hm_mod = _ilu2.module_from_spec(_hm_spec)
_hm_spec.loader.exec_module(_hm_mod)
WFLWHeatmapDataset = _hm_mod.WFLWHeatmapDataset

MODEL_NAME = "hrnet_heatmap_wflw"
SCRIPT_DIR = _SCRIPT_DIR

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


def save_overlay(img_tensor, coords_pred, coords_gt, heatmaps, save_path, input_size=256):
    """Save side-by-side: coordinate overlay (left) + max heatmap (right)."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Resize to input_size if needed
    if img_bgr.shape[0] != input_size:
        img_bgr = cv2.resize(img_bgr, (input_size, input_size))

    # Left: coordinate overlay
    coord_panel = img_bgr.copy()
    for x, y in (coords_gt.cpu().numpy() * input_size):
        cv2.circle(coord_panel, (int(x), int(y)), 2, (0, 255, 0), -1)
    for x, y in (coords_pred.cpu().numpy() * input_size):
        cv2.circle(coord_panel, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Right: max heatmap across landmarks — shows whether peaks are forming
    hm_np  = heatmaps.cpu().numpy()              # (K, H, W)
    hm_max = hm_np.max(axis=0)                   # (H, W)
    hm_vis = (hm_max / (hm_max.max() + 1e-8) * 255).astype(np.uint8)
    hm_col = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
    hm_col = cv2.resize(hm_col, (input_size, input_size))

    combined = np.concatenate([coord_panel, hm_col], axis=1)
    cv2.imwrite(str(save_path), combined)


def compute_nme_batch(pred_coords, gt_coords):
    """Per-batch mean NME normalised by inter-ocular distance (landmarks 60, 72)."""
    IOD_L, IOD_R = 60, 72
    iod = (gt_coords[:, IOD_L] - gt_coords[:, IOD_R]).norm(dim=-1)  # (B,)
    dists = (pred_coords - gt_coords).norm(dim=-1)                   # (B, N)
    nme_per_sample = dists.mean(dim=-1) / iod.clamp(min=1e-6)       # (B,)
    return nme_per_sample.sum().item()


def main():
    parser = argparse.ArgumentParser(
        description="Train HRNet heatmap regression on WFLW"
    )
    parser.add_argument("--split", type=str, required=True,
                        help="Path to split JSON with train/val/test keys")
    parser.add_argument(
        "--config", type=str,
        default=str(SCRIPT_DIR / "configs" / "wflw-config.json"),
        help="Path to WFLW training config JSON",
    )
    args = parser.parse_args()

    config_path = _Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)

    setup_logging()
    logging.info(f"Config: {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    num_landmarks = cfg.get("num_landmarks", 98)
    input_size    = cfg.get("input_size", 512)
    heatmap_size  = cfg.get("heatmap_size", 128)
    sigma         = cfg.get("sigma", 2.0)
    epochs        = cfg.get("epochs", 120)
    batch_size    = cfg.get("batch_size", 32)
    val_batch     = cfg.get("val_batch_size", 64)
    num_workers   = cfg.get("num_workers", 4)

    # ── Datasets ──────────────────────────────────────────────────────────
    split_path = _Path(args.split)
    if not split_path.exists():
        logging.error(f"Split file not found: {split_path}")
        sys.exit(1)
    with open(split_path) as f:
        split_data = json.load(f)

    train_dataset = WFLWHeatmapDataset(
        split_data["train"], input_size=input_size,
        num_landmarks=num_landmarks, augment=True,
    )
    val_dataset = WFLWHeatmapDataset(
        split_data["val"], input_size=input_size,
        num_landmarks=num_landmarks, augment=False,
    )
    test_files = split_data.get("test", [])
    test_dataset = WFLWHeatmapDataset(
        test_files, input_size=input_size,
        num_landmarks=num_landmarks, augment=False,
    ) if test_files else None
    logging.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset) if test_dataset else 0}"
    )

    if len(val_dataset) == 0:
        logging.error("Val set is empty.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=val_batch, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    ) if test_dataset else None

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

    # Plain Adam matching paper exactly (WD=0, no weight decay)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr_backbone"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"]
    )
    logging.info(
        f"LR: backbone={cfg['lr_backbone']}, head={cfg['lr_head']} | "
        f"step decay at {cfg['lr_milestones']}, gamma={cfg['lr_gamma']}"
    )

    criterion = torch.nn.MSELoss()

    ckpt_dir = SCRIPT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = SCRIPT_DIR / "visualizations" / MODEL_NAME
    vis_dir.mkdir(parents=True, exist_ok=True)

    best_nme = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for imgs, coords_gt, _, _flipped in train_loader:
            imgs      = imgs.to(device)
            coords_gt = coords_gt.to(device)

            target_hm = make_gaussian_heatmaps(
                coords_gt, heatmap_size=heatmap_size, sigma=sigma
            )
            pred_hm, _ = model(imgs)

            # Paper-faithful: heatmap MSE only, no coordinate loss.
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
        nme_total = 0.0
        px_err_total = 0.0

        with torch.no_grad():
            for imgs, coords_gt, _, _flipped in val_loader:
                imgs      = imgs.to(device)
                coords_gt = coords_gt.to(device)

                target_hm = make_gaussian_heatmaps(
                    coords_gt, heatmap_size=heatmap_size, sigma=sigma
                )
                pred_hm, _ = model(imgs)

                val_loss  += criterion(pred_hm, target_hm).item() * imgs.size(0)
                # Use hard argmax for NME — matches paper's decode_preds (argmax)
                # rather than soft-argmax which averages over diffuse peaks.
                coords_pred = hard_argmax(pred_hm)
                nme_total += compute_nme_batch(coords_pred, coords_gt)
                # Per-sample mean pixel error in 512px space, summed over batch
                px_err_total += (
                    (coords_pred - coords_gt).norm(dim=-1).mean(dim=-1).sum().item() * 512
                )

        val_loss  /= len(val_dataset)
        val_nme    = nme_total / len(val_dataset)
        val_px_err = px_err_total / len(val_dataset)

        logging.info(
            f"Epoch {epoch}/{epochs}, "
            f"Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, "
            f"Val NME: {val_nme:.4f}, "
            f"Val Pixel Error (512px): {val_px_err:.2f}"
        )

        scheduler.step()

        if val_nme < best_nme:
            best_nme = val_nme
            torch.save(
                model.state_dict(),
                str(ckpt_dir / f"{MODEL_NAME}_best.pth"),
            )

        # Overlays: every epoch for first 20, then every 10
        if epoch <= 20 or epoch % 10 == 0:
            hm_vis = torch.sigmoid(pred_hm[0])   # sigmoid only for visualization
            save_overlay(
                imgs[0], coords_pred[0], coords_gt[0], hm_vis,
                save_path=vis_dir / f"overlay_epoch{epoch:04d}.jpg",
                input_size=input_size,
            )

    logging.info(f"Training complete. Best val NME: {best_nme:.4f}")

    # ── Test set evaluation ────────────────────────────────────────────────
    # Load the best checkpoint and evaluate on the held-out official test split.
    # This is the number that should be compared against the paper's 4.6 NME —
    # not the val NME logged during training, which uses a sampled subset.
    if test_loader is not None:
        best_ckpt = str(ckpt_dir / f"{MODEL_NAME}_best.pth")
        logging.info(f"Loading best checkpoint for test evaluation: {best_ckpt}")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        model.eval()

        test_nme_total = 0.0
        test_px_total  = 0.0

        with torch.no_grad():
            for imgs, coords_gt, _, _flipped in test_loader:
                imgs      = imgs.to(device)
                coords_gt = coords_gt.to(device)

                pred_hm, _ = model(imgs)
                coords_pred = hard_argmax(pred_hm)

                test_nme_total += compute_nme_batch(coords_pred, coords_gt)
                test_px_total  += (
                    (coords_pred - coords_gt).norm(dim=-1).mean(dim=-1).sum().item() * 512
                )

        test_nme = test_nme_total / len(test_dataset)
        test_px  = test_px_total  / len(test_dataset)

        logging.info(
            f"Test NME: {test_nme:.4f}, "
            f"Test Pixel Error (512px): {test_px:.2f} "
            f"[{len(test_dataset)} samples]"
        )
    else:
        logging.warning(
            "No 'test' key in split JSON — skipping test evaluation. "
            "Re-run generate_split.py to include a test set."
        )


if __name__ == "__main__":
    main()
