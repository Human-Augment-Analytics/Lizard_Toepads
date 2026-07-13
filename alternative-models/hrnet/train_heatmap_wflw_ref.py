"""
HRNet heatmap regression training script — reference pipeline.

Supersedes _stash/train_heatmap_wflw.py.

Key differences from the stashed baseline:
  - Uses WFLWPtDataset which calls the reference crop() for augmentation,
    recovering context-rich scale jitter + rotation on the 512px image.
  - Evaluation uses decode_preds() + compute_nme() from the reference repo,
    which back-transforms predictions to 512px space and normalises NME
    by inter-ocular distance — exactly matching the paper's eval protocol.
  - scheduler.step() is called at the start of each epoch (before training),
    matching the reference tools/train.py behaviour.

Hyperparameters from wflw-config.json match the paper YAML exactly:
  lr=0.0001, WD=0.0, milestones=[30,50], gamma=0.1, epochs=60,
  sigma=1.5, heatmap_size=64, input_size=256, batch_size=16.

Usage:
    python train_heatmap_wflw_ref.py \\
        --split /path/to/splits/wflw_1.0_seed42.json \\
        --config configs/wflw-config.json

Checkpoint: checkpoints/hrnet_heatmap_wflw_ref_best.pth
Log:        logs/hrnet_heatmap_wflw_ref.log
"""
import sys
import argparse
import json
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Resolve paths ────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_WORKSPACE   = _SCRIPT_DIR.parent.parent.parent
_REF_REPO    = _WORKSPACE / "HRNet-Facial-Landmark-Detection"
_REF_LIB     = _REF_REPO / "lib"

# Verify the path resolved correctly — fail early with a clear message if not
if not _REF_LIB.exists():
    # Fallback: walk up from cwd looking for the reference repo
    _cwd = Path.cwd()
    for _candidate in [_cwd, _cwd.parent, _cwd.parent.parent]:
        _try = _candidate / "HRNet-Facial-Landmark-Detection"
        if _try.exists():
            _REF_REPO = _try
            _REF_LIB  = _REF_REPO / "lib"
            break
    else:
        print(
            f"ERROR: HRNet-Facial-Landmark-Detection not found.\n"
            f"  Tried: {_WORKSPACE / 'HRNet-Facial-Landmark-Detection'}\n"
            f"  Ensure the reference repo is checked out alongside Lizard_Toepads.",
            file=sys.stderr,
        )
        sys.exit(1)

# Add the REPO ROOT (parent of lib) to sys.path so that relative imports inside
# the reference package (e.g. from ..utils.transforms import ...) resolve correctly.
# With this, we import as: from lib.core.evaluation import decode_preds
_REF_ROOT = str(_REF_REPO)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

print(f"[train_heatmap_wflw_ref] ref repo path: {_REF_REPO}", flush=True)

# Reference evaluation functions — imported as lib.core.evaluation because
# evaluation.py uses relative imports (from ..utils.transforms import ...)
# which require lib to be a package, not a path entry.
from lib.core.evaluation import decode_preds, compute_nme as ref_compute_nme

from hrnet_heatmap import HRNetHeatmap, hard_argmax
from wflw_pt_dataset import WFLWPtDataset

MODEL_NAME  = "hrnet_heatmap_wflw_ref"
SCRIPT_DIR  = _SCRIPT_DIR

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


def save_overlay(img_tensor, coords_pred, coords_gt, heatmaps, save_path,
                 input_size=256):
    """Side-by-side: coordinate overlay (left) + max heatmap (right)."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img_bgr.shape[0] != input_size:
        img_bgr = cv2.resize(img_bgr, (input_size, input_size))

    coord_panel = img_bgr.copy()
    for x, y in (coords_gt.cpu().numpy() * input_size):
        cv2.circle(coord_panel, (int(x), int(y)), 2, (0, 255, 0), -1)
    for x, y in (coords_pred.cpu().numpy() * input_size):
        cv2.circle(coord_panel, (int(x), int(y)), 2, (0, 0, 255), -1)

    hm_np  = heatmaps.cpu().numpy()
    hm_max = hm_np.max(axis=0)
    hm_vis = (hm_max / (hm_max.max() + 1e-8) * 255).astype(np.uint8)
    hm_col = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
    hm_col = cv2.resize(hm_col, (input_size, input_size))

    combined = np.concatenate([coord_panel, hm_col], axis=1)
    cv2.imwrite(str(save_path), combined)


def run_nme_eval(model, loader, dataset_size, device, heatmap_size):
    """Run NME evaluation using reference decode_preds + compute_nme.

    Returns: (nme_avg, px_err_avg)
      nme_avg:    mean NME over the split (reference eval, 512px space)
      px_err_avg: mean per-landmark pixel error in 512px space
    """
    model.eval()
    nme_count  = 0
    nme_sum    = 0.0
    px_sum     = 0.0

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs = imgs.to(device)
            pred_hm, _ = model(imgs)

            score_map = pred_hm.cpu()

            # decode_preds: argmax + sub-pixel refinement + inverse affine
            # → predictions in 512px coordinate space
            preds = decode_preds(
                score_map,
                meta["center"],
                meta["scale"],
                [heatmap_size, heatmap_size],
            )  # (B, 98, 2) in 512px space

            # compute_nme: normalises by inter-ocular distance (LM 60 and 72)
            nme_batch = ref_compute_nme(preds, meta)   # (B,) per-sample NME
            nme_sum   += np.sum(nme_batch)
            nme_count += imgs.size(0)

            # Also track pixel error in 512px space using hard_argmax
            coords_norm = hard_argmax(pred_hm)   # (B, 98, 2) in [0,1]
            gt_norm = meta["pts"].to(device) / 512.0
            px_sum += (
                (coords_norm - gt_norm).norm(dim=-1)
                .mean(dim=-1).sum().item() * 512
            )

    nme_avg  = nme_sum  / nme_count
    px_avg   = px_sum   / dataset_size
    return nme_avg, px_avg


def main():
    parser = argparse.ArgumentParser(
        description="Train HRNet heatmap regression on WFLW — reference pipeline"
    )
    parser.add_argument("--split", type=str, required=True,
                        help="Path to split JSON with train/val/test keys")
    parser.add_argument(
        "--config", type=str,
        default=str(SCRIPT_DIR / "configs" / "wflw-config.json"),
        help="Path to config JSON (default: configs/wflw-config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)

    setup_logging()
    logging.info(f"Config: {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # ── Hyperparameters ───────────────────────────────────────────────────
    num_landmarks = cfg.get("num_landmarks", 98)
    heatmap_size  = cfg.get("heatmap_size",  64)
    epochs        = cfg.get("epochs",        60)
    batch_size    = cfg.get("batch_size",    16)
    val_batch     = cfg.get("val_batch_size", 32)
    num_workers   = cfg.get("num_workers",   4)
    lr            = cfg.get("lr_backbone",   1e-4)
    weight_decay  = cfg.get("weight_decay",  0.0)
    lr_milestones = cfg.get("lr_milestones", [30, 50])
    lr_gamma      = cfg.get("lr_gamma",      0.1)
    grad_clip     = cfg.get("grad_clip",     1.0)

    # ── Datasets ──────────────────────────────────────────────────────────
    split_path = Path(args.split)
    if not split_path.exists():
        logging.error(f"Split file not found: {split_path}")
        sys.exit(1)
    with open(split_path) as f:
        split_data = json.load(f)

    train_dataset = WFLWPtDataset(split_data["train"], augment=True)
    val_dataset   = WFLWPtDataset(split_data["val"],   augment=False)
    test_files    = split_data.get("test", [])
    test_dataset  = WFLWPtDataset(test_files, augment=False) if test_files else None

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
        f"HRNetHeatmap | landmarks={num_landmarks} | heatmap_size={heatmap_size}"
    )

    # ── Optimizer — single Adam, WD=0, matching paper exactly ─────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # MultiStepLR — called at the START of each epoch (before training),
    # matching reference tools/train.py which does lr_scheduler.step() then train().
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_gamma
    )
    logging.info(
        f"LR={lr}, WD={weight_decay} | "
        f"MultiStepLR milestones={lr_milestones}, gamma={lr_gamma}"
    )

    criterion = torch.nn.MSELoss()

    # Suppress the "scheduler.step() before optimizer.step()" warning —
    # this is intentional to match reference tools/train.py behaviour.
    warnings.filterwarnings(
        "ignore",
        message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
        category=UserWarning,
    )

    ckpt_dir = SCRIPT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = SCRIPT_DIR / "visualizations" / MODEL_NAME
    vis_dir.mkdir(parents=True, exist_ok=True)

    best_nme = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        # Step LR at epoch start (matches reference tools/train.py)
        scheduler.step()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for imgs, targets, meta in train_loader:
            imgs    = imgs.to(device)
            targets = targets.to(device)

            pred_hm, _ = model(imgs)
            loss = criterion(pred_hm, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_dataset)

        # ── Validate ──────────────────────────────────────────────────────
        val_nme, val_px_err = run_nme_eval(
            model, val_loader, len(val_dataset), device, heatmap_size
        )

        logging.info(
            f"Epoch {epoch}/{epochs}, "
            f"Train Loss: {train_loss:.6f}, "
            f"Val NME: {val_nme:.4f}, "
            f"Val Pixel Error (512px): {val_px_err:.2f}"
        )

        if val_nme < best_nme:
            best_nme = val_nme
            torch.save(
                model.state_dict(),
                str(ckpt_dir / f"{MODEL_NAME}_best.pth"),
            )

        # Overlays: every epoch for first 20, then every 10
        if epoch <= 20 or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Use last batch from val_loader for overlay
                for imgs_v, _, meta_v in val_loader:
                    imgs_v = imgs_v.to(device)
                    pred_hm_v, _ = model(imgs_v)
                    coords_pred_v = hard_argmax(pred_hm_v)
                    gt_norm_v     = meta_v["pts"].to(device) / 512.0
                    hm_vis_v      = torch.sigmoid(pred_hm_v[0])
                    save_overlay(
                        imgs_v[0], coords_pred_v[0], gt_norm_v[0], hm_vis_v,
                        save_path=vis_dir / f"overlay_epoch{epoch:04d}.jpg",
                    )
                    break
            model.train()

    logging.info(f"Training complete. Best val NME: {best_nme:.4f}")

    # ── Test evaluation ────────────────────────────────────────────────────
    # Load the best checkpoint and evaluate on the official held-out test split.
    # This is the number to compare against the paper's 4.60% NME.
    if test_loader is not None:
        best_ckpt = str(ckpt_dir / f"{MODEL_NAME}_best.pth")
        logging.info(f"Loading best checkpoint for test evaluation: {best_ckpt}")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        test_nme, test_px = run_nme_eval(
            model, test_loader, len(test_dataset), device, heatmap_size
        )
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
