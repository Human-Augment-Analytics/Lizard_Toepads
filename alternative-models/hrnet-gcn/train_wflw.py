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
from pathlib import Path as _Path

_SCRIPT_DIR = _Path(__file__).resolve().parent
# Insert alternative-datasets/ onto sys.path so 'common' package is importable
# Works regardless of cwd since __file__ is always absolute after resolve()
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

from hrnet_gcn import HRNetGNN
from hrnet_gcn_ms import HRNetGNN_MS
from hrnet_gcn_coord import HRNetGNN_Coord
from hrnet_gcn_fused import HRNetGNN_Fused
from hrnet_gcn_fused_global import HRNetGNN_FusedGlobal
from hrnet_gcn_hinit import HRNetGNN_HInit

# Locate wflw_dataset.py — handle case differences between systems
# (repo uses lowercase 'wflw', cluster may have uppercase 'WFLW')
import importlib.util as _ilu2
_wflw_dataset_path = None
for _candidate in ["wflw", "WFLW"]:
    _p = _ALT_DATASETS / _candidate / "wflw_dataset.py"
    if _p.exists():
        _wflw_dataset_path = _p
        break
if _wflw_dataset_path is None:
    raise FileNotFoundError(
        f"wflw_dataset.py not found under {_ALT_DATASETS}/wflw/ or {_ALT_DATASETS}/WFLW/"
    )
_ds_spec = _ilu2.spec_from_file_location("wflw_dataset", str(_wflw_dataset_path))
_ds_mod = _ilu2.module_from_spec(_ds_spec)
_ds_spec.loader.exec_module(_ds_mod)
WFLWDataset = _ds_mod.WFLWDataset
_FLIP_PERM_98 = _ds_mod._FLIP_PERM_98

# Define landmark_loss and compute_rescaled_pixel_error inline to avoid
# importing utils.py, which has a module-level dependency on common.obb_utils
# that is irrelevant to WFLW training and causes import failures.
import torch.nn.functional as _F

def landmark_loss(pred_coords, gt_coords):
    # Pure coordinate MSE — the dist_loss term from the original 9-point Lizard
    # chain loss is omitted here. It assumes consecutive landmark indices form a
    # meaningful path, which is not true for the 98-point WFLW facial topology.
    # The GCN graph structure already encodes spatial relationships.
    return _F.mse_loss(pred_coords, gt_coords)

def compute_rescaled_pixel_error(pred_coords, coords, orig_size, device="cuda"):
    pred_px = pred_coords * 512
    gt_px   = coords * 512
    orig_h  = orig_size[:, 0].to(device)
    orig_w  = orig_size[:, 1].to(device)
    scale_x = 512 / orig_w
    scale_y = 512 / orig_h
    dx = (pred_px[:, :, 0] - gt_px[:, :, 0]) / scale_x.unsqueeze(1)
    dy = (pred_px[:, :, 1] - gt_px[:, :, 1]) / scale_y.unsqueeze(1)
    error = torch.sqrt(dx**2 + dy**2)
    return error.mean(dim=1).sum().item()

# Import directly by absolute path to avoid shadowing by alternative-models/common/
import importlib.util as _ilu
_gt_spec = _ilu.spec_from_file_location(
    "alt_datasets_graph_topologies",
    str(_ALT_DATASETS / "common" / "graph_topologies.py")
)
_gt_mod = _ilu.module_from_spec(_gt_spec)
_gt_spec.loader.exec_module(_gt_mod)
get_edge_index = _gt_mod.get_edge_index

MODEL_NAME = "hrnet_gcn_wflw"
SCRIPT_DIR = _SCRIPT_DIR

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def save_overlay(img_tensor, pred_coords, gt_coords, save_path, input_size=512):
    """Draw predicted (red) and GT (green) landmarks on the actual face crop image."""
    # Denormalize image: reverse ImageNet normalization
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # HWC float
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pred = pred_coords.cpu().numpy() * input_size   # (N, 2) in pixel space
    gt   = gt_coords.cpu().numpy()   * input_size

    for x, y in gt:
        cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)   # green GT
    for x, y in pred:
        cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 0, 255), -1)   # red pred

    cv2.imwrite(str(save_path), img_bgr)


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
        default=str(SCRIPT_DIR.parent.parent / "alternative-datasets" / "wflw" / "configs" / "wflw-config.json"),
        help="Path to WFLW training config JSON",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Path to split JSON file (must have train/val/test keys)",
    )
    parser.add_argument(
        "--mean-shape",
        type=str,
        default=None,
        help="Path to mean_shape.pt (overrides mean_shape_path in config)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    # Wrap raw dict in a simple namespace for attribute-style access
    class _Cfg:
        pass
    config = _Cfg()
    config.num_landmarks    = cfg.get("num_landmarks", 98)
    config.feat_dim         = cfg.get("feat_dim", 64)
    config.gnn_hidden       = cfg.get("gnn_hidden", 128)
    config.num_layers       = cfg.get("num_layers", 3)
    config.num_iters        = cfg.get("num_iters", 4)
    config.input_size       = cfg.get("input_size", 512)
    config.epochs           = cfg.get("epochs", 150)
    config.batch_size       = cfg.get("batch_size", 32)
    config.val_batch_size   = cfg.get("val_batch_size", 64)
    config.lr               = cfg.get("lr", 1e-4)
    config.graph_topology   = cfg.get("graph_topology", "wflw")
    config.mean_shape_path  = cfg.get("mean_shape_path", None)
    config.init_noise_sigma = cfg.get("init_noise_sigma", 0.05)
    config.model_variant    = cfg.get("model_variant", "standard")  # "standard" | "multiscale" | "coord" | "fused" | "hinit"
    config.scale_indices    = cfg.get("scale_indices", [0, 1, 2, 3])
    config.use_coarse_init  = cfg.get("use_coarse_init", True)
    config.coarse_init_ramp = cfg.get("coarse_init_ramp", 20)
    config.rot_factor       = cfg.get("rot_factor", 0)
    config.heatmap_checkpoint = cfg.get("heatmap_checkpoint", None)  # for "hinit" variant
    config.lr_milestones    = cfg.get("lr_milestones", [60, 90])     # epochs to drop LR
    config.lr_gamma         = cfg.get("lr_gamma", 0.1)               # LR multiplier at each milestone
    config.grad_clip        = cfg.get("grad_clip", 0.5)              # gradient norm clip
    config.lr_backbone      = cfg.get("lr_backbone", 1e-5)           # backbone fine-tune LR
    config.weight_decay     = cfg.get("weight_decay", 1e-4)          # L2 reg on GCN head only
    # Optional: path to a .pth checkpoint whose backbone weights should replace
    # timm's ImageNet init. Accepts HRNetHeatmap checkpoints (trained via the
    # reference pipeline) or any state_dict containing "backbone.*" keys.
    # GCN head weights are always initialised from scratch regardless.
    config.backbone_pretrained_path = cfg.get("backbone_pretrained_path", None)

    setup_logging()

    split_path = Path(args.split)
    if not split_path.exists():
        logging.error(f"Split file not found: {split_path}")
        sys.exit(1)

    with open(split_path) as f:
        split_data = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # CLI --mean-shape overrides the config value
    if args.mean_shape:
        config.mean_shape_path = args.mean_shape

    # Load mean shape
    if not config.mean_shape_path or not Path(config.mean_shape_path).exists():
        logging.error(
            f"mean_shape_path not set or not found: {config.mean_shape_path}\n"
            "Run compute_mean_shape.py first and update wflw-config.json."
        )
        sys.exit(1)

    mean_shape = torch.load(config.mean_shape_path, map_location=device)
    logging.info(f"Loaded mean shape: {mean_shape.shape} from {config.mean_shape_path}")

    # Pre-compute flipped mean shape for use with horizontally-flipped training samples.
    # Since we can't track per-sample flip state through the DataLoader, we randomly
    # use the canonical or flipped mean shape per-batch during training. This ensures
    # the GCN sees both initializations and learns to refine from either orientation.
    _flip_perm = torch.tensor(_FLIP_PERM_98, dtype=torch.long, device=device)
    mean_shape_flipped = mean_shape.clone()
    mean_shape_flipped[:, 0] = 1.0 - mean_shape_flipped[:, 0]
    mean_shape_flipped = mean_shape_flipped[_flip_perm]

    # Build edge index from topology registry
    edge_index = get_edge_index(config.graph_topology, config.num_landmarks).to(device)
    logging.info(
        f"Graph topology: '{config.graph_topology}', "
        f"edge_index shape: {edge_index.shape}"
    )

    # Datasets — use WFLWDataset which supports any num_landmarks (not hardcoded to 9)
    train_dataset = WFLWDataset(
        split_data["train"],
        input_size=config.input_size,
        num_landmarks=config.num_landmarks,
        augment=True,
        rot_factor=config.rot_factor,
    )
    val_dataset = WFLWDataset(
        split_data["val"],
        input_size=config.input_size,
        num_landmarks=config.num_landmarks,
        augment=False,
    )
    logging.info(
        f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples"
    )
    if len(val_dataset) == 0:
        logging.error(
            "Val set is empty. Re-run generate_split.py with --val-fraction > 0."
        )
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.val_batch_size, shuffle=False
    )

    # Model — select variant based on config
    if config.model_variant == "multiscale":
        model = HRNetGNN_MS(
            hrnet_backbone="hrnet_w18",
            feat_dim=config.feat_dim,
            gnn_hidden=config.gnn_hidden,
            num_layers=config.num_layers,
            num_landmarks=config.num_landmarks,
            num_iters=config.num_iters,
            scale_indices=config.scale_indices,
        ).to(device)
        logging.info(f"Model: HRNetGNN_MS (multi-scale), scale_indices={config.scale_indices}")
    elif config.model_variant == "coord":
        model = HRNetGNN_Coord(
            hrnet_backbone="hrnet_w18",
            feat_dim=config.feat_dim,
            gnn_hidden=config.gnn_hidden,
            num_layers=config.num_layers,
            num_landmarks=config.num_landmarks,
            num_iters=config.num_iters,
            use_coarse_init=config.use_coarse_init,
        ).to(device)
        logging.info(
            f"Model: HRNetGNN_Coord (coord embedding, "
            f"use_coarse_init={config.use_coarse_init})"
        )
    elif config.model_variant == "fused":
        model = HRNetGNN_Fused(
            hrnet_backbone="hrnet_w18",
            feat_dim=config.feat_dim,
            gnn_hidden=config.gnn_hidden,
            num_layers=config.num_layers,
            num_landmarks=config.num_landmarks,
            num_iters=config.num_iters,
        ).to(device)
        logging.info("Model: HRNetGNN_Fused (pre-fused multi-scale feature map)")
    elif config.model_variant == "fused_global":
        model = HRNetGNN_FusedGlobal(
            hrnet_backbone="hrnet_w18",
            feat_dim=config.feat_dim,
            gnn_hidden=config.gnn_hidden,
            num_layers=config.num_layers,
            num_landmarks=config.num_landmarks,
            num_iters=config.num_iters,
        ).to(device)
        logging.info("Model: HRNetGNN_FusedGlobal (fused + GAP global + landmark embeddings)")
    elif config.model_variant == "hinit":
        if not config.heatmap_checkpoint or not Path(config.heatmap_checkpoint).exists():
            logging.error(
                f"hinit variant requires heatmap_checkpoint in config. "
                f"Got: {config.heatmap_checkpoint}"
            )
            sys.exit(1)
        model = HRNetGNN_HInit(
            heatmap_checkpoint=config.heatmap_checkpoint,
            hrnet_backbone="hrnet_w18",
            feat_dim=config.feat_dim,
            gnn_hidden=config.gnn_hidden,
            num_layers=config.num_layers,
            num_landmarks=config.num_landmarks,
            num_iters=config.num_iters,
        ).to(device)
        logging.info(
            f"Model: HRNetGNN_HInit (frozen heatmap initializer + fused GCN)\n"
            f"  heatmap_checkpoint: {config.heatmap_checkpoint}"
        )
    else:
        model = HRNetGNN(
            hrnet_backbone="hrnet_w18",
            feat_dim=config.feat_dim,
            gnn_hidden=config.gnn_hidden,
            num_layers=config.num_layers,
            num_landmarks=config.num_landmarks,
            num_iters=config.num_iters,
        ).to(device)
        logging.info("Model: HRNetGNN (standard, single scale)")

    # ── Optional backbone weight replacement ───────────────────────────────
    # If backbone_pretrained_path is set, load backbone weights from an external
    # checkpoint (e.g. a trained HRNetHeatmap .pth) instead of timm's ImageNet
    # weights. Only keys matching "backbone.*" in the GCN model are replaced;
    # GCN head weights (node_feat_proj, gnn_layers, delta_head, etc.) are
    # always freshly initialised regardless.
    if config.backbone_pretrained_path:
        bp_path = Path(config.backbone_pretrained_path)
        if not bp_path.exists():
            logging.error(
                f"backbone_pretrained_path not found: {bp_path}\n"
                "Train the reference heatmap model first and update the config."
            )
            sys.exit(1)

        ext_state = torch.load(bp_path, map_location=device)
        # Support both raw state_dicts and checkpoint dicts with a "state_dict" key
        if isinstance(ext_state, dict) and "state_dict" in ext_state:
            ext_state = ext_state["state_dict"]

        # Filter to backbone keys only, stripping the "backbone." prefix from
        # the source so they align with model.backbone.state_dict() keys.
        # The source .pth uses "backbone.<timm_key>" regardless of model variant.
        backbone_state = model.backbone.state_dict()
        matched, skipped = {}, []
        for k, v in ext_state.items():
            if k.startswith("backbone."):
                inner_key = k[len("backbone."):]
                if inner_key in backbone_state and backbone_state[inner_key].shape == v.shape:
                    matched[inner_key] = v
                else:
                    skipped.append(k)

        backbone_state.update(matched)
        model.backbone.load_state_dict(backbone_state, strict=True)
        logging.info(
            f"Backbone weights loaded from: {bp_path}\n"
            f"  Matched: {len(matched)} / {len(backbone_state)} backbone keys\n"
            f"  Skipped (shape mismatch or missing): {len(skipped)}"
        )
        if len(matched) < len(backbone_state) * 0.9:
            logging.warning(
                f"Less than 90% of backbone keys matched — "
                f"verify the source checkpoint is an HRNetHeatmap .pth."
            )

    # Separate LR for backbone (fine-tune slowly) vs GCN head (train from scratch).
    backbone_params = list(model.backbone.parameters())
    backbone_ids    = {id(p) for p in backbone_params}
    head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": config.lr_backbone, "weight_decay": 0.0},
        {"params": head_params,     "lr": config.lr,          "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
    )
    logging.info(
        f"LR: backbone={config.lr_backbone}, head={config.lr} | "
        f"step decay at epochs {config.lr_milestones}, gamma={config.lr_gamma}"
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

        for batch in train_loader:
            if config.rot_factor > 0:
                imgs, coords, _, flipped, _rot_angles = batch
            else:
                imgs, coords, _, flipped = batch

            imgs = imgs.to(device)
            coords = coords.to(device)
            B = imgs.shape[0]

            # Per-sample mean shape: use flipped mean shape for flipped samples.
            # When rotation augmentation is active the mean shape stays upright —
            # the GCN must learn to recover from the mismatch, which is the
            # rotation robustness we want. At inference there is no angle to
            # provide so training with an upright prior matches inference.
            ms_base = mean_shape.unsqueeze(0).expand(B, -1, -1)
            ms_flip = mean_shape_flipped.unsqueeze(0).expand(B, -1, -1)
            flip_mask = flipped.to(device).view(B, 1, 1).float()
            ms = ms_flip * flip_mask + ms_base * (1.0 - flip_mask)

            noise = torch.randn(B, config.num_landmarks, 2, device=device) * config.init_noise_sigma
            initial_coords = ms + noise

            # Forward pass — coord variant returns (gcn_coords, coarse_coords)
            # when use_coarse_init=True, otherwise just gcn_coords.
            out = model(imgs, initial_coords, edge_index)
            if isinstance(out, tuple):
                pred_coords, coarse_coords = out
                # Coarse init loss: zero for first coarse_init_warmup epochs,
                # then ramp 0→1 over coarse_init_ramp epochs.
                # Keeps early backbone gradients from the coarse MLP (random init)
                # from destabilising training before the GCN has converged.
                coarse_init_warmup = 15
                if epoch <= coarse_init_warmup:
                    coarse_weight = 0.0
                else:
                    coarse_weight = min(1.0, (epoch - coarse_init_warmup) / max(1, config.coarse_init_ramp))
                loss = landmark_loss(pred_coords, coords)
                if coarse_weight > 0:
                    loss = loss + coarse_weight * landmark_loss(coarse_coords, coords)
            else:
                pred_coords = out
                loss = landmark_loss(pred_coords, coords)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        epoch_loss /= len(train_dataset)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        pxerr_total = 0.0

        with torch.no_grad():
            for imgs, coords, orig_size, _ in val_loader:
                imgs = imgs.to(device)
                coords = coords.to(device)
                B = imgs.shape[0]

                # Use coarse init for GCN only after full ramp completion.
                # coarse_init_warmup (15) + coarse_init_ramp (20) = epoch 35
                coarse_init_warmup = 15
                coarse_fully_trained = coarse_init_warmup + config.coarse_init_ramp
                if hasattr(model, 'use_coarse_init') and model.use_coarse_init and epoch > coarse_fully_trained:
                    # Run a quick coarse forward to get the learned initialization
                    feat_maps = model.backbone(imgs)
                    feat_map = feat_maps[model.backbone_out_idx]
                    global_feat = feat_map.mean(dim=[2, 3])
                    coarse_flat = model.coarse_init_mlp(global_feat)
                    coarse_init = torch.sigmoid(coarse_flat.view(B, model.num_landmarks, 2))
                    initial_coords = coarse_init.detach()
                else:
                    initial_coords = mean_shape.unsqueeze(0).repeat(B, 1, 1)

                out = model(imgs, initial_coords, edge_index)
                pred_coords = out[0] if isinstance(out, tuple) else out

                val_loss_total += landmark_loss(pred_coords, coords).item() * imgs.size(0)
                pxerr_total += compute_rescaled_pixel_error(pred_coords, coords, orig_size, device)

        val_loss = val_loss_total / len(val_dataset)
        pix_err = pxerr_total / len(val_dataset)

        # Also log raw 512-space pixel error for reference (no rescaling)
        pix_err_512 = (
            (pred_coords - coords).norm(dim=-1).mean().item() * config.input_size
        )

        logging.info(
            f"Epoch {epoch}/{config.epochs}, "
            f"Train Loss: {epoch_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, "
            f"Avg Pixel Error (orig px): {pix_err:.2f}, "
            f"Avg Pixel Error (512px): {pix_err_512:.2f}"
        )

        scheduler.step()  # MultiStepLR: step every epoch regardless of val loss

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                str(ckpt_dir / f"{MODEL_NAME}_best.pth"),
            )

        # Save overlays: every epoch for first 20 to watch early convergence,
        # then every 10 epochs. Uses save_overlay which correctly denormalizes
        # the ImageNet-normalized image before drawing landmarks.
        if epoch <= 20 or epoch % 10 == 0:
            n_samples = min(3, imgs.shape[0])
            for i in range(n_samples):
                save_overlay(
                    imgs[i],
                    pred_coords[i],
                    coords[i],
                    save_path=vis_dir / f"overlay_epoch{epoch:04d}_sample{i}.jpg",
                    input_size=config.input_size,
                )

        model.train()


if __name__ == "__main__":
    main()
