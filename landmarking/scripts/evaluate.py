"""CLI tool for evaluating a trained landmark detection model.

Loads a checkpoint, runs inference on the test split, and outputs
dataset-appropriate metrics:
  - WFLW: NME, FR@0.1, AUC@0.1 (full + per-attribute subsets)
  - Lizard: pixel error, mm error (per-landmark)

Usage:
    # Evaluate GCN on WFLW
    python -m landmarking.scripts.evaluate \
        --config landmarking/config/defaults/wflw.json \
        --checkpoint runs/wflw/fused_global/.../checkpoints/best.pth \
        --split runs/wflw/splits/wflw_split.json \
        --output runs/wflw/fused_global/eval.json

    # Evaluate heatmap on WFLW (uses reference decode_preds)
    python -m landmarking.scripts.evaluate \
        --config landmarking/config/defaults/wflw_heatmap.json \
        --checkpoint runs/wflw/heatmap/.../checkpoints/best.pth \
        --split runs/wflw/splits/wflw_split.json \
        --output runs/wflw/heatmap/eval.json

    # Override variant at CLI
    python -m landmarking.scripts.evaluate \
        --config landmarking/config/defaults/wflw.json \
        --variant fused_global \
        --checkpoint runs/wflw/fused_global/.../checkpoints/best.pth
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config.schema import LandmarkingConfig
from ..common.graph_topologies import get_edge_index
from ..models.registry import get_model
from ..evaluation.metrics_wflw import compute_nme, compute_fr, compute_auc
from ..training.utils import set_seed, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ATTR_NAMES = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained landmark detection model."
    )
    parser.add_argument("--config", type=str, required=True, help="Config JSON path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--split", type=str, default=None, help="Split JSON path (uses test partition). If not set, uses config.dataset.split_path.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for results.")
    parser.add_argument("--variant", type=str, default=None, help="Override model variant.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument("--mean-shape", type=str, default=None, help="Override mean shape path (GCN models).")
    return parser.parse_args(argv)


def evaluate_wflw_gcn(model, test_loader, mean_shape, edge_index, device, num_landmarks):
    """Evaluate a GCN model on WFLW test set. Returns NME/FR/AUC."""
    model.eval()
    nme_buckets = {name: [] for name in ["full"] + ATTR_NAMES}

    with torch.no_grad():
        for batch in test_loader:
            imgs, coords, metadata = batch
            imgs = imgs.to(device)
            coords = coords.to(device)
            B = imgs.shape[0]

            # Mean shape initialization (no noise at eval)
            if mean_shape is not None:
                initial_coords = mean_shape.unsqueeze(0).expand(B, -1, -1)
            else:
                initial_coords = torch.full((B, num_landmarks, 2), 0.5, device=device)

            out = model(imgs, initial_coords, edge_index)
            pred_coords = out[0] if isinstance(out, tuple) else out

            # Convert to 512px space for NME
            pred_px = pred_coords.cpu().numpy() * 512.0
            gt_px = coords.cpu().numpy() * 512.0

            for i in range(B):
                nme = compute_nme(pred_px[i], gt_px[i])
                if nme is None:
                    continue
                nme_buckets["full"].append(nme)

                # Per-attribute subsets
                if "attrs" in metadata:
                    attrs = metadata["attrs"]
                    if hasattr(attrs, "numpy"):
                        attrs_np = attrs[i].numpy() if attrs.dim() > 1 else attrs.numpy()
                    else:
                        attrs_np = np.array(attrs[i]) if isinstance(attrs, list) else attrs
                    for j, attr_name in enumerate(ATTR_NAMES):
                        if j < len(attrs_np) and attrs_np[j] == 1:
                            nme_buckets[attr_name].append(nme)

    subset_keys = ["full"] + ATTR_NAMES
    results = {
        "nme": {k: float(np.mean(nme_buckets[k])) if nme_buckets[k] else None for k in subset_keys},
        "fr": {k: compute_fr(nme_buckets[k]) for k in subset_keys},
        "auc": {k: compute_auc(nme_buckets[k]) for k in subset_keys},
        "counts": {k: len(nme_buckets[k]) for k in subset_keys},
    }
    return results


def evaluate_wflw_heatmap(model, test_loader, device, heatmap_size):
    """Evaluate a heatmap model on WFLW test set using reference decode_preds."""
    from ..evaluation.decode_preds import decode_preds, compute_nme as compute_nme_ref

    model.eval()
    nme_buckets = {name: [] for name in ["full"] + ATTR_NAMES}

    with torch.no_grad():
        for batch in test_loader:
            imgs, target_hm, meta = batch
            imgs = imgs.to(device)
            B = imgs.shape[0]

            pred_hm, _ = model(imgs)
            score_map = pred_hm.cpu()

            preds = decode_preds(
                score_map, meta["center"], meta["scale"],
                [heatmap_size, heatmap_size],
            )

            nme_batch = compute_nme_ref(preds, meta)

            # Bucket by full + per-attribute subset
            for i in range(B):
                nme_val = float(nme_batch[i])
                nme_buckets["full"].append(nme_val)

                if "attrs" in meta:
                    attrs = meta["attrs"]
                    if hasattr(attrs, "numpy"):
                        attrs_i = attrs[i].numpy() if attrs.dim() > 1 else attrs.numpy()
                    elif isinstance(attrs, torch.Tensor):
                        attrs_i = attrs[i].numpy()
                    else:
                        attrs_i = np.array(attrs[i]) if isinstance(attrs, list) else None
                    if attrs_i is not None:
                        for j, attr_name in enumerate(ATTR_NAMES):
                            if j < len(attrs_i) and attrs_i[j] == 1:
                                nme_buckets[attr_name].append(nme_val)

    subset_keys = ["full"] + ATTR_NAMES
    results = {
        "nme": {k: float(np.mean(nme_buckets[k])) if nme_buckets[k] else None for k in subset_keys},
        "fr": {k: compute_fr(nme_buckets[k]) for k in subset_keys},
        "auc": {k: compute_auc(nme_buckets[k]) for k in subset_keys},
        "counts": {k: len(nme_buckets[k]) for k in subset_keys},
    }
    return results


def main(argv=None):
    args = parse_args(argv)

    # Load config
    config = LandmarkingConfig.from_json(args.config)
    config.resolve_paths()

    if args.variant:
        config.model.variant = args.variant
    if args.device:
        config.training.device = args.device
    if args.mean_shape:
        config.dataset.mean_shape_path = args.mean_shape

    device = get_device(config.training.device)
    set_seed(config.training.seed)
    is_heatmap = config.model.variant == "heatmap"

    # Load split
    split_path = args.split or config.dataset.split_path
    if split_path and Path(split_path).exists():
        with open(split_path) as f:
            split_data = json.load(f)
        test_paths = split_data.get("test", [])
    else:
        # Auto-discover test set from data directory
        data_dir = Path(config.dataset.data_dir)
        if config.dataset.name == "wflw":
            # WFLW always uses the official test set from pt_crops/test/
            test_dir = data_dir / "pt_crops" / "test"
        else:
            test_dir = data_dir / "test"

        if test_dir.exists():
            test_paths = sorted([str(p) for p in test_dir.glob("*.pt")])
            logger.info(f"Auto-discovered test set from: {test_dir}")
        else:
            logger.error(
                f"No split file and no test directory found at {test_dir}. "
                f"Provide --split or ensure test data exists."
            )
            sys.exit(1)

    if not test_paths:
        logger.error("No test files found.")
        sys.exit(1)
    logger.info(f"Test set: {len(test_paths)} samples")

    # Build model
    model_kwargs = {"num_landmarks": config.dataset.num_landmarks}
    if not is_heatmap:
        model_kwargs.update({
            "feat_dim": config.model.feat_dim,
            "gnn_hidden": config.model.gnn_hidden,
            "num_layers": config.model.num_layers,
            "num_iters": config.model.num_iters,
        })
        if config.model.variant in ("multiscale", "fused"):
            model_kwargs["scale_indices"] = config.model.scale_indices
    else:
        model_kwargs["heatmap_size"] = config.model.heatmap_size

    model = get_model(config.model.variant, **model_kwargs)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Build test dataloader
    if is_heatmap and config.dataset.name == "wflw":
        from ..datasets.wflw.dataset_ref import WFLWRefDataset
        test_ds = WFLWRefDataset(pt_paths=test_paths, augment=False)
    elif config.dataset.name == "wflw":
        from ..datasets.wflw.dataset import WFLWDataset
        test_ds = WFLWDataset(
            pt_paths=test_paths,
            input_size=config.dataset.input_size,
            num_landmarks=config.dataset.num_landmarks,
            augment=False,
        )
    else:
        from ..datasets.lizard.dataset import LizardDataset
        test_ds = LizardDataset(
            pt_paths=test_paths,
            input_size=config.dataset.input_size,
            num_landmarks=config.dataset.num_landmarks,
            augment=False,
        )

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Run evaluation
    if is_heatmap and config.dataset.name == "wflw":
        results = evaluate_wflw_heatmap(model, test_loader, device, config.model.heatmap_size)
    elif config.dataset.name == "wflw":
        # Load mean shape for GCN
        mean_shape = None
        if config.dataset.mean_shape_path and Path(config.dataset.mean_shape_path).exists():
            mean_shape = torch.load(config.dataset.mean_shape_path, map_location=device, weights_only=False)
        edge_index = get_edge_index(config.dataset.graph_topology, config.dataset.num_landmarks).to(device)
        results = evaluate_wflw_gcn(model, test_loader, mean_shape, edge_index, device, config.dataset.num_landmarks)
    else:
        # Lizard evaluation — pixel error
        from ..evaluation.metrics_lizard import compute_pixel_error
        model.eval()
        errors = []
        edge_index = get_edge_index(config.dataset.graph_topology, config.dataset.num_landmarks).to(device)
        mean_shape = None
        if config.dataset.mean_shape_path and Path(config.dataset.mean_shape_path).exists():
            mean_shape = torch.load(config.dataset.mean_shape_path, map_location=device, weights_only=False)
        with torch.no_grad():
            for batch in test_loader:
                imgs, coords, *rest = batch
                imgs = imgs.to(device)
                coords = coords.to(device)
                B = imgs.shape[0]
                if mean_shape is not None:
                    init = mean_shape.unsqueeze(0).expand(B, -1, -1)
                else:
                    init = torch.full((B, config.dataset.num_landmarks, 2), 0.5, device=device)
                out = model(imgs, init, edge_index)
                pred = out[0] if isinstance(out, tuple) else out
                pred_px = pred.cpu().numpy() * config.dataset.input_size
                gt_px = coords.cpu().numpy() * config.dataset.input_size
                for i in range(B):
                    errors.append(compute_pixel_error(pred_px[i], gt_px[i]).mean())
        results = {
            "mean_px_error": float(np.mean(errors)),
            "median_px_error": float(np.median(errors)),
            "n_evaluated": len(errors),
        }

    # Log results
    logger.info("=" * 50)
    logger.info(f"Results ({config.dataset.name} / {config.model.variant}):")
    if "nme" in results:
        for k in results["nme"]:
            nme_v = results["nme"][k]
            fr_v = results["fr"][k]
            auc_v = results["auc"][k]
            n = results["counts"][k]
            nme_s = f"{nme_v:.4f}" if nme_v is not None else "N/A"
            logger.info(f"  {k:<16} NME={nme_s}  FR={fr_v:.4f}  AUC={auc_v:.4f}  (n={n})")
    else:
        logger.info(f"  Mean pixel error: {results['mean_px_error']:.2f}")
        logger.info(f"  Median pixel error: {results['median_px_error']:.2f}")
        logger.info(f"  Samples evaluated: {results['n_evaluated']}")
    logger.info("=" * 50)

    # Save results
    output_path = args.output
    if not output_path:
        output_path = str(Path(args.checkpoint).parent.parent / "eval_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
