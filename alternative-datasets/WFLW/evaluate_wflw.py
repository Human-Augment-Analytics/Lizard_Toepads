"""
WFLW NME evaluator for Model A (HRNet-GCN with mean-shape initialization).

Computes Normalized Mean Error (NME), Failure Rate (FR@0.1), and AUC@0.1
on the full WFLW test set and each of the six attribute subsets
(pose, expression, illumination, makeup, occlusion, blur).

NME = mean(||pred_i - gt_i||_2) / inter_ocular_distance   (per image)
then averaged across all test images.

FR  = fraction of test images where NME > FR_THRESHOLD (default 0.1)

AUC = area under the Cumulative Error Distribution curve from 0 to
      AUC_THRESHOLD (default 0.1), normalised to [0, 1].

Inter-ocular distance = ||gt[60] - gt[72]|| in 512-pixel space
(outer eye corners in WFLW's 98-point scheme).

Output JSON format:
{
    "nme":  {"full": float, "pose": float, ...},
    "fr":   {"full": float, "pose": float, ...},
    "auc":  {"full": float, "pose": float, ...},
    "counts": {"full": int, "pose": int, ...}
}

Usage:
    python evaluate_wflw.py \\
        --checkpoint ../../alternative-models/hrnet-gcn/checkpoints/hrnet_gcn_wflw_best.pth \\
        --split splits/wflw_1.0_seed42.json \\
        --mean-shape mean_shapes/mean_shape_wflw.pt \\
        --config ../../alternative-models/hrnet-gcn/wflw-config.json \\
        --output-json results/wflw_eval.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
ALT_MODELS_DIR = SCRIPT_DIR.parent.parent / "alternative-models"
sys.path.insert(0, str(ALT_MODELS_DIR / "hrnet-gcn"))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hrnet_gcn import HRNetGNN
from common.graph_topologies import get_edge_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ATTR_NAMES = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]
TARGET_SIZE = 512

# Inter-ocular distance landmarks (WFLW 98-point: outer eye corners)
IOD_LM_LEFT = 60   # right outer eye corner (from viewer's left)
IOD_LM_RIGHT = 72  # left outer eye corner (from viewer's right)

FR_THRESHOLD  = 0.10   # NME > this → failure
AUC_THRESHOLD = 0.10   # CED curve integrated up to this NME value
AUC_STEPS     = 1000   # resolution of the AUC numerical integration


def compute_nme(pred_px: np.ndarray, gt_px: np.ndarray):
    """Per-image NME normalised by inter-ocular distance.

    Returns None when IOD is zero (degenerate annotation).
    """
    iod = float(np.linalg.norm(gt_px[IOD_LM_LEFT] - gt_px[IOD_LM_RIGHT]))
    if iod <= 0:
        return None
    dists = np.linalg.norm(pred_px - gt_px, axis=1)  # (N,)
    return float(dists.mean() / iod)


def compute_auc(nme_list: list, threshold: float = AUC_THRESHOLD) -> float:
    """Area under the CED curve, normalised to [0, 1].

    CED(x) = fraction of samples with NME <= x.
    AUC = integral from 0 to threshold of CED(x) dx, divided by threshold.
    """
    if not nme_list:
        return 0.0
    nme_arr = np.array(nme_list)
    xs = np.linspace(0, threshold, AUC_STEPS + 1)
    ced = np.array([(nme_arr <= x).mean() for x in xs])
    # trapezoid rule, then normalise by threshold so result is in [0,1]
    return float(np.trapz(ced, xs) / threshold)


def compute_fr(nme_list: list, threshold: float = FR_THRESHOLD) -> float:
    """Fraction of samples where NME > threshold."""
    if not nme_list:
        return 0.0
    nme_arr = np.array(nme_list)
    return float((nme_arr > threshold).mean())


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HRNet-GCN (mean-init) on WFLW — NME / FR / AUC"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, required=True,
                        help="Split JSON with 'test' key")
    parser.add_argument("--mean-shape", type=str, required=True,
                        help="Path to mean_shape .pt file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to wflw-config.json")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Output path for results JSON")
    args = parser.parse_args()

    # Validate inputs
    for path, label in [
        (args.checkpoint, "checkpoint"),
        (args.split, "split"),
        (args.mean_shape, "mean-shape"),
        (args.config, "config"),
    ]:
        if not Path(path).exists():
            logging.error(f"{label} not found: {path}")
            sys.exit(1)

    # Load config — use raw JSON to avoid dependency on Lizard config class
    with open(args.config) as f:
        cfg = json.load(f)

    class _Cfg:
        pass
    config = _Cfg()
    config.num_landmarks  = cfg.get("num_landmarks", 98)
    config.feat_dim       = cfg.get("feat_dim", 64)
    config.gnn_hidden     = cfg.get("gnn_hidden", 128)
    config.num_layers     = cfg.get("num_layers", 3)
    config.num_iters      = cfg.get("num_iters", 4)
    config.graph_topology = cfg.get("graph_topology", "wflw")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load split
    with open(args.split) as f:
        split_data = json.load(f)
    test_files = split_data.get("test", [])
    if not test_files:
        logging.error("No test files found in split JSON")
        sys.exit(1)
    logging.info(f"Evaluating on {len(test_files)} test samples")

    # Load mean shape and edge index
    mean_shape = torch.load(args.mean_shape, map_location=device)
    edge_index = get_edge_index(config.graph_topology, config.num_landmarks).to(device)

    # Load model
    model = HRNetGNN(
        hrnet_backbone="hrnet_w18",
        feat_dim=config.feat_dim,
        gnn_hidden=config.gnn_hidden,
        num_layers=config.num_layers,
        num_landmarks=config.num_landmarks,
        num_iters=config.num_iters,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Accumulate per-sample NME values per subset
    nme_buckets = {name: [] for name in ["full"] + ATTR_NAMES}
    skipped = 0

    with torch.no_grad():
        for pt_path in test_files:
            try:
                data = torch.load(pt_path, map_location="cpu")
                img_np = data["image"].permute(1, 2, 0).numpy()   # HWC uint8
                gt_norm = data["tps"].numpy()                       # (N, 2) [0,1]
                attrs = data["attrs"].numpy()                       # (6,) uint8
            except Exception as e:
                logging.warning(f"Failed to load {pt_path}: {e}")
                skipped += 1
                continue

            # Normalise image
            img_f = img_np.astype(np.float32) / 255.0
            img_norm = (img_f - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = (
                torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
            )

            # Inference — clean mean shape, no noise
            initial_coords = mean_shape.unsqueeze(0)
            pred_norm = model(img_tensor, initial_coords, edge_index)

            pred_px = pred_norm[0].cpu().numpy() * TARGET_SIZE   # (N, 2)
            gt_px   = gt_norm * TARGET_SIZE                       # (N, 2)

            nme = compute_nme(pred_px, gt_px)
            if nme is None:
                logging.warning(f"Zero inter-ocular distance in {pt_path}, skipping")
                skipped += 1
                continue

            nme_buckets["full"].append(nme)
            for i, attr_name in enumerate(ATTR_NAMES):
                if attrs[i] == 1:
                    nme_buckets[attr_name].append(nme)

    logging.info(
        f"Evaluation complete. Samples: {len(nme_buckets['full'])}, Skipped: {skipped}"
    )

    subset_keys = ["full"] + ATTR_NAMES

    def mean_or_none(lst):
        return float(np.mean(lst)) if lst else None

    results = {
        "nme":  {k: mean_or_none(nme_buckets[k]) for k in subset_keys},
        "fr":   {k: compute_fr(nme_buckets[k])   for k in subset_keys},
        "auc":  {k: compute_auc(nme_buckets[k])  for k in subset_keys},
        "counts": {k: len(nme_buckets[k])         for k in subset_keys},
    }

    # Log summary table
    logging.info(f"\n{'Subset':<16} {'NME':>8} {'FR@0.1':>8} {'AUC@0.1':>9} {'N':>6}")
    logging.info("-" * 52)
    for k in subset_keys:
        nme_v  = results["nme"][k]
        fr_v   = results["fr"][k]
        auc_v  = results["auc"][k]
        n      = results["counts"][k]
        nme_s  = f"{nme_v:.4f}" if nme_v is not None else "  N/A"
        logging.info(f"{k:<16} {nme_s:>8} {fr_v:>8.4f} {auc_v:>9.4f} {n:>6}")

    # Save JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
