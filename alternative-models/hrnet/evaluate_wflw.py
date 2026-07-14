"""
WFLW NME evaluator for HRNet heatmap regression.

Mirrors alternative-datasets/WFLW/evaluate_wflw.py exactly — same metrics,
same subset breakdown, same output format — for direct comparison with the
GCN model results.

Computes NME, FR@0.1, AUC@0.1 on the full WFLW test set and all six
attribute subsets (pose, expression, illumination, makeup, occlusion, blur).

Usage:
    python evaluate_wflw.py \\
        --checkpoint checkpoints/hrnet_heatmap_wflw_best.pth \\
        --split /path/to/splits/wflw_0.8_seed42.json \\
        --config configs/wflw-config.json \\
        --output-json results/wflw_eval_hrnet_0.8.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

SCRIPT_DIR = Path(__file__).parent.resolve()
ALT_DATASETS = SCRIPT_DIR.parent.parent / "alternative-datasets"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ALT_DATASETS))

# Import reference decode_preds + compute_nme for paper-faithful evaluation
_REF_REPO = SCRIPT_DIR.parent.parent.parent / "HRNet-Facial-Landmark-Detection"
if not _REF_REPO.exists():
    _cwd = Path.cwd()
    for _c in [_cwd, _cwd.parent, _cwd.parent.parent]:
        if (_c / "HRNet-Facial-Landmark-Detection").exists():
            _REF_REPO = _c / "HRNet-Facial-Landmark-Detection"
            break
if str(_REF_REPO) not in sys.path:
    sys.path.insert(0, str(_REF_REPO))

from lib.core.evaluation import decode_preds as ref_decode_preds, compute_nme as ref_compute_nme
from hrnet_heatmap import HRNetHeatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ATTR_NAMES    = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]
TARGET_SIZE   = 512
IOD_LM_LEFT   = 60   # outer right eye corner
IOD_LM_RIGHT  = 72   # outer left eye corner
FR_THRESHOLD  = 0.10
AUC_THRESHOLD = 0.10
AUC_STEPS     = 1000


def compute_nme(pred_px, gt_px):
    iod = float(np.linalg.norm(gt_px[IOD_LM_LEFT] - gt_px[IOD_LM_RIGHT]))
    if iod <= 0:
        return None
    return float(np.linalg.norm(pred_px - gt_px, axis=1).mean() / iod)


def compute_auc(nme_list, threshold=AUC_THRESHOLD):
    if not nme_list:
        return 0.0
    nme_arr = np.array(nme_list)
    xs  = np.linspace(0, threshold, AUC_STEPS + 1)
    ced = np.array([(nme_arr <= x).mean() for x in xs])
    return float(np.trapz(ced, xs) / threshold)


def compute_fr(nme_list, threshold=FR_THRESHOLD):
    if not nme_list:
        return 0.0
    return float((np.array(nme_list) > threshold).mean())


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HRNet heatmap regression on WFLW — NME / FR / AUC"
    )
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--split",       type=str, required=True,
                        help="Split JSON with 'test' key")
    parser.add_argument("--config",      type=str, required=True,
                        help="Path to wflw-config.json")
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    for path, label in [
        (args.checkpoint, "checkpoint"),
        (args.split,      "split"),
        (args.config,     "config"),
    ]:
        if not Path(path).exists():
            logging.error(f"{label} not found: {path}")
            sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    num_landmarks = cfg.get("num_landmarks", 98)
    heatmap_size  = cfg.get("heatmap_size", 128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    with open(args.split) as f:
        split_data = json.load(f)
    test_files = split_data.get("test", [])
    if not test_files:
        logging.error("No test files in split JSON")
        sys.exit(1)
    logging.info(f"Evaluating on {len(test_files)} test samples")

    model = HRNetHeatmap(
        num_landmarks=num_landmarks,
        pretrained=False,
        heatmap_size=heatmap_size,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    nme_buckets = {k: [] for k in ["full"] + ATTR_NAMES}
    skipped = 0

    with torch.no_grad():
        for pt_path in test_files:
            try:
                data  = torch.load(pt_path, map_location="cpu")
                img_np  = data["image"].permute(1, 2, 0).numpy()
                gt_norm = data["tps"].numpy()
                attrs   = data["attrs"].numpy()
            except Exception as e:
                logging.warning(f"Failed to load {pt_path}: {e}")
                skipped += 1
                continue

            img_f    = img_np.astype(np.float32) / 255.0
            # Resize to the training input size (model was trained at 256px)
            input_size = cfg.get("input_size", 256)
            img_np_resized = cv2.resize(img_np, (input_size, input_size))
            img_f    = img_np_resized.astype(np.float32) / 255.0
            img_norm = (img_f - IMAGENET_MEAN) / IMAGENET_STD
            img_t    = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

            heatmaps, _ = model(img_t)
            # Paper-faithful evaluation: use decode_preds (argmax + sub-pixel
            # refinement + inverse affine to 512px space) — same as training script.
            score_map = heatmaps.cpu()
            center = torch.Tensor([256.0, 256.0])
            scale  = 512.0 / 200.0  # = 2.56, fixed for pre-cropped 512px images
            preds = ref_decode_preds(score_map, center.unsqueeze(0), torch.tensor([scale]), [64, 64])
            # preds: (1, 98, 2) in 512px space
            pred_px = preds[0].numpy()
            gt_px   = gt_norm * TARGET_SIZE

            nme = compute_nme(pred_px, gt_px)
            if nme is None:
                logging.warning(f"Zero IOD in {pt_path}, skipping")
                skipped += 1
                continue

            nme_buckets["full"].append(nme)
            for i, attr in enumerate(ATTR_NAMES):
                if attrs[i] == 1:
                    nme_buckets[attr].append(nme)

    logging.info(
        f"Evaluation complete. Samples: {len(nme_buckets['full'])}, Skipped: {skipped}"
    )

    subset_keys = ["full"] + ATTR_NAMES

    def mean_or_none(lst):
        return float(np.mean(lst)) if lst else None

    results = {
        "nme":    {k: mean_or_none(nme_buckets[k]) for k in subset_keys},
        "fr":     {k: compute_fr(nme_buckets[k])   for k in subset_keys},
        "auc":    {k: compute_auc(nme_buckets[k])  for k in subset_keys},
        "counts": {k: len(nme_buckets[k])           for k in subset_keys},
    }

    logging.info(f"\n{'Subset':<16} {'NME':>8} {'FR@0.1':>8} {'AUC@0.1':>9} {'N':>6}")
    logging.info("-" * 52)
    for k in subset_keys:
        nme_v = results["nme"][k]
        fr_v  = results["fr"][k]
        auc_v = results["auc"][k]
        n     = results["counts"][k]
        nme_s = f"{nme_v:.4f}" if nme_v is not None else "  N/A"
        logging.info(f"{k:<16} {nme_s:>8} {fr_v:>8.4f} {auc_v:>9.4f} {n:>6}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
