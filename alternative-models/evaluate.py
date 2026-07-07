import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import logging
import base64
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.tps_utils import get_tps_coords

try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SHARED_DATA_DIR = "/storage/ice-shared/cs8903onl/alternative-models/data"
TPS_FILES_DIR = "/storage/ice-shared/cs8903onl/hourglass-data/raw_data/tps_files"
RAW_IMAGES_DIR = "/storage/ice-shared/cs8903onl/hourglass-data/raw_data/miami_fall_24_jpgs"

ALT_MODELS_DIR = Path(__file__).parent.resolve()

MODELS = [
    {"name": "stacked_hourglass", "dir": ALT_MODELS_DIR / "stacked-hourglass"},
    {"name": "vit",               "dir": ALT_MODELS_DIR / "vit"},
    {"name": "hrnet",             "dir": ALT_MODELS_DIR / "hrnet"},
    {"name": "hrnet_gcn",         "dir": ALT_MODELS_DIR / "hrnet-gcn"},
    {"name": "hrnet_gcn_hinit",   "dir": ALT_MODELS_DIR / "hrnet-gcn"},
    {"name": "ml_morph",          "dir": ALT_MODELS_DIR / "ml-morph"},
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class PerfStats:
    """Wall-clock time and peak VRAM for a model's inference run."""
    wall_clock_s: Optional[float] = None      # GPU (or default device) wall clock, all test crops
    peak_vram_mb: Optional[float] = None      # peak VRAM allocated during inference (MB)
    cpu_wall_clock_s: Optional[float] = None  # CPU-only wall clock, all test crops

    @property
    def ms_per_sample(self) -> Optional[float]:
        return None  # filled in after we know sample count

    def ms_per_sample_for(self, n_samples: int) -> Optional[float]:
        if self.wall_clock_s is None or n_samples == 0:
            return None
        return (self.wall_clock_s / n_samples) * 1000.0

    def cpu_ms_per_sample_for(self, n_samples: int) -> Optional[float]:
        if self.cpu_wall_clock_s is None or n_samples == 0:
            return None
        return (self.cpu_wall_clock_s / n_samples) * 1000.0


def load_test_files(data_dir):
    test_dir = Path(data_dir) / "test"
    if not test_dir.exists():
        logging.error(f"Test directory not found: {test_dir}")
        sys.exit(1)
    files = sorted(test_dir.glob("*.pt"))
    if not files:
        logging.error(f"No .pt files found in {test_dir}")
        sys.exit(1)
    return files


def discover_checkpoint(model_info):
    name = model_info["name"]
    # Check multiple naming conventions
    candidates = [
        model_info["dir"] / "checkpoints" / f"{name}_best.dat",
        model_info["dir"] / "checkpoints" / f"{name}_best.pth",
        model_info["dir"] / "checkpoints" / f"best_{name}.pth",
    ]
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    return None


def back_project(coords_512, M, scale, pad_x, pad_y):
    # Undo letterbox: (coord - pad) / scale → OBB crop space
    coords_raw = coords_512.copy().astype(np.float64)
    coords_raw[:, 0] = (coords_512[:, 0] - pad_x) / scale
    coords_raw[:, 1] = (coords_512[:, 1] - pad_y) / scale

    # Invert the full 3x3 perspective transform (M is from getPerspectiveTransform)
    M_inv = np.linalg.inv(M.astype(np.float64))

    # Homogeneous multiply + perspective divide → original image space
    ones = np.ones((coords_raw.shape[0], 1), dtype=np.float64)
    coords_h = np.hstack([coords_raw, ones])   # (N, 3)
    proj = coords_h @ M_inv.T                  # (N, 3)
    global_pts = proj[:, :2] / proj[:, 2:3]   # perspective divide

    return global_pts


def extract_imgid_from_filename(filename):
    parts = filename.replace(".pt", "").split("_")
    return parts[0]


def run_stacked_hourglass(model_dir, ckpt_path, test_files):
    sys.path.insert(0, str(model_dir))
    try:
        from model import StackedHourGlass
    finally:
        sys.path.pop(0)

    model = StackedHourGlass()
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img = data["image"].permute(1, 2, 0).float() / 255.0  # uint8 CHW → HWC float
            img_batch = img.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(img_batch)

            last_stack = out[0, -1].cpu()
            coords_512 = []
            for c in range(last_stack.shape[0]):
                hm = last_stack[c].numpy()
                idx = np.unravel_index(np.argmax(hm), hm.shape)
                coords_512.append((idx[1] * 4, idx[0] * 4))
            predictions.append(np.array(coords_512, dtype=np.float64))
        except Exception as e:
            logging.error(f"[stacked_hourglass] Error on {f.name}: {e}")
            predictions.append(None)
    return predictions


def run_vit(model_dir, ckpt_path, test_files):
    sys.path.insert(0, str(model_dir))
    try:
        from model import ViTLandmark
    finally:
        sys.path.pop(0)

    model = ViTLandmark(pretrained=False)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    vit_transform = A.Compose([
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(224, 224, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img = data["image"].permute(1, 2, 0).numpy()  # uint8 HWC
            aug = vit_transform(image=img)
            img_tensor = aug["image"].unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                out = model(img_tensor)

            coords_224 = out[0].cpu().reshape(9, 2).numpy() * 224
            coords_512 = coords_224 * (512.0 / 224.0)
            predictions.append(coords_512.astype(np.float64))
        except Exception as e:
            logging.error(f"[vit] Error on {f.name}: {e}")
            predictions.append(None)
    return predictions


def run_hrnet(model_dir, ckpt_path, test_files):
    sys.path.insert(0, str(model_dir))
    try:
        from model import HRNetLandmarkModel
    finally:
        sys.path.pop(0)

    model = HRNetLandmarkModel(pretrained=False)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img_np = data["image"].permute(1, 2, 0).numpy()  # uint8 HWC
            img_norm = (img_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                out = model(img_tensor)

            coords_512 = out[0].cpu().numpy() * 512
            predictions.append(coords_512.astype(np.float64))
        except Exception as e:
            logging.error(f"[hrnet] Error on {f.name}: {e}")
            predictions.append(None)
    return predictions


def run_hrnet_gcn(model_dir, ckpt_path, test_files):
    sys.path.insert(0, str(model_dir))
    try:
        from hrnet_gcn import HRNetGNN
        from utils import make_chain_edge_index
        from config import HRNetGCNTrainingConfig
    finally:
        sys.path.pop(0)

    config = HRNetGCNTrainingConfig(str(model_dir / "default-config.json"))

    model = HRNetGNN(hrnet_backbone="hrnet_w18", feat_dim=config.feat_dim, gnn_hidden=config.gnn_hidden,
                     num_layers=config.num_layers, num_landmarks=config.num_landmarks, num_iters=config.num_iters)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    mean_shape = torch.tensor([
        [0.3, 0.9], [0.4, 0.8], [0.5, 0.7],
        [0.6, 0.6], [0.7, 0.5], [0.8, 0.4],
        [0.7, 0.3], [0.6, 0.2], [0.5, 0.1],
    ], dtype=torch.float).to(DEVICE)
    edge_index = make_chain_edge_index(num_landmarks=config.num_landmarks).to(DEVICE)

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img_np = data["image"].permute(1, 2, 0).numpy()  # uint8 HWC
            img_norm = (img_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            initial_coords = mean_shape.unsqueeze(0)

            with torch.no_grad():
                out = model(img_tensor, initial_coords, edge_index)

            coords_512 = out[0].cpu().numpy() * 512
            predictions.append(coords_512.astype(np.float64))
        except Exception as e:
            logging.error(f"[hrnet_gcn] Error on {f.name}: {e}")
            predictions.append(None)
    return predictions


def run_hrnet_gcn_hinit(model_dir, ckpt_path, test_files):
    sys.path.insert(0, str(model_dir))
    try:
        from hrnet_gcn_hinit import HRNetGNNWithInit
        from utils import make_chain_edge_index
        from config import HRNetGCNTrainingConfig
    finally:
        sys.path.pop(0)

    config = HRNetGCNTrainingConfig(str(model_dir / "default-config.json"))

    model = HRNetGNNWithInit(
        hrnet_backbone="hrnet_w18",
        feat_dim=config.feat_dim,
        gnn_hidden=config.gnn_hidden,
        num_layers=config.num_layers,
        num_landmarks=config.num_landmarks,
        num_iters=config.num_iters,
    )
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    edge_index = make_chain_edge_index(num_landmarks=config.num_landmarks).to(DEVICE)

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img_np = data["image"].permute(1, 2, 0).numpy()  # uint8 HWC
            img_norm = (img_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                out = model(img_tensor, edge_index)

            # out is (initial_coords, final_coords) — use final_coords
            coords_512 = out[1][0].cpu().numpy() * 512
            predictions.append(coords_512.astype(np.float64))
        except Exception as e:
            logging.error(f"[hrnet_gcn_hinit] Error on {f.name}: {e}")
            predictions.append(None)
    return predictions


def run_ml_morph(model_dir, ckpt_path, test_files):
    if not _DLIB_AVAILABLE:
        logging.warning("[ml_morph] dlib not installed, skipping")
        return [None] * len(test_files)

    predictor = dlib.shape_predictor(str(ckpt_path))
    rect = dlib.rectangle(0, 0, 512, 512)

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img = data["image"].permute(1, 2, 0).numpy()

            shape = predictor(img, rect)
            if shape.num_parts != 9:
                logging.warning(f"[ml_morph] {f.name}: expected 9 parts, got {shape.num_parts}")
                predictions.append(None)
                continue

            coords = np.array(
                [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)],
                dtype=np.float64
            )
            predictions.append(coords)
        except Exception as e:
            logging.error(f"[ml_morph] Error on {f.name}: {e}")
            predictions.append(None)
    return predictions


RUNNERS = {
    "stacked_hourglass": run_stacked_hourglass,
    "vit": run_vit,
    "hrnet": run_hrnet,
    "hrnet_gcn": run_hrnet_gcn,
    "hrnet_gcn_hinit": run_hrnet_gcn_hinit,
    "ml_morph": run_ml_morph,
}


def compute_metrics(predictions, test_files, tps_data_dir, raw_images_dir):
    errors = []
    per_landmark_errors = [[] for _ in range(9)]

    # Diagnostic accumulators — first 8 valid crops with full per-landmark detail
    _DIAG_MAX = 8
    diag_samples = []

    n_total = 0
    n_skip_pred_none = 0
    n_skip_backproject = 0
    n_skip_img_missing = 0
    n_skip_tps_missing = 0
    n_skip_gt_count = 0
    n_evaluated = 0

    for pred, f in zip(predictions, test_files):
        n_total += 1
        if pred is None:
            n_skip_pred_none += 1
            continue

        data = torch.load(f, map_location="cpu")
        M = data["M"].numpy()
        scale = data["scale"].item()
        pad_x, pad_y = data["pad"].tolist()
        orig_h, orig_w = data["orig_size"].tolist()  # OBB crop dims before resize_and_pad

        try:
            pred_global = back_project(pred, M, scale, pad_x, pad_y)
        except Exception as e:
            logging.warning(f"Back-projection failed for {f.name}: {e}, skipping")
            n_skip_backproject += 1
            continue

        imgid = extract_imgid_from_filename(f.name)
        img_path = os.path.join(raw_images_dir, f"{imgid}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Could not read raw image: {img_path}")
            n_skip_img_missing += 1
            continue

        full_h, full_w = img.shape[:2]

        try:
            gt_coords = get_tps_coords(imgid, img, tps_data_dir)
        except Exception as e:
            logging.warning(f"Could not load TPS for {imgid}: {e}")
            n_skip_tps_missing += 1
            continue

        class_name = data.get("class_name", None)
        if class_name is None:
            gt_finger = gt_coords.get("finger", [])
            gt_toe = gt_coords.get("toe", [])
            gt_all = gt_finger + gt_toe
        else:
            gt_all = gt_coords.get(class_name, [])

        if len(gt_all) != 9:
            logging.warning(f"GT landmark count {len(gt_all)} != 9 for {f.name} (class={class_name})")
            n_skip_gt_count += 1
            continue

        gt_arr = np.array(gt_all, dtype=np.float64)
        dists = np.linalg.norm(pred_global - gt_arr, axis=1)
        errors.append(np.mean(dists))
        n_evaluated += 1

        for lm in range(9):
            per_landmark_errors[lm].append(dists[lm])

        # Collect full per-landmark diagnostic for first _DIAG_MAX valid crops
        if len(diag_samples) < _DIAG_MAX:
            lm_detail = []
            for lm in range(9):
                lm_detail.append({
                    "lm": lm,
                    "pred_512": (round(float(pred[lm, 0]), 2), round(float(pred[lm, 1]), 2)),
                    "pred_global": (round(float(pred_global[lm, 0]), 2), round(float(pred_global[lm, 1]), 2)),
                    "gt_global": (round(float(gt_arr[lm, 0]), 2), round(float(gt_arr[lm, 1]), 2)),
                    "dist_px": round(float(dists[lm]), 2),
                })
            diag_samples.append({
                "file": f.name,
                "imgid": imgid,
                "class": class_name or "N/A",
                "orig_img_wh": (int(full_w), int(full_h)),
                "obb_crop_wh": (int(orig_w), int(orig_h)),
                "resize_scale": round(float(scale), 4),
                "pad_xy": (int(pad_x), int(pad_y)),
                "pred_512_xrange": (round(float(pred[:, 0].min()), 1), round(float(pred[:, 0].max()), 1)),
                "pred_512_yrange": (round(float(pred[:, 1].min()), 1), round(float(pred[:, 1].max()), 1)),
                "pred_global_xrange": (round(float(pred_global[:, 0].min()), 1), round(float(pred_global[:, 0].max()), 1)),
                "pred_global_yrange": (round(float(pred_global[:, 1].min()), 1), round(float(pred_global[:, 1].max()), 1)),
                "gt_xrange": (round(float(gt_arr[:, 0].min()), 1), round(float(gt_arr[:, 0].max()), 1)),
                "gt_yrange": (round(float(gt_arr[:, 1].min()), 1), round(float(gt_arr[:, 1].max()), 1)),
                "mean_error_px": round(float(np.mean(dists)), 2),
                "landmarks": lm_detail,
            })

    coverage = {
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "n_skip_pred_none": n_skip_pred_none,
        "n_skip_backproject": n_skip_backproject,
        "n_skip_img_missing": n_skip_img_missing,
        "n_skip_tps_missing": n_skip_tps_missing,
        "n_skip_gt_count": n_skip_gt_count,
    }

    if not errors:
        return {"mean": None, "median": None, "std": None, "p25": None, "p75": None, "p90": None,
                "per_landmark": [None] * 9, "diag_samples": diag_samples, "coverage": coverage}

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "p25": float(np.percentile(errors, 25)),
        "p75": float(np.percentile(errors, 75)),
        "p90": float(np.percentile(errors, 90)),
        "per_landmark": [float(np.mean(e)) if e else None for e in per_landmark_errors],
        "diag_samples": diag_samples,
        "coverage": coverage,
    }


def draw_overlay(image_bgr, coords_px):
    out = image_bgr.copy()
    for i, (x, y) in enumerate(coords_px):
        px, py = int(round(float(x))), int(round(float(y)))
        cv2.circle(out, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(out, str(i), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def generate_overlays(predictions, test_files, overlay_dir, model_name, max_overlays=10):
    saved = []
    count = 0
    for pred, f in zip(predictions, test_files):
        if pred is None:
            continue
        if count >= max_overlays:
            break

        data = torch.load(f, map_location="cpu")
        img = data["image"].permute(1, 2, 0).numpy()  # CHW uint8 → HWC
        img_bgr = img.astype(np.uint8)

        overlay = draw_overlay(img_bgr, pred.astype(np.float32))

        out_path = overlay_dir / f"{model_name}_{count}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(str(out_path))
        count += 1

    return saved


def generate_unannotated_overlays(predictions, unannotated_files, overlay_dir, model_name, max_overlays=10):
    """Generate overlays for unannotated (vertically flipped RHS) crops.
    
    Predictions are flipped back vertically after inference so the
    visualization shows the original (unflipped) orientation.
    The model infers on the vertically-flipped crop (RHS appears as LHS).
    After inference, we undo the vertical flip on both image and Y coordinates.
    """
    saved = []
    count = 0
    for pred, f in zip(predictions, unannotated_files):
        if pred is None:
            continue
        if count >= max_overlays:
            break

        data = torch.load(f, map_location="cpu")
        img = data["image"].permute(1, 2, 0).numpy().astype(np.uint8)  # uint8 HWC

        img_unflipped = cv2.flip(img, 0)  # undo vertical flip

        pred_unflipped = pred.copy()
        pred_unflipped[:, 1] = 512.0 - pred[:, 1]  # flip Y back (vertical)

        overlay = draw_overlay(img_unflipped, pred_unflipped.astype(np.float32))

        out_path = overlay_dir / f"{model_name}_unannotated_{count}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(str(out_path))
        count += 1

    return saved


def _fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_html_report(all_metrics, all_overlays, all_unannotated_overlays, output_dir,
                      all_perf=None, n_test=0):
    sections = []

    sections.append("<h2>Per-Model Pixel Error (Original Image Space)</h2>")
    sections.append("<table border='1' cellpadding='6' cellspacing='0'>")
    sections.append("<tr><th>Model</th><th>Mean</th><th>Median</th><th>Std</th><th>P25</th><th>P75</th><th>P90</th></tr>")
    for name, metrics in all_metrics.items():
        def _f(k): return f"{metrics[k]:.2f}" if metrics.get(k) is not None else "N/A"
        sections.append(
            f"<tr><td>{name}</td><td>{_f('mean')}</td><td>{_f('median')}</td>"
            f"<td>{_f('std')}</td><td>{_f('p25')}</td><td>{_f('p75')}</td><td>{_f('p90')}</td></tr>"
        )
    sections.append("</table>")

    # --- Wall-clock timing table ---
    if all_perf:
        sections.append("<h2>Inference Wall-Clock Time</h2>")
        sections.append("<table border='1' cellpadding='6' cellspacing='0'>")
        sections.append(
            "<tr><th>Model</th><th>GPU total (s)</th>"
            f"<th>GPU ms/sample (n={n_test})</th>"
            "<th>CPU total (s)</th>"
            f"<th>CPU ms/sample (n={n_test})</th></tr>"
        )
        for name, perf in all_perf.items():
            gpu_total = f"{perf.wall_clock_s:.2f}" if perf.wall_clock_s is not None else "N/A"
            gpu_per = perf.ms_per_sample_for(n_test)
            gpu_per_str = f"{gpu_per:.1f}" if gpu_per is not None else "N/A"
            cpu_total = f"{perf.cpu_wall_clock_s:.2f}" if perf.cpu_wall_clock_s is not None else "N/A"
            cpu_per = perf.cpu_ms_per_sample_for(n_test)
            cpu_per_str = f"{cpu_per:.1f}" if cpu_per is not None else "N/A"
            sections.append(
                f"<tr><td>{name}</td><td>{gpu_total}</td><td>{gpu_per_str}</td>"
                f"<td>{cpu_total}</td><td>{cpu_per_str}</td></tr>"
            )
        sections.append("</table>")

    # --- Peak VRAM table ---
    if all_perf:
        sections.append("<h2>Peak VRAM Usage</h2>")
        sections.append("<table border='1' cellpadding='6' cellspacing='0'>")
        sections.append("<tr><th>Model</th><th>Peak VRAM (MB)</th></tr>")
        for name, perf in all_perf.items():
            vram = f"{perf.peak_vram_mb:.1f}" if perf.peak_vram_mb is not None else "N/A (CPU)"
            sections.append(f"<tr><td>{name}</td><td>{vram}</td></tr>")
        sections.append("</table>")

    # --- Sample coverage table ---
    sections.append("<h2>Evaluation Coverage (how many test crops were actually scored)</h2>")
    sections.append("<p style='font-size:0.85em;color:#c00'><b>If n_evaluated is much less than n_total, the reported mean/median is unreliable.</b></p>")
    sections.append("<table border='1' cellpadding='6' cellspacing='0'>")
    sections.append(
        "<tr><th>Model</th><th>n_total</th><th>n_evaluated</th>"
        "<th>skip: pred=None</th><th>skip: backproject</th>"
        "<th>skip: img missing</th><th>skip: TPS missing</th><th>skip: GT count≠9</th></tr>"
    )
    for name, metrics in all_metrics.items():
        cov = metrics.get("coverage", {})
        def _c(k): return str(cov.get(k, "?"))
        n_eval = cov.get("n_evaluated", 0)
        n_tot = cov.get("n_total", 0)
        # highlight row red if fewer than 50% evaluated
        style = " style='background:#ffe0e0'" if n_tot > 0 and n_eval < n_tot * 0.5 else ""
        sections.append(
            f"<tr{style}><td>{name}</td><td>{_c('n_total')}</td><td><b>{_c('n_evaluated')}</b></td>"
            f"<td>{_c('n_skip_pred_none')}</td><td>{_c('n_skip_backproject')}</td>"
            f"<td>{_c('n_skip_img_missing')}</td><td>{_c('n_skip_tps_missing')}</td>"
            f"<td>{_c('n_skip_gt_count')}</td></tr>"
        )
    sections.append("</table>")

    # --- Coordinate-space diagnostics ---
    sections.append("<h2>Coordinate-Space Diagnostics (first 5 valid samples per model)</h2>")
    sections.append("<p style='font-size:0.85em;color:#555'>")
    sections.append("Checks that back-projected predictions and GT both land in original image space.<br/>")
    sections.append("<b>pred_global</b> x/y range should be within orig_img_w/h. "
                    "If ranges match obb_crop_w/h instead, back-projection is not reaching original image space.")
    sections.append("</p>")

    for name, metrics in all_metrics.items():
        samples = metrics.get("diag_samples", [])
        sections.append(f"<h3>{name}</h3>")
        if not samples:
            sections.append("<p><em>No diagnostic samples (no checkpoint or no valid crops).</em></p>")
            continue
        sections.append("<table border='1' cellpadding='4' cellspacing='0' style='font-size:0.8em'>")
        sections.append(
            "<tr>"
            "<th>File</th>"
            "<th>Class</th>"
            "<th>Orig img W×H</th>"
            "<th>OBB crop W×H</th>"
            "<th>Resize scale</th>"
            "<th>Pad X,Y</th>"
            "<th>Pred 512 X range</th>"
            "<th>Pred 512 Y range</th>"
            "<th>Pred global X range</th>"
            "<th>Pred global Y range</th>"
            "<th>GT X range</th>"
            "<th>GT Y range</th>"
            "<th>Mean err (px)</th>"
            "</tr>"
        )
        for s in samples:
            ow, oh = s["orig_img_wh"]
            cw, ch = s["obb_crop_wh"]
            px0, px1 = s["pred_global_xrange"]
            py0, py1 = s["pred_global_yrange"]
            in_bounds = (px0 >= -0.2 * ow) and (px1 <= 1.2 * ow) and (py0 >= -0.2 * oh) and (py1 <= 1.2 * oh)
            row_style = "" if in_bounds else " style='background:#ffe0e0'"
            sections.append(
                f"<tr{row_style}>"
                f"<td>{s['file']}</td>"
                f"<td>{s['class']}</td>"
                f"<td>{ow}×{oh}</td>"
                f"<td>{cw}×{ch}</td>"
                f"<td>{s['resize_scale']}</td>"
                f"<td>{s['pad_xy'][0]},{s['pad_xy'][1]}</td>"
                f"<td>{s['pred_512_xrange'][0]}–{s['pred_512_xrange'][1]}</td>"
                f"<td>{s['pred_512_yrange'][0]}–{s['pred_512_yrange'][1]}</td>"
                f"<td><b>{px0}–{px1}</b></td>"
                f"<td><b>{py0}–{py1}</b></td>"
                f"<td>{s['gt_xrange'][0]}–{s['gt_xrange'][1]}</td>"
                f"<td>{s['gt_yrange'][0]}–{s['gt_yrange'][1]}</td>"
                f"<td>{s['mean_error_px']}</td>"
                "</tr>"
            )
            # Per-landmark detail subtable
            if s.get("landmarks"):
                sections.append(
                    "<tr><td colspan='13' style='padding:4px 16px'>"
                    "<table border='1' cellpadding='3' cellspacing='0' style='font-size:0.75em;background:#f9f9f9'>"
                    "<tr><th>LM</th>"
                    "<th>Pred 512 (x,y)</th>"
                    "<th>Pred global (x,y)</th>"
                    "<th>GT global (x,y)</th>"
                    "<th>Error (px)</th></tr>"
                )
                for lm in s["landmarks"]:
                    p512 = f"{lm['pred_512'][0]}, {lm['pred_512'][1]}"
                    pg   = f"{lm['pred_global'][0]}, {lm['pred_global'][1]}"
                    gt   = f"{lm['gt_global'][0]}, {lm['gt_global'][1]}"
                    err  = lm['dist_px']
                    err_style = " style='background:#ffdddd'" if err > 50 else ""
                    sections.append(
                        f"<tr{err_style}>"
                        f"<td>{lm['lm']}</td>"
                        f"<td>{p512}</td>"
                        f"<td>{pg}</td>"
                        f"<td>{gt}</td>"
                        f"<td><b>{err}</b></td>"
                        "</tr>"
                    )
                sections.append("</table></td></tr>")
        sections.append("</table>")

    sections.append("<h2>Per-Landmark Mean Pixel Error</h2>")
    sections.append("<table border='1' cellpadding='6' cellspacing='0'>")
    header = "<tr><th>Model</th>" + "".join(f"<th>LM {i}</th>" for i in range(9)) + "</tr>"
    sections.append(header)
    for name, metrics in all_metrics.items():
        row = f"<tr><td>{name}</td>"
        for v in metrics["per_landmark"]:
            row += f"<td>{v:.2f}</td>" if v is not None else "<td>N/A</td>"
        row += "</tr>"
        sections.append(row)
    sections.append("</table>")

    sections.append("<h2>Landmark Overlays (Test Set — Annotated)</h2>")
    for name, paths in all_overlays.items():
        sections.append(f"<h3>{name}</h3>")
        if not paths:
            sections.append("<p><em>No overlays (checkpoint missing or inference failed)</em></p>")
        else:
            for p in paths:
                b64 = _img_to_base64(p)
                sections.append(f'<img src="data:image/png;base64,{b64}" style="max-width:300px;margin:4px"/>')

    sections.append("<h2>Landmark Overlays (Unannotated — RHS Flipped Back)</h2>")
    for name, paths in all_unannotated_overlays.items():
        sections.append(f"<h3>{name}</h3>")
        if not paths:
            sections.append("<p><em>No unannotated overlays available</em></p>")
        else:
            for p in paths:
                b64 = _img_to_base64(p)
                sections.append(f'<img src="data:image/png;base64,{b64}" style="max-width:300px;margin:4px"/>')

    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Unified Pipeline Evaluation Report</title>
<style>
body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
table {{ border-collapse: collapse; }}
</style>
</head>
<body>
<h1>Unified Pipeline Evaluation Report</h1>
{body}
</body>
</html>
"""
    return html


def build_markdown_summary(all_metrics, all_perf=None, n_test=0):
    lines = [
        "# Evaluation Summary\n",
        "## Pixel Error (Original Image Space)\n",
        "| Model | Mean | Median | Std | P25 | P75 | P90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, metrics in all_metrics.items():
        def _f(k): return f"{metrics[k]:.2f}" if metrics.get(k) is not None else "N/A"
        lines.append(f"| {name} | {_f('mean')} | {_f('median')} | {_f('std')} | {_f('p25')} | {_f('p75')} | {_f('p90')} |")

    if all_perf:
        lines += [
            "",
            f"## Inference Wall-Clock Time (n={n_test} test crops)\n",
            "| Model | GPU total (s) | GPU ms/sample | CPU total (s) | CPU ms/sample |",
            "|---|---|---|---|---|",
        ]
        for name, perf in all_perf.items():
            gpu_total = f"{perf.wall_clock_s:.2f}" if perf.wall_clock_s is not None else "N/A"
            gpu_per = perf.ms_per_sample_for(n_test)
            gpu_per_str = f"{gpu_per:.1f}" if gpu_per is not None else "N/A"
            cpu_total = f"{perf.cpu_wall_clock_s:.2f}" if perf.cpu_wall_clock_s is not None else "N/A"
            cpu_per = perf.cpu_ms_per_sample_for(n_test)
            cpu_per_str = f"{cpu_per:.1f}" if cpu_per is not None else "N/A"
            lines.append(f"| {name} | {gpu_total} | {gpu_per_str} | {cpu_total} | {cpu_per_str} |")

        lines += [
            "",
            "## Peak VRAM Usage\n",
            "| Model | Peak VRAM (MB) |",
            "|---|---|",
        ]
        for name, perf in all_perf.items():
            vram = f"{perf.peak_vram_mb:.1f}" if perf.peak_vram_mb is not None else "N/A (CPU)"
            lines.append(f"| {name} | {vram} |")

    return "\n".join(lines) + "\n"


def load_unannotated_files(data_dir):
    unannotated_dir = Path(data_dir) / "unannotated"
    if not unannotated_dir.exists():
        logging.warning(f"Unannotated directory not found: {unannotated_dir}")
        return []
    files = sorted(unannotated_dir.glob("*.pt"))
    return files


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on held-out test set")
    parser.add_argument("--data-dir", type=str, default=SHARED_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=str(ALT_MODELS_DIR / "benchmarking" / "report"))
    parser.add_argument("--tps-data-dir", type=str, default=TPS_FILES_DIR)
    parser.add_argument("--raw-images-dir", type=str, default=RAW_IMAGES_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    test_files = load_test_files(args.data_dir)
    logging.info(f"Loaded {len(test_files)} test crops")

    unannotated_files = load_unannotated_files(args.data_dir)
    logging.info(f"Loaded {len(unannotated_files)} unannotated crops")

    all_metrics = {}
    all_overlays = {}
    all_unannotated_overlays = {}
    all_perf = {}

    for model_info in MODELS:
        name = model_info["name"]
        ckpt = discover_checkpoint(model_info)
        if ckpt is None:
            logging.warning(f"[{name}] No checkpoint found, skipping")
            all_metrics[name] = {"mean": None, "median": None, "std": None,
                                 "p25": None, "p75": None, "p90": None,
                                 "per_landmark": [None] * 9,
                                 "diag_samples": [],
                                 "coverage": {"n_total": 0, "n_evaluated": 0, "n_skip_pred_none": 0,
                                              "n_skip_backproject": 0, "n_skip_img_missing": 0,
                                              "n_skip_tps_missing": 0, "n_skip_gt_count": 0}}
            all_overlays[name] = []
            all_unannotated_overlays[name] = []
            all_perf[name] = PerfStats()
            continue

        logging.info(f"[{name}] Running inference on {len(test_files)} test crops...")
        runner = RUNNERS[name]

        # --- timed + VRAM-tracked inference ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_before = torch.cuda.max_memory_allocated()
        t0 = time.perf_counter()
        predictions = runner(model_info["dir"], ckpt, test_files)
        wall_s = time.perf_counter() - t0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            # If nothing was allocated above baseline, report None (CPU-only runner)
            peak_vram_mb = peak_vram_mb if peak_vram_mb > 1.0 else None
        else:
            peak_vram_mb = None
        all_perf[name] = PerfStats(wall_clock_s=wall_s, peak_vram_mb=peak_vram_mb)
        # ----------------------------------------

        logging.info(f"[{name}] Computing metrics...")
        metrics = compute_metrics(predictions, test_files, args.tps_data_dir, args.raw_images_dir)
        all_metrics[name] = metrics

        if metrics["mean"] is not None:
            cov = metrics.get("coverage", {})
            logging.info(
                f"[{name}] Mean pixel error: {metrics['mean']:.2f}, Median: {metrics['median']:.2f} "
                f"— evaluated {cov.get('n_evaluated','?')}/{cov.get('n_total','?')} crops "
                f"(skip: img_missing={cov.get('n_skip_img_missing',0)}, "
                f"tps_missing={cov.get('n_skip_tps_missing',0)}, "
                f"gt_count={cov.get('n_skip_gt_count',0)})"
            )

        logging.info(f"[{name}] Generating test overlays...")
        overlays = generate_overlays(predictions, test_files, overlay_dir, name)
        all_overlays[name] = overlays

        if unannotated_files:
            logging.info(f"[{name}] Running inference on {len(unannotated_files)} unannotated crops...")
            unannotated_preds = runner(model_info["dir"], ckpt, unannotated_files)
            logging.info(f"[{name}] Generating unannotated overlays...")
            u_overlays = generate_unannotated_overlays(unannotated_preds, unannotated_files, overlay_dir, name)
            all_unannotated_overlays[name] = u_overlays
        else:
            all_unannotated_overlays[name] = []

    logging.info("Running CPU-only timing pass...")
    global DEVICE
    _original_device = DEVICE
    DEVICE = 'cpu'
    for model_info in MODELS:
        name = model_info["name"]
        if name not in all_perf:
            continue
        ckpt = discover_checkpoint(model_info)
        if ckpt is None:
            continue
        runner = RUNNERS[name]
        try:
            t0 = time.perf_counter()
            runner(model_info["dir"], ckpt, test_files)
            all_perf[name].cpu_wall_clock_s = time.perf_counter() - t0
            logging.info(f"[{name}] CPU wall clock: {all_perf[name].cpu_wall_clock_s:.2f}s")
        except Exception as e:
            logging.warning(f"[{name}] CPU timing failed: {e}")
    DEVICE = _original_device

    logging.info("Writing report...")
    html = build_html_report(all_metrics, all_overlays, all_unannotated_overlays, output_dir,
                             all_perf=all_perf, n_test=len(test_files))
    md = build_markdown_summary(all_metrics, all_perf=all_perf, n_test=len(test_files))

    html_path = output_dir / "benchmark_report.html"
    md_path = output_dir / "benchmark_summary.md"

    with open(html_path, "w") as f:
        f.write(html)
    with open(md_path, "w") as f:
        f.write(md)

    logging.info(f"Report: {html_path}")
    logging.info(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
