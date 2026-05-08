import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import logging
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.tps_utils import get_tps_coords

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
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


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
    ckpt = model_info["dir"] / "checkpoints" / f"{name}_best.pth"
    if ckpt.exists():
        return ckpt
    return None


def back_project(coords_512, M, orig_size):
    h, w = float(orig_size[0]), float(orig_size[1])
    scale = min(512.0 / h, 512.0 / w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_x = (512 - new_w) // 2
    pad_y = (512 - new_h) // 2

    coords_raw = coords_512.copy()
    coords_raw[:, 0] = (coords_512[:, 0] - pad_x) / scale
    coords_raw[:, 1] = (coords_512[:, 1] - pad_y) / scale

    M_inv = np.linalg.inv(M)
    pts = coords_raw.reshape(1, -1, 2).astype(np.float64)
    global_pts = cv2.perspectiveTransform(pts, M_inv)
    return global_pts.reshape(-1, 2)


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
    model.eval()

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img = data["image"].permute(1, 2, 0).float() / 255.0
            img_batch = img.unsqueeze(0)

            with torch.no_grad():
                out = model(img_batch)

            last_stack = out[0, -1]
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
            img = data["image"].permute(1, 2, 0).numpy()
            aug = vit_transform(image=img)
            img_tensor = aug["image"].unsqueeze(0).float()

            with torch.no_grad():
                out = model(img_tensor)

            coords_224 = out[0].reshape(9, 2).numpy() * 224
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
    model.eval()

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img = data["image"].float()
            img_np = img.permute(1, 2, 0).numpy()
            img_norm = (img_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

            with torch.no_grad():
                out = model(img_tensor)

            coords_512 = out[0].numpy() * 512
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
    finally:
        sys.path.pop(0)

    model = HRNetGNN(hrnet_backbone="hrnet_w18", feat_dim=1024, gnn_hidden=128,
                     num_layers=3, num_landmarks=9, num_iters=6)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()

    mean_shape = torch.tensor([
        [0.3, 0.9], [0.4, 0.8], [0.5, 0.7],
        [0.6, 0.6], [0.7, 0.5], [0.8, 0.4],
        [0.7, 0.3], [0.6, 0.2], [0.5, 0.1],
    ], dtype=torch.float)
    edge_index = make_chain_edge_index(num_landmarks=9)

    predictions = []
    for f in test_files:
        try:
            data = torch.load(f, map_location="cpu")
            img = data["image"].float()
            img_np = img.permute(1, 2, 0).numpy()
            img_norm = (img_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
            initial_coords = mean_shape.unsqueeze(0)

            with torch.no_grad():
                out = model(img_tensor, initial_coords, edge_index)

            coords_512 = out[0].numpy() * 512
            predictions.append(coords_512.astype(np.float64))
        except Exception as e:
            logging.error(f"[hrnet_gcn] Error on {f.name}: {e}")
            predictions.append(None)
    return predictions


RUNNERS = {
    "stacked_hourglass": run_stacked_hourglass,
    "vit": run_vit,
    "hrnet": run_hrnet,
    "hrnet_gcn": run_hrnet_gcn,
}


def compute_metrics(predictions, test_files, tps_data_dir, raw_images_dir):
    errors = []
    per_landmark_errors = [[] for _ in range(9)]

    for pred, f in zip(predictions, test_files):
        if pred is None:
            continue

        data = torch.load(f, map_location="cpu")
        M = data["M"].numpy()
        orig_size = data["orig_size"].numpy()

        try:
            pred_global = back_project(pred, M, orig_size)
        except np.linalg.LinAlgError:
            logging.warning(f"Singular M in {f.name}, skipping")
            continue

        imgid = extract_imgid_from_filename(f.name)
        img_path = os.path.join(raw_images_dir, f"{imgid}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_coords = get_tps_coords(imgid, img, tps_data_dir)
        class_name = data.get("class_name", None)
        if class_name is None:
            gt_finger = gt_coords.get("finger", [])
            gt_toe = gt_coords.get("toe", [])
            gt_all = gt_finger + gt_toe
        else:
            gt_all = gt_coords.get(class_name, [])

        if len(gt_all) != 9:
            continue

        gt_arr = np.array(gt_all, dtype=np.float64)
        dists = np.linalg.norm(pred_global - gt_arr, axis=1)
        errors.append(np.mean(dists))

        for lm in range(9):
            per_landmark_errors[lm].append(dists[lm])

    if not errors:
        return {"mean": None, "median": None, "per_landmark": [None] * 9}

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "per_landmark": [float(np.mean(e)) if e else None for e in per_landmark_errors],
    }


def draw_overlay(image_bgr, coords_px):
    out = image_bgr.copy()
    for i, (x, y) in enumerate(coords_px):
        px, py = int(round(float(x))), int(round(float(y)))
        cv2.circle(out, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(out, str(i), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def generate_overlays(predictions, test_files, overlay_dir, model_name, max_overlays=5):
    saved = []
    count = 0
    for pred, f in zip(predictions, test_files):
        if pred is None:
            continue
        if count >= max_overlays:
            break

        data = torch.load(f, map_location="cpu")
        img = data["image"].permute(1, 2, 0).numpy()
        img_bgr = img.astype(np.uint8)

        overlay = draw_overlay(img_bgr, pred.astype(np.float32))

        out_path = overlay_dir / f"{model_name}_{count}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(str(out_path))
        count += 1

    return saved


def generate_unannotated_overlays(predictions, unannotated_files, overlay_dir, model_name, max_overlays=5):
    """Generate overlays for unannotated (flipped RHS) crops.
    
    Predictions are flipped back horizontally before drawing so the
    visualization shows the original (unflipped) orientation.
    """
    saved = []
    count = 0
    for pred, f in zip(predictions, unannotated_files):
        if pred is None:
            continue
        if count >= max_overlays:
            break

        data = torch.load(f, map_location="cpu")
        img = data["image"].permute(1, 2, 0).numpy().astype(np.uint8)

        img_unflipped = cv2.flip(img, 1)

        pred_unflipped = pred.copy()
        pred_unflipped[:, 0] = 512.0 - pred[:, 0]

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


def build_html_report(all_metrics, all_overlays, all_unannotated_overlays, output_dir):
    sections = []

    sections.append("<h2>Per-Model Pixel Error (Global Image Space)</h2>")
    sections.append("<table border='1' cellpadding='6' cellspacing='0'>")
    sections.append("<tr><th>Model</th><th>Mean Pixel Error</th><th>Median Pixel Error</th></tr>")
    for name, metrics in all_metrics.items():
        mean_str = f"{metrics['mean']:.2f}" if metrics["mean"] is not None else "N/A"
        median_str = f"{metrics['median']:.2f}" if metrics["median"] is not None else "N/A"
        sections.append(f"<tr><td>{name}</td><td>{mean_str}</td><td>{median_str}</td></tr>")
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


def build_markdown_summary(all_metrics):
    lines = [
        "# Evaluation Summary\n",
        "| Model | Mean Pixel Error | Median Pixel Error |",
        "|---|---|---|",
    ]
    for name, metrics in all_metrics.items():
        mean_str = f"{metrics['mean']:.2f}" if metrics["mean"] is not None else "N/A"
        median_str = f"{metrics['median']:.2f}" if metrics["median"] is not None else "N/A"
        lines.append(f"| {name} | {mean_str} | {median_str} |")
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

    for model_info in MODELS:
        name = model_info["name"]
        ckpt = discover_checkpoint(model_info)
        if ckpt is None:
            logging.warning(f"[{name}] No checkpoint found, skipping")
            all_metrics[name] = {"mean": None, "median": None, "per_landmark": [None] * 9}
            all_overlays[name] = []
            all_unannotated_overlays[name] = []
            continue

        logging.info(f"[{name}] Running inference on {len(test_files)} test crops...")
        runner = RUNNERS[name]
        predictions = runner(model_info["dir"], ckpt, test_files)

        logging.info(f"[{name}] Computing metrics...")
        metrics = compute_metrics(predictions, test_files, args.tps_data_dir, args.raw_images_dir)
        all_metrics[name] = metrics

        if metrics["mean"] is not None:
            logging.info(f"[{name}] Mean pixel error: {metrics['mean']:.2f}, Median: {metrics['median']:.2f}")

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

    logging.info("Writing report...")
    html = build_html_report(all_metrics, all_overlays, all_unannotated_overlays, output_dir)
    md = build_markdown_summary(all_metrics)

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
