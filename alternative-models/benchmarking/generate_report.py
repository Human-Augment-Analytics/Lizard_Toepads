import sys
import os
import json
import re
import argparse
import base64
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

BENCHMARKING_DIR = Path(__file__).parent.resolve()
ALT_MODELS_DIR = BENCHMARKING_DIR.parent

REQUIRED_FIELDS = [
    "split_seed", "train_val_ratio",
    "stacked_hourglass_data_dir", "vit_data_dir",
    "hrnet_data_dir", "hrnet_gcn_data_dir",
]

MODELS = [
    {
        "name":        "stacked_hourglass",
        "dir":         ALT_MODELS_DIR / "stacked-hourglass",
        "split_file":  BENCHMARKING_DIR / "splits" / "stacked_hourglass_split.json",
        "log":         ALT_MODELS_DIR / "stacked-hourglass" / "logs" / "stacked_hourglass.log",
        "checkpoint":  ALT_MODELS_DIR / "stacked-hourglass" / "checkpoints" / "stacked_hourglass_best.pth",
        "has_pixel_error": False,
    },
    {
        "name":        "vit",
        "dir":         ALT_MODELS_DIR / "vit",
        "split_file":  BENCHMARKING_DIR / "splits" / "vit_split.json",
        "log":         ALT_MODELS_DIR / "vit" / "logs" / "vit.log",
        "checkpoint":  ALT_MODELS_DIR / "vit" / "checkpoints" / "vit_best.pth",
        "has_pixel_error": True,
    },
    {
        "name":        "hrnet",
        "dir":         ALT_MODELS_DIR / "hrnet",
        "split_file":  BENCHMARKING_DIR / "splits" / "hrnet_split.json",
        "log":         ALT_MODELS_DIR / "hrnet" / "logs" / "hrnet.log",
        "checkpoint":  ALT_MODELS_DIR / "hrnet" / "checkpoints" / "hrnet_best.pth",
        "has_pixel_error": False,
    },
    {
        "name":        "hrnet_gcn",
        "dir":         ALT_MODELS_DIR / "hrnet-gcn",
        "split_file":  BENCHMARKING_DIR / "splits" / "hrnet_gcn_split.json",
        "log":         ALT_MODELS_DIR / "hrnet-gcn" / "logs" / "hrnet_gcn.log",
        "checkpoint":  ALT_MODELS_DIR / "hrnet-gcn" / "checkpoints" / "hrnet_gcn_best.pth",
        "has_pixel_error": True,
    },
]

LOG_PATTERNS = {
    "stacked_hourglass": re.compile(
        r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)"
    ),
    "vit": re.compile(
        r"Epoch (\d+)/\d+ - Val Loss: ([\d.]+), Avg Pixel Error: ([\d.]+)"
    ),
    "hrnet": re.compile(
        r"Epoch (\d+) \| Train ([\d.]+) \| Val ([\d.]+)"
    ),
    "hrnet_gcn": re.compile(
        r"Epoch (\d+)/\d+, Train Loss: ([\d.]+), Avg Pixel Error: ([^\s,]+), Val Loss: ([\d.]+)"
    ),
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        config = json.load(f)
    for field in REQUIRED_FIELDS:
        if field not in config:
            print(f"ERROR: required field '{field}' missing from {path}", file=sys.stderr)
            sys.exit(1)
    return config


def parse_log(log_path: str, model_name: str) -> list:
    p = Path(log_path)
    if not p.exists():
        logging.warning(f"[{model_name}] Log file not found: {log_path}")
        return []
    pattern = LOG_PATTERNS.get(model_name)
    if pattern is None:
        logging.warning(f"[{model_name}] No log pattern defined")
        return []
    records = []
    with open(p) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            try:
                if model_name == "stacked_hourglass":
                    epoch, train_loss, val_loss = int(m.group(1)), float(m.group(2)), float(m.group(3))
                    records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "pixel_error": None})
                elif model_name == "vit":
                    epoch, val_loss, pixel_error = int(m.group(1)), float(m.group(2)), float(m.group(3))
                    records.append({"epoch": epoch, "train_loss": None, "val_loss": val_loss, "pixel_error": pixel_error})
                elif model_name == "hrnet":
                    epoch, train_loss, val_loss = int(m.group(1)), float(m.group(2)), float(m.group(3))
                    records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "pixel_error": None})
                elif model_name == "hrnet_gcn":
                    epoch, train_loss, pix_str, val_loss = int(m.group(1)), float(m.group(2)), m.group(3), float(m.group(4))
                    pixel_error = None if pix_str == "None" else float(pix_str)
                    records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "pixel_error": pixel_error})
            except (ValueError, IndexError):
                continue
    return records


def denormalize_imagenet(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_overlay(image_bgr: np.ndarray, coords_px: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    for i, (x, y) in enumerate(coords_px):
        px, py = int(round(float(x))), int(round(float(y)))
        cv2.circle(out, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(out, str(i), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def _load_split_val_paths(split_file: Path, n: int = 5) -> list:
    if not split_file.exists():
        return []
    with open(split_file) as f:
        data = json.load(f)
    return data.get("val", [])[:n]


def generate_overlays_stacked_hourglass(model_dir: Path, ckpt_path: Path, split_file: Path, overlay_dir: Path) -> list:
    sys.path.insert(0, str(model_dir))
    try:
        from model import StackedHourGlass
        from dataset import LizardDataset
    finally:
        sys.path.pop(0)

    val_paths = _load_split_val_paths(split_file)
    if not val_paths:
        return []

    model = StackedHourGlass()
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()

    saved = []
    for i, path in enumerate(val_paths):
        data = np.load(path)
        img = data["image"]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            out = model(img_tensor)

        last_stack = out[0, -1]
        coords_px = []
        for c in range(last_stack.shape[0]):
            hm = last_stack[c].numpy()
            idx = np.unravel_index(np.argmax(hm), hm.shape)
            coords_px.append((idx[1] * 4, idx[0] * 4))
        coords_px = np.array(coords_px, dtype=np.float32)

        img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = draw_overlay(img_bgr, coords_px)
        out_path = overlay_dir / f"stacked_hourglass_{i}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(str(out_path))

    return saved


def generate_overlays_vit(model_dir: Path, ckpt_path: Path, split_file: Path, overlay_dir: Path) -> list:
    sys.path.insert(0, str(model_dir))
    try:
        from model import ViTLandmark
        from dataset import ViTDataset
    finally:
        sys.path.pop(0)

    val_paths = _load_split_val_paths(split_file)
    if not val_paths:
        return []

    model = ViTLandmark(pretrained=False)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()

    saved = []
    for i, path in enumerate(val_paths):
        data = torch.load(path, map_location="cpu")
        img_tensor = data["image"].float().unsqueeze(0)

        with torch.no_grad():
            out = model(img_tensor)

        coords_px = out[0].reshape(9, 2).numpy() * 224

        img_bgr = denormalize_imagenet(img_tensor[0])
        overlay = draw_overlay(img_bgr, coords_px)
        out_path = overlay_dir / f"vit_{i}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(str(out_path))

    return saved


def generate_overlays_hrnet(model_dir: Path, ckpt_path: Path, split_file: Path, overlay_dir: Path) -> list:
    sys.path.insert(0, str(model_dir))
    try:
        from model import HRNetLandmarkModel
        from dataset import LizardDataset
    finally:
        sys.path.pop(0)

    val_paths = _load_split_val_paths(split_file)
    if not val_paths:
        return []

    model = HRNetLandmarkModel(pretrained=False)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()

    saved = []
    for i, path in enumerate(val_paths):
        data = torch.load(path, map_location="cpu")
        img_tensor = data["image"].float()
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float()
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            out = model(img_tensor)

        coords_px = out[0].numpy() * 512

        img_bgr = denormalize_imagenet(img_tensor[0])
        overlay = draw_overlay(img_bgr, coords_px)
        out_path = overlay_dir / f"hrnet_{i}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(str(out_path))

    return saved


def generate_overlays_hrnet_gcn(model_dir: Path, ckpt_path: Path, split_file: Path, overlay_dir: Path) -> list:
    sys.path.insert(0, str(model_dir))
    try:
        from hrnet_gcn import HRNetGNN
        from lizard_dataset import LizardDataset
        from utils import make_chain_edge_index
    finally:
        sys.path.pop(0)

    val_paths = _load_split_val_paths(split_file)
    if not val_paths:
        return []

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

    saved = []
    for i, path in enumerate(val_paths):
        data = torch.load(path, map_location="cpu")
        img_tensor = data["image"].float()
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float()
        img_tensor = img_tensor.unsqueeze(0)
        initial_coords = mean_shape.unsqueeze(0)

        with torch.no_grad():
            out = model(img_tensor, initial_coords, edge_index)

        coords_px = out[0].numpy() * 512

        img_bgr = denormalize_imagenet(img_tensor[0])
        overlay = draw_overlay(img_bgr, coords_px)
        out_path = overlay_dir / f"hrnet_gcn_{i}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(str(out_path))

    return saved


def generate_all_overlays(overlay_dir: Path) -> dict:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    generators = {
        "stacked_hourglass": generate_overlays_stacked_hourglass,
        "vit":               generate_overlays_vit,
        "hrnet":             generate_overlays_hrnet,
        "hrnet_gcn":         generate_overlays_hrnet_gcn,
    }
    for model in MODELS:
        name = model["name"]
        ckpt = model["checkpoint"]
        split = model["split_file"]
        model_dir = model["dir"]
        if not ckpt.exists():
            logging.warning(f"[{name}] Checkpoint not found: {ckpt}")
            results[name] = []
            continue
        try:
            paths = generators[name](model_dir, ckpt, split, overlay_dir)
            results[name] = paths
            logging.info(f"[{name}] Saved {len(paths)} overlay images")
        except Exception as e:
            logging.error(f"[{name}] Overlay generation failed: {e}")
            results[name] = []
    return results


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_val_loss_plot(metrics_by_model: dict) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, records in metrics_by_model.items():
        if not records:
            continue
        epochs = [r["epoch"] for r in records]
        vals = [r["val_loss"] for r in records]
        ax.plot(epochs, vals, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss — All Models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def build_pixel_error_plot(metrics_by_model: dict) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ["vit", "hrnet_gcn"]:
        records = metrics_by_model.get(name, [])
        if not records:
            continue
        epochs = [r["epoch"] for r in records if r.get("pixel_error") is not None]
        errors = [r["pixel_error"] for r in records if r.get("pixel_error") is not None]
        if epochs:
            ax.plot(epochs, errors, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Pixel Error")
    ax.set_title("Pixel Error — ViT and HRNet-GCN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def build_best_checkpoint_table(metrics_by_model: dict) -> list:
    rows = []
    for model in MODELS:
        name = model["name"]
        records = metrics_by_model.get(name, [])
        if not records:
            rows.append({"model": name, "best_val_loss": "N/A", "best_pixel_error": "N/A", "best_epoch": "N/A"})
            continue
        best = min(records, key=lambda r: r["val_loss"])
        pixel_errors = [r["pixel_error"] for r in records if r.get("pixel_error") is not None]
        best_px = f"{min(pixel_errors):.2f}" if pixel_errors else "N/A"
        rows.append({
            "model": name,
            "best_val_loss": f"{best['val_loss']:.6f}",
            "best_pixel_error": best_px,
            "best_epoch": best["epoch"],
        })
    return rows


def build_markdown_summary(best_table: list) -> str:
    lines = [
        "# Benchmark Summary\n",
        "| Model | Best Val Loss | Best Pixel Error | Epoch of Best Val Loss |",
        "|---|---|---|---|",
    ]
    for row in best_table:
        lines.append(f"| {row['model']} | {row['best_val_loss']} | {row['best_pixel_error']} | {row['best_epoch']} |")
    return "\n".join(lines) + "\n"


def build_html_report(metrics_by_model: dict, overlay_paths: dict, val_loss_b64: str, pixel_error_b64: str, best_table: list) -> str:
    sections = []

    sections.append("<h2>Validation Loss Curves</h2>")
    sections.append(f'<img src="data:image/png;base64,{val_loss_b64}" style="max-width:100%"/>')

    sections.append("<h2>Pixel Error Curves (ViT &amp; HRNet-GCN)</h2>")
    sections.append(f'<img src="data:image/png;base64,{pixel_error_b64}" style="max-width:100%"/>')

    sections.append("<h2>Best Checkpoint Summary</h2>")
    sections.append("<table border='1' cellpadding='6' cellspacing='0'>")
    sections.append("<tr><th>Model</th><th>Best Val Loss</th><th>Best Pixel Error</th><th>Epoch</th></tr>")
    for row in best_table:
        sections.append(f"<tr><td>{row['model']}</td><td>{row['best_val_loss']}</td><td>{row['best_pixel_error']}</td><td>{row['best_epoch']}</td></tr>")
    sections.append("</table>")

    sections.append("<h2>Landmark Overlays</h2>")
    for model in MODELS:
        name = model["name"]
        paths = overlay_paths.get(name, [])
        sections.append(f"<h3>{name}</h3>")
        if not paths:
            sections.append("<p><em>No overlays available (checkpoint missing or generation failed)</em></p>")
        else:
            for p in paths:
                b64 = _img_to_base64(p)
                sections.append(f'<img src="data:image/png;base64,{b64}" style="max-width:300px;margin:4px"/>')

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Model Benchmark Report</title>
<style>
body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
table {{ border-collapse: collapse; }}
</style>
</head>
<body>
<h1>Model Benchmark Report</h1>
{body}
</body>
</html>
"""


def _make_synthetic_metrics() -> dict:
    metrics = {}
    for model in MODELS:
        name = model["name"]
        records = []
        for epoch in range(1, 11):
            base = 0.5 / epoch
            rec = {"epoch": epoch, "train_loss": base * 1.1, "val_loss": base, "pixel_error": None}
            if model["has_pixel_error"]:
                rec["pixel_error"] = 20.0 / epoch
            records.append(rec)
        metrics[name] = records
    return metrics


def _make_synthetic_overlays(overlay_dir: Path) -> dict:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for model in MODELS:
        name = model["name"]
        paths = []
        for i in range(3):
            img = np.full((512, 512, 3), 40, dtype=np.uint8)
            coords = np.array([[100 + i*30 + j*40, 100 + j*40] for j in range(9)], dtype=np.float32)
            overlay = draw_overlay(img, coords)
            out_path = overlay_dir / f"{name}_{i}.png"
            cv2.imwrite(str(out_path), overlay)
            paths.append(str(out_path))
        results[name] = paths
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument("--config", type=str, default=str(BENCHMARKING_DIR / "benchmark_config.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic data — no checkpoints or real logs needed")
    args = parser.parse_args()

    if not args.dry_run:
        load_config(args.config)

    report_dir = BENCHMARKING_DIR / "report"
    overlay_dir = report_dir / "overlays"
    report_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logging.info("DRY RUN — using synthetic data, skipping checkpoints and real logs")
        metrics_by_model = _make_synthetic_metrics()
        overlay_paths = _make_synthetic_overlays(overlay_dir)
    else:
        logging.info("Parsing log files...")
        metrics_by_model = {}
        for model in MODELS:
            name = model["name"]
            records = parse_log(str(model["log"]), name)
            metrics_by_model[name] = records
            logging.info(f"[{name}] Parsed {len(records)} epoch records")

        logging.info("Generating overlay images...")
        overlay_paths = generate_all_overlays(overlay_dir)

    logging.info("Building plots...")
    val_loss_b64 = build_val_loss_plot(metrics_by_model)
    pixel_error_b64 = build_pixel_error_plot(metrics_by_model)
    best_table = build_best_checkpoint_table(metrics_by_model)

    logging.info("Writing report files...")
    html = build_html_report(metrics_by_model, overlay_paths, val_loss_b64, pixel_error_b64, best_table)
    md = build_markdown_summary(best_table)

    html_path = report_dir / "benchmark_report.html"
    md_path = report_dir / "benchmark_summary.md"

    with open(html_path, "w") as f:
        f.write(html)
    with open(md_path, "w") as f:
        f.write(md)

    logging.info(f"Report written to {html_path}")
    logging.info(f"Summary written to {md_path}")


if __name__ == "__main__":
    main()
