#!/usr/bin/env python3
"""
Post-training script: generate plots and export ONNX from a completed run.

Usage:
    uv run python scripts/postprocess/generate_plots_and_export.py --config configs/H9_obb_botonly.yaml
    uv run python scripts/postprocess/generate_plots_and_export.py --config configs/H9_obb_botonly.yaml --skip-export
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots and export ONNX from a completed training run")
    parser.add_argument("--config", required=True, help="Path to project YAML config")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export")
    parser.add_argument("--skip-val", action="store_true", help="Skip validation (only generate results.png)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    task = train_cfg.get("task", "detect")
    name = train_cfg["name"]
    imgsz = train_cfg.get("imgsz", 1280)
    save_dir = Path(f"runs/{task}/{name}")
    best_pt = save_dir / "weights" / "best.pt"
    csv_file = save_dir / "results.csv"

    # Build data yaml
    dataset_cfg = cfg.get("dataset", {})
    data_file = Path("data") / "temp_data.yaml"
    data_file.parent.mkdir(exist_ok=True)
    with open(data_file, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    # 1. Generate results.png from results.csv
    if csv_file.exists():
        print(f"Generating results.png from {csv_file}...")
        plot_results(file=csv_file)
        print(f"  Saved: {save_dir / 'results.png'}")
    else:
        print(f"WARNING: {csv_file} not found, skipping results.png")

    # 2. Validation with plots (confusion matrix, F1, PR, P, R curves)
    if not args.skip_val:
        print(f"\nRunning validation with {best_pt}...")
        model = YOLO(str(best_pt), task=task)
        # Use project/name so val results go into runs/{task}/{name}/ directly
        model.val(data=str(data_file), plots=True, project=f"runs/{task}", name=name, exist_ok=True)
        print("  Generated: confusion_matrix.png, F1_curve.png, PR_curve.png, P_curve.png, R_curve.png")

    # 3. Export ONNX (fp32 + fp16)
    if not args.skip_export:
        from scripts.training.train_yolo import export_onnx
        export_onnx(str(best_pt), task, imgsz, save_dir / "weights")

    print("\nDone!")


if __name__ == "__main__":
    main()
