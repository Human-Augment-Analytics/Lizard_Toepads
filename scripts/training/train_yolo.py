#!/usr/bin/env python3
"""
Config-driven YOLO training for both detect and OBB tasks.

All parameters in the config's `train:` section are passed directly to
model.train(**kwargs).  Adding a new YOLO parameter requires zero code
changes — just add it to the YAML config.

Usage:
    uv run python scripts/training/train_yolo.py --config configs/H5.yaml
    uv run python scripts/training/train_yolo.py --config configs/H7_obb_6class.yaml --epochs 50
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven YOLO training")
    parser.add_argument("--config", default="configs/H5.yaml",
                        help="Path to project YAML config (default: configs/H5.yaml)")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--device", help="Override device (e.g. 0, 0,1, cpu)")
    parser.add_argument("--name", help="Override experiment name")
    parser.add_argument("--task", help="Override task (detect or obb)")
    return parser.parse_args()


def resolve_data(cfg: dict) -> str:
    """Build a data YAML from the config's `dataset:` section."""
    dataset_cfg = cfg.get("dataset")
    if isinstance(dataset_cfg, dict):
        data_dict = {k: v for k, v in dataset_cfg.items() if v is not None}
        temp_file = Path("data") / "temp_data.yaml"
        temp_file.parent.mkdir(exist_ok=True)
        with open(temp_file, "w") as f:
            yaml.dump(data_dict, f, default_flow_style=False)
        return str(temp_file)
    return "data/data.yaml"


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Build train kwargs from config — every key in train: goes to YOLO
    train_args = {k: v for k, v in (cfg.get("train") or {}).items() if v is not None}

    # Pop keys that are not model.train() parameters
    model_path = train_args.pop("model", "yolo11n.pt")
    task = train_args.pop("task", "detect")

    # Resolve data: explicit train.data > dataset section > default
    if "data" not in train_args:
        train_args["data"] = resolve_data(cfg)

    # CLI overrides (highest priority)
    cli_overrides = {
        "model": args.model, "epochs": args.epochs, "batch": args.batch,
        "imgsz": args.imgsz, "device": args.device, "name": args.name,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            if key == "model":
                model_path = val
            else:
                train_args[key] = val
    if args.task:
        task = args.task

    print(f"Task: {task}")
    print(f"Model: {model_path}")
    print(f"Train args: {train_args}")

    model = YOLO(model_path, task=task)
    model.train(**train_args)

    # Final validation with plots
    print("\nRunning final validation...")
    model.val(data=train_args["data"], plots=True)


if __name__ == "__main__":
    main()
