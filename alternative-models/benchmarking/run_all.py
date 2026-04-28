import sys
import os
import json
import argparse
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

BENCHMARKING_DIR = Path(__file__).parent.resolve()
ALT_MODELS_DIR = BENCHMARKING_DIR.parent

MODELS = [
    {
        "name":       "stacked_hourglass",
        "dir":        ALT_MODELS_DIR / "stacked-hourglass",
        "split_file": BENCHMARKING_DIR / "splits" / "stacked_hourglass_split.json",
        "data_key":   "stacked_hourglass_data_dir",
        "train_script": "train.py",
    },
    {
        "name":       "vit",
        "dir":        ALT_MODELS_DIR / "vit",
        "split_file": BENCHMARKING_DIR / "splits" / "vit_split.json",
        "data_key":   "vit_data_dir",
        "train_script": "train.py",
    },
    {
        "name":       "hrnet",
        "dir":        ALT_MODELS_DIR / "hrnet",
        "split_file": BENCHMARKING_DIR / "splits" / "hrnet_split.json",
        "data_key":   "hrnet_data_dir",
        "train_script": "train.py",
    },
    {
        "name":       "hrnet_gcn",
        "dir":        ALT_MODELS_DIR / "hrnet-gcn",
        "split_file": BENCHMARKING_DIR / "splits" / "hrnet_gcn_split.json",
        "data_key":   "hrnet_gcn_data_dir",
        "train_script": "train.py",
    },
]

REQUIRED_FIELDS = [
    "split_seed", "train_val_ratio",
    "stacked_hourglass_data_dir", "vit_data_dir",
    "hrnet_data_dir", "hrnet_gcn_data_dir",
]


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


def main():
    parser = argparse.ArgumentParser(description="Run all model training runs sequentially")
    parser.add_argument("--config", type=str, default=str(BENCHMARKING_DIR / "benchmark_config.json"))
    args = parser.parse_args()

    config = load_config(args.config)

    results = []

    for model in MODELS:
        name = model["name"]
        split_file = model["split_file"]
        model_dir = model["dir"]
        data_dir = config[model["data_key"]]

        if not split_file.exists():
            logging.warning(f"[{name}] Split file not found: {split_file} — skipping")
            results.append({"model": name, "exit_code": "SKIPPED", "elapsed": 0.0})
            continue

        logging.info(f"[{name}] Starting training...")
        start = time.time()

        cmd = [sys.executable, model["train_script"], "--split", str(split_file), "--data", data_dir]

        result = subprocess.run(cmd, cwd=str(model_dir), check=False)

        elapsed = time.time() - start

        if result.returncode != 0:
            logging.error(f"[{name}] Training failed with exit code {result.returncode}")
        else:
            logging.info(f"[{name}] Training completed in {elapsed:.1f}s")

        results.append({"model": name, "exit_code": result.returncode, "elapsed": elapsed})

    print()
    print("=" * 60)
    print(f"{'Model':<20} {'Exit Code':<12} {'Elapsed (s)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} {str(r['exit_code']):<12} {r['elapsed']:<12.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
