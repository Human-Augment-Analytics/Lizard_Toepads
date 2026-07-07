import sys
import os
import argparse
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

BENCHMARKING_DIR = Path(__file__).parent.resolve()
ALT_MODELS_DIR = BENCHMARKING_DIR.parent

DEFAULT_SPLIT = str(BENCHMARKING_DIR / "splits" / "split.json")

MODELS = [
    {
        "name": "stacked_hourglass",
        "dir": ALT_MODELS_DIR / "stacked-hourglass",
        "train_script": "train.py",
    },
    {
        "name": "vit",
        "dir": ALT_MODELS_DIR / "vit",
        "train_script": "train.py",
    },
    {
        "name": "hrnet",
        "dir": ALT_MODELS_DIR / "hrnet",
        "train_script": "train.py",
    },
    {
        "name": "hrnet_heatmap",
        "dir": ALT_MODELS_DIR / "hrnet",
        "train_script": "train_heatmap.py",
    },
    {
        "name": "hrnet_gcn",
        "dir": ALT_MODELS_DIR / "hrnet-gcn",
        "train_script": "train.py",
    },
    {
        "name": "ml_morph",
        "dir": ALT_MODELS_DIR / "ml-morph",
        "train_script": "train.py",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run all model training runs sequentially")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT,
                        help="Path to shared split.json file")
    args = parser.parse_args()

    split_file = Path(args.split).resolve()
    if not split_file.exists():
        print(f"ERROR: split file not found: {split_file}", file=sys.stderr)
        sys.exit(1)

    results = []

    for model in MODELS:
        name = model["name"]
        model_dir = model["dir"]

        logging.info(f"[{name}] Starting training...")
        start = time.time()

        cmd = [sys.executable, model["train_script"], "--split", str(split_file)]

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
