"""
Run Model A training — HRNet-GCN with mean-shape initialization on WFLW.

Orchestrates train_wflw.py via subprocess, then prints a summary table.
Follows the same conventions as alternative-models/benchmarking/run_all.py.

Usage:
    python alternative-datasets/wflw/run_wflw.py \\
        --split alternative-datasets/wflw/splits/wflw_1.0_seed42.json

    # Override config:
    python alternative-datasets/wflw/run_wflw.py \\
        --split alternative-datasets/wflw/splits/wflw_0.25_seed42.json \\
        --config alternative-models/hrnet-gcn/wflw-config.json
"""
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SCRIPT_DIR = Path(__file__).parent.resolve()
ALT_MODELS_DIR = SCRIPT_DIR.parent.parent / "alternative-models"

DEFAULT_CONFIG = str(ALT_MODELS_DIR / "hrnet-gcn" / "wflw-config.json")

MODELS = [
    {
        "name": "hrnet_gcn_wflw",
        "dir": ALT_MODELS_DIR / "hrnet-gcn",
        "train_script": "train_wflw.py",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Run HRNet-GCN (mean-init) training on WFLW"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Path to WFLW split JSON file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to WFLW training config JSON (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    split_file = Path(args.split).resolve()
    if not split_file.exists():
        print(f"ERROR: split file not found: {split_file}", file=sys.stderr)
        sys.exit(1)

    config_file = Path(args.config).resolve()
    if not config_file.exists():
        print(f"ERROR: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    results = []

    for model in MODELS:
        name = model["name"]
        model_dir = model["dir"]
        train_script = model["train_script"]

        logging.info(f"[{name}] Starting training...")
        start = time.time()

        cmd = [
            sys.executable,
            train_script,
            "--config", str(config_file),
            "--split", str(split_file),
        ]

        result = subprocess.run(cmd, cwd=str(model_dir), check=False)
        elapsed = time.time() - start

        if result.returncode != 0:
            logging.error(f"[{name}] Training failed with exit code {result.returncode}")
        else:
            logging.info(f"[{name}] Training completed in {elapsed:.1f}s")

        results.append({"model": name, "exit_code": result.returncode, "elapsed": elapsed})

    print()
    print("=" * 60)
    print(f"{'Model':<25} {'Exit Code':<12} {'Elapsed (s)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<25} {str(r['exit_code']):<12} {r['elapsed']:<12.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
