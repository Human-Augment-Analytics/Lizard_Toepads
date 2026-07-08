"""
Run HRNet heatmap regression training on WFLW.

Parallel to alternative-datasets/WFLW/run_wflw.py (which runs the GCN model).
Orchestrates train_heatmap_wflw.py via subprocess.

Usage:
    python alternative-models/hrnet/run_wflw.py \\
        --split alternative-datasets/wflw/splits/wflw_0.8_seed42.json

    # Override config:
    python alternative-models/hrnet/run_wflw.py \\
        --split alternative-datasets/wflw/splits/wflw_0.25_seed42.json \\
        --config alternative-models/hrnet/configs/wflw-config.json
"""
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = str(SCRIPT_DIR / "configs" / "wflw-config.json")

MODEL_NAME = "hrnet_heatmap_wflw"


def main():
    parser = argparse.ArgumentParser(
        description="Run HRNet heatmap training on WFLW"
    )
    parser.add_argument("--split", type=str, required=True,
                        help="Path to WFLW split JSON file")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"Path to WFLW config JSON (default: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    split_file = Path(args.split).resolve()
    if not split_file.exists():
        print(f"ERROR: split file not found: {split_file}", file=sys.stderr)
        sys.exit(1)

    config_file = Path(args.config).resolve()
    if not config_file.exists():
        print(f"ERROR: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    logging.info(f"[{MODEL_NAME}] Starting training...")
    start = time.time()

    cmd = [
        sys.executable,
        "train_heatmap_wflw.py",
        "--config", str(config_file),
        "--split",  str(split_file),
    ]

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        logging.error(f"[{MODEL_NAME}] Training failed with exit code {result.returncode}")
    else:
        logging.info(f"[{MODEL_NAME}] Training completed in {elapsed:.1f}s")

    print()
    print("=" * 60)
    print(f"{'Model':<30} {'Exit Code':<12} {'Elapsed (s)':<12}")
    print("-" * 60)
    print(f"{MODEL_NAME:<30} {str(result.returncode):<12} {elapsed:<12.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
