import sys
import os
import json
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

REQUIRED_FIELDS = [
    "split_seed",
    "train_val_ratio",
    "stacked_hourglass_data_dir",
    "vit_data_dir",
    "hrnet_data_dir",
    "hrnet_gcn_data_dir",
]


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: benchmark_config.json not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: malformed JSON in {path}: {e}", file=sys.stderr)
            sys.exit(1)
    for field in REQUIRED_FIELDS:
        if field not in config:
            print(f"ERROR: required field '{field}' missing from {path}", file=sys.stderr)
            sys.exit(1)
    return config


def generate_split(file_paths: list, seed: int, ratio: float) -> dict:
    sorted_paths = sorted(file_paths)
    train_paths, val_paths = train_test_split(sorted_paths, train_size=ratio, random_state=seed)
    return {"train": train_paths, "val": val_paths, "seed": seed}


def write_split(split: dict, out_path: str) -> None:
    p = Path(out_path)
    if p.exists():
        logging.warning(f"Overwriting existing split file: {out_path}")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(split, f, indent=2)
    logging.info(f"Wrote split ({len(split['train'])} train, {len(split['val'])} val) to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate shared train/val split files")
    parser.add_argument("--config", type=str, default="benchmark_config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["split_seed"]
    ratio = config["train_val_ratio"]

    splits_dir = Path(args.config).parent / "splits"

    models = [
        ("stacked_hourglass", config["stacked_hourglass_data_dir"], "*.npz"),
        ("vit",               config["vit_data_dir"],               "*.pt"),
        ("hrnet",             config["hrnet_data_dir"],             "*.pt"),
        ("hrnet_gcn",         config["hrnet_gcn_data_dir"],         "*.pt"),
    ]

    for name, data_dir, pattern in models:
        d = Path(data_dir)
        if not d.exists():
            logging.error(f"[{name}] Data directory not found: {data_dir} — skipping")
            continue
        files = [str(p.resolve()) for p in d.glob(pattern)]
        if not files:
            logging.error(f"[{name}] No {pattern} files found in {data_dir} — skipping")
            continue
        split = generate_split(files, seed, ratio)
        out_path = splits_dir / f"{name}_split.json"
        write_split(split, str(out_path))


if __name__ == "__main__":
    main()
