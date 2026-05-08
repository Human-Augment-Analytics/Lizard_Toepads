import sys
import os
import json
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SHARED_DATA_DIR = "/storage/ice-shared/cs8903onl/alternative-models/data"


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
    parser = argparse.ArgumentParser(description="Generate shared train/val split file")
    parser.add_argument("--data-dir", type=str, default=SHARED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.8)
    args = parser.parse_args()

    train_dir = Path(args.data_dir) / "train"
    if not train_dir.exists():
        print(f"ERROR: train directory not found: {train_dir}", file=sys.stderr)
        sys.exit(1)

    files = [str(p.resolve()) for p in train_dir.glob("*.pt")]
    if not files:
        print(f"ERROR: no .pt files found in {train_dir}", file=sys.stderr)
        sys.exit(1)

    logging.info(f"Found {len(files)} .pt files in {train_dir}")

    split = generate_split(files, args.seed, args.ratio)

    splits_dir = Path(__file__).parent / "splits"
    out_path = splits_dir / "split.json"
    write_split(split, str(out_path))


if __name__ == "__main__":
    main()
