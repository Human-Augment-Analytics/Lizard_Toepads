"""
Health check script: loads 2 random crops from each pool (train, test, unannotated)
and saves overlay images showing keypoints drawn on the crop.

Usage:
    python preprocess_healthcheck.py [--data-dir /path/to/data] [--output-dir ./healthcheck]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
import numpy as np
import cv2
import torch
from pathlib import Path

SHARED_DATA_DIR = "/storage/ice-shared/cs8903onl/alternative-models/data"


def draw_keypoints(img_bgr, keypoints, color=(0, 255, 0), radius=5):
    """Draw numbered keypoints on an image."""
    out = img_bgr.copy()
    for i, (x, y) in enumerate(keypoints):
        px, py = int(round(float(x))), int(round(float(y)))
        cv2.circle(out, (px, py), radius, color, -1)
        cv2.putText(out, str(i), (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def visualize_crop(pt_path, output_path, label=""):
    """Load a .pt crop file and save a visualization with keypoints overlaid."""
    data = torch.load(pt_path, map_location="cpu")

    img = data["image"].permute(1, 2, 0).numpy().astype(np.uint8)  # CHW → HWC BGR
    tps = data["tps"].numpy()  # (9, 2)
    is_flipped = data.get("flipped", torch.tensor(False)).item()
    class_name = data.get("class_name", "unknown")

    has_annotation = not np.allclose(tps, 0.0)

    if has_annotation:
        overlay = draw_keypoints(img, tps, color=(0, 255, 0))
    else:
        overlay = img.copy()

    # Add label text
    info = f"{label} | class={class_name} | flipped={is_flipped} | annotated={has_annotation}"
    cv2.putText(overlay, info, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, Path(pt_path).name, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), overlay)
    print(f"  Saved: {output_path}")


def sample_files(directory, n=2):
    """Sample n random .pt files from a directory."""
    d = Path(directory)
    if not d.exists():
        print(f"  WARNING: directory not found: {d}")
        return []
    files = list(d.glob("*.pt"))
    if not files:
        print(f"  WARNING: no .pt files in {d}")
        return []
    return random.sample(files, min(n, len(files)))


def main():
    parser = argparse.ArgumentParser(description="Preprocessing health check — visualize sample crops")
    parser.add_argument("--data-dir", type=str, default=SHARED_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default="./healthcheck")
    parser.add_argument("--n", type=int, default=2, help="Number of samples per pool")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pools = [
        ("train", Path(args.data_dir) / "train"),
        ("test", Path(args.data_dir) / "test"),
        ("unannotated", Path(args.data_dir) / "unannotated"),
    ]

    print(f"Health check: sampling {args.n} crops from each pool")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {output_dir}")
    print()

    for pool_name, pool_dir in pools:
        print(f"[{pool_name}]")
        files = sample_files(pool_dir, args.n)
        for idx, f in enumerate(files):
            out_path = output_dir / f"{pool_name}_{idx}.png"
            visualize_crop(str(f), str(out_path), label=pool_name)
        print()

    print("Done. Check the output directory for overlay images.")


if __name__ == "__main__":
    main()
