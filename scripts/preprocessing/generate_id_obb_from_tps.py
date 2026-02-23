#!/usr/bin/env python3
"""Generate YOLO OBB labels for the id class (class 5) from _id.TPS files.

Each _id.TPS file contains 2 landmarks that are opposite diagonal corners
of the ID label rectangle (TPS bottom-left origin coords).

Conversion:
  1. Parse 2 points in TPS coords (bottom-left origin)
  2. Convert to image coords: (x, img_height - y)
  3. Derive 4 axis-aligned corners from the 2 diagonal points
  4. Normalize by image dimensions
  5. Write YOLO OBB line: 5 x1 y1 x2 y2 x3 y3 x4 y4

Output: one .txt file per image in --output-dir
"""
import argparse
import os
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ID_CLASS = 5


def parse_id_tps(tps_path: Path):
    """Read 2 landmark points from an _id.TPS file.

    Returns list of (x, y) in TPS coords (bottom-left origin), or None if invalid.
    """
    pts = []
    with open(tps_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("LM=") or line.startswith("IMAGE="):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return pts if len(pts) == 2 else None


def tps_to_image_coords(pts, img_height):
    """Convert TPS bottom-left origin to image top-left origin."""
    return [(x, img_height - y) for x, y in pts]


def diagonal_to_four_corners(pt1, pt2):
    """Convert 2 diagonal corners to 4 axis-aligned corners (TL, TR, BR, BL)."""
    x1, y1 = pt1
    x2, y2 = pt2
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return [
        (xmin, ymin),  # top-left
        (xmax, ymin),  # top-right
        (xmax, ymax),  # bottom-right
        (xmin, ymax),  # bottom-left
    ]


def normalize_corners(corners, img_width, img_height):
    """Normalize corner coordinates to [0, 1]."""
    return [
        (max(0.0, min(1.0, x / img_width)),
         max(0.0, min(1.0, y / img_height)))
        for x, y in corners
    ]


def find_image(stem, image_dir):
    """Find image file for a given stem."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(image_dir) / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    project_root = os.environ.get("PROJECT_ROOT", "/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads")
    parser = argparse.ArgumentParser(description="Generate id OBB labels from _id.TPS files")
    parser.add_argument("--tps-dir", default="/storage/ice-shared/cs8903onl/tps_files")
    parser.add_argument("--image-dir", default="/storage/ice-shared/cs8903onl/miami_fall_24_jpgs")
    parser.add_argument("--output-dir", default=f"{project_root}/data/id_obb_from_tps")
    args = parser.parse_args()

    tps_dir = Path(args.tps_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_tps_files = sorted(tps_dir.glob("*_id.TPS"))
    print(f"Found {len(id_tps_files)} _id.TPS files")

    count = 0
    skipped = 0
    for tps_path in id_tps_files:
        stem = tps_path.stem.replace("_id", "")

        img_path = find_image(stem, args.image_dir)
        if img_path is None:
            skipped += 1
            continue

        pts_tps = parse_id_tps(tps_path)
        if pts_tps is None:
            skipped += 1
            continue

        with Image.open(img_path) as img:
            img_width, img_height = img.size

        pts_img = tps_to_image_coords(pts_tps, img_height)
        corners = diagonal_to_four_corners(*pts_img)
        norm = normalize_corners(corners, img_width, img_height)

        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
        line = f"{ID_CLASS} {coords_str}"

        out_path = output_dir / f"{stem}.txt"
        with open(out_path, "w") as f:
            f.write(line + "\n")
        count += 1

    print(f"Generated id OBB labels: {count} written, {skipped} skipped")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
