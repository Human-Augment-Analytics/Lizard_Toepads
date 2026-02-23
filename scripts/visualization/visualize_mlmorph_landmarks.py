#!/usr/bin/env python3
"""Visualize ml-morph TPS landmarks on lizard images.

Draws the 9 biological landmarks (skipping the first 2 ruler points) for
finger and/or toe TPS files. Optionally also draws the ruler points.

TPS format (bottom-left origin):
  LM=11
  x y   <- landmark 0 (ruler pt 1)
  x y   <- landmark 1 (ruler pt 2)
  x y   <- landmark 2  (bio pt 1)
  ...
  x y   <- landmark 10 (bio pt 9)
  IMAGE=<stem>.jpg

Usage:
    python scripts/visualization/visualize_mlmorph_landmarks.py --stems 1001 1003
    python scripts/visualization/visualize_mlmorph_landmarks.py --stems 1001 --show-ruler
    python scripts/visualization/visualize_mlmorph_landmarks.py --stems 1001 --type finger
"""
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Colors (BGR)
COLOR_FINGER  = (255, 0, 255)   # magenta
COLOR_TOE     = (0, 255, 0)     # green
COLOR_RULER   = (0, 165, 255)   # orange
COLOR_INDEX   = (255, 255, 255) # white  (landmark index labels)


def parse_tps(tps_path: Path):
    """Parse a TPS file. Returns list of (x, y) in TPS coords (bottom-left origin)."""
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
    return pts


def tps_to_img(pts, img_height):
    """Convert TPS (bottom-left origin) → image (top-left origin) coords."""
    return [(x, img_height - y) for x, y in pts]


def draw_landmarks(img, points_img, color, label_prefix="", show_index=True, radius=None):
    """Draw filled circles and optional index labels for each landmark."""
    h, w = img.shape[:2]
    r = radius or max(8, w // 400)
    font_scale = max(0.6, w / 4000)
    for i, (x, y) in enumerate(points_img):
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(img, (ix, iy), r, color, -1)
        cv2.circle(img, (ix, iy), r + 2, (0, 0, 0), 2)  # black outline
        if show_index:
            txt = f"{label_prefix}{i}"
            cv2.putText(img, txt, (ix + r + 4, iy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_INDEX, 2)
    # Connect bio landmarks with a line to show order
    if len(points_img) > 1:
        poly = np.array([[int(x), int(y)] for x, y in points_img], dtype=np.int32)
        cv2.polylines(img, [poly], False, color, max(2, w // 1000))


def process_stem(stem, tps_dir, image_dir, show_ruler, digit_type):
    """Load image + TPS files for a stem, return annotated image."""
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(image_dir) / f"{stem}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        print(f"  Image not found for stem: {stem}")
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Could not read: {img_path}")
        return None
    h, w = img.shape[:2]

    types_to_draw = []
    if digit_type in ("finger", "both"):
        types_to_draw.append(("finger", COLOR_FINGER))
    if digit_type in ("toe", "both"):
        types_to_draw.append(("toe", COLOR_TOE))

    for dtype, color in types_to_draw:
        tps_path = Path(tps_dir) / f"{stem}_{dtype}.TPS"
        if not tps_path.exists():
            print(f"  TPS not found: {tps_path}")
            continue
        pts_tps = parse_tps(tps_path)
        if len(pts_tps) < 3:
            print(f"  Too few landmarks in {tps_path}")
            continue

        pts_img = tps_to_img(pts_tps, h)
        ruler_pts = pts_img[:2]
        bio_pts   = pts_img[2:]

        if show_ruler:
            draw_landmarks(img, ruler_pts, COLOR_RULER, label_prefix="r", show_index=True)

        draw_landmarks(img, bio_pts, color, label_prefix="", show_index=True)

        # Label the digit type in the image near the first bio point
        fx, fy = int(bio_pts[0][0]), int(bio_pts[0][1])
        cv2.putText(img, dtype, (fx + 20, fy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, max(1.5, w / 2000), color, 3)

    return img


def main():
    project_root = os.environ.get("PROJECT_ROOT", "/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads")
    parser = argparse.ArgumentParser(description="Visualize ml-morph TPS landmarks")
    parser.add_argument("--tps-dir", default="/storage/ice-shared/cs8903onl/tps_files",
                        help="Directory containing per-image _finger.TPS and _toe.TPS files")
    parser.add_argument("--image-dir", default="/storage/ice-shared/cs8903onl/miami_fall_24_jpgs",
                        help="Directory containing original images")
    parser.add_argument("--stems", nargs="+", default=["1001", "1003"],
                        help="Image stems to visualize")
    parser.add_argument("--type", dest="digit_type", default="both",
                        choices=["finger", "toe", "both"],
                        help="Which digit type to show")
    parser.add_argument("--show-ruler", action="store_true",
                        help="Also draw the 2 ruler/scale landmarks (orange)")
    parser.add_argument("--output-dir",
                        default=f"{project_root}/data/visualizations/mlmorph_landmarks",
                        help="Output directory")
    parser.add_argument("--width", type=int, default=1500,
                        help="Output image width in pixels")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in args.stems:
        vis = process_stem(stem, args.tps_dir, args.image_dir,
                           args.show_ruler, args.digit_type)
        if vis is None:
            continue

        scale = args.width / vis.shape[1]
        out = cv2.resize(vis, None, fx=scale, fy=scale)
        out_path = out_dir / f"{stem}_{args.digit_type}.jpg"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
