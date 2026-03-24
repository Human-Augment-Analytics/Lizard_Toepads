#!/usr/bin/env python3
"""Generate tight OBB labels for bot_finger and bot_toe from TPS landmark files.

Uses cv2.minAreaRect on biological landmarks (skipping first 2 ruler points)
to compute the minimum-area oriented bounding box, then adds configurable padding.

Output: YOLO OBB format (class x1 y1 x2 y2 x3 y3 x4 y4, normalized)
  Class IDs match 6-class scheme: 2=bot_finger, 3=bot_toe
"""
import argparse
import os
import re
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# 6-class scheme
BOT_FINGER_CLS = 2
BOT_TOE_CLS = 3


def read_tps_landmarks(tps_path):
    """Read landmarks from TPS file. Returns list of (x, y) in TPS coords (bottom-left origin)."""
    pts = []
    with open(tps_path) as f:
        for line in f:
            line = line.strip()
            if re.match(r'^-?\d', line):
                parts = line.split()
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
    return pts


def tps_to_image_coords(pts, img_height):
    """Convert TPS (bottom-left origin) to image (top-left origin) coordinates."""
    return [(x, img_height - y) for x, y in pts]


def compute_obb(landmarks_img, img_width, img_height,
                long_padding_ratio=0.15, short_padding_ratio=2.5):
    """Compute minimum-area OBB from image-coordinate landmarks with asymmetric padding.

    Landmarks mark specific points on the toepad/finger pad, so the minAreaRect
    fits tightly around the points. The long axis (along the digit) is already
    well-represented, but the short axis (across the pad width) needs much more
    padding to encompass the full pad structure.

    Returns 4 corners as normalized [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] or None.
    """
    if len(landmarks_img) < 3:
        return None

    pts = np.array(landmarks_img, dtype=np.float32)
    rect = cv2.minAreaRect(pts)  # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect

    # Apply asymmetric padding: small on long axis, large on short axis
    # cv2.minAreaRect returns (w, h) where w and h are not necessarily
    # ordered by size, so we identify which is the longer dimension.
    if w >= h:
        w *= (1.0 + long_padding_ratio)
        h *= (1.0 + short_padding_ratio)
    else:
        w *= (1.0 + short_padding_ratio)
        h *= (1.0 + long_padding_ratio)

    # Get 4 corners of the padded rotated rect
    padded_rect = ((cx, cy), (w, h), angle)
    corners = cv2.boxPoints(padded_rect)

    # Normalize to [0, 1] and clamp
    norm = []
    for x, y in corners:
        norm.append((
            max(0.0, min(1.0, x / img_width)),
            max(0.0, min(1.0, y / img_height)),
        ))
    return norm


def process_image(img_stem, tps_dir, image_dir):
    """Process one image: read TPS files, compute OBBs, return label lines."""
    finger_tps = os.path.join(tps_dir, f"{img_stem}_finger.TPS")
    toe_tps = os.path.join(tps_dir, f"{img_stem}_toe.TPS")
    img_path = None
    for ext in ('.jpg', '.jpeg', '.png'):
        candidate = os.path.join(image_dir, f"{img_stem}{ext}")
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        return []

    with Image.open(img_path) as img:
        img_width, img_height = img.size

    lines = []

    # Bot finger
    if os.path.exists(finger_tps):
        pts_tps = read_tps_landmarks(finger_tps)
        bio_pts = pts_tps[2:]  # skip ruler
        bio_img = tps_to_image_coords(bio_pts, img_height)
        corners = compute_obb(bio_img, img_width, img_height)
        if corners:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
            lines.append(f"{BOT_FINGER_CLS} {coords}")

    # Bot toe
    if os.path.exists(toe_tps):
        pts_tps = read_tps_landmarks(toe_tps)
        bio_pts = pts_tps[2:]  # skip ruler
        bio_img = tps_to_image_coords(bio_pts, img_height)
        corners = compute_obb(bio_img, img_width, img_height)
        if corners:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
            lines.append(f"{BOT_TOE_CLS} {coords}")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate OBB labels from TPS files")
    parser.add_argument("--tps-dir", default="/storage/ice-shared/cs8903onl/tps_files")
    parser.add_argument("--image-dir", default="/storage/ice-shared/cs8903onl/miami_fall_24_jpgs")
    project_root = os.environ.get("PROJECT_ROOT", "/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads")
    parser.add_argument("--output-dir", default=f"{project_root}/data/bot_obb_from_tps")
    parser.add_argument("--long-padding-ratio", type=float, default=0.15, help="Padding ratio for long axis (default: 0.15 = 15%%)")
    parser.add_argument("--short-padding-ratio", type=float, default=2.5, help="Padding ratio for short axis (default: 2.5 = 250%%)")
    parser.add_argument("--visualize", type=int, default=0, help="Number of images to visualize (0=none)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all image stems that have at least one TPS file
    tps_files = os.listdir(args.tps_dir)
    stems = set()
    for f in tps_files:
        if f.endswith('.TPS'):
            stem = f.replace('_finger.TPS', '').replace('_toe.TPS', '').replace('_id.TPS', '')
            stems.add(stem)

    stems = sorted(stems)
    print(f"Found {len(stems)} image stems with TPS files")

    count = 0
    for stem in stems:
        lines = process_image(stem, args.tps_dir, args.image_dir)
        if lines:
            label_path = os.path.join(args.output_dir, f"{stem}.txt")
            with open(label_path, 'w') as f:
                f.write("\n".join(lines) + "\n")
            count += 1

    print(f"Generated OBB labels for {count} images in {args.output_dir}")

    # Optional visualization
    if args.visualize > 0:
        import random
        random.seed(42)
        vis_stems = random.sample(stems[:count], min(args.visualize, count))
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        crops = []
        for stem in vis_stems:
            label_path = os.path.join(args.output_dir, f"{stem}.txt")
            img_path = os.path.join(args.image_dir, f"{stem}.jpg")
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue

            img_bgr = cv2.imread(img_path)
            h, w = img_bgr.shape[:2]

            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    corners = np.array([
                        [coords[0] * w, coords[1] * h],
                        [coords[2] * w, coords[3] * h],
                        [coords[4] * w, coords[5] * h],
                        [coords[6] * w, coords[7] * h],
                    ], dtype=np.int32)
                    color = (0, 255, 0) if cls_id == BOT_TOE_CLS else (255, 0, 255)
                    label = "bot_toe" if cls_id == BOT_TOE_CLS else "bot_finger"
                    cv2.polylines(img_bgr, [corners], True, color, thickness=4)
                    cx = int(np.mean(corners[:, 0]))
                    cy = int(np.min(corners[:, 1])) - 15
                    cv2.putText(img_bgr, label, (cx - 60, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            scale = 2000 / max(h, w)
            out = cv2.resize(img_bgr, None, fx=scale, fy=scale)
            vis_path = os.path.join(vis_dir, f"{stem}_obb.jpg")
            cv2.imwrite(vis_path, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Visualization: {vis_path}")


if __name__ == "__main__":
    main()
