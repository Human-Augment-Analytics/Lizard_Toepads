#!/usr/bin/env python3
"""Visualize ml-morph TPS landmarks with ground truth and model predictions.

Draws both ground truth (green) and predicted (red) landmarks cropped tightly
around the bounding box from the test XML. Uses the best obb_aligned model
for each digit type.

Usage:
    python scripts/visualization/visualize_mlmorph_landmarks.py
    python scripts/visualization/visualize_mlmorph_landmarks.py --type finger --n 4
    python scripts/visualization/visualize_mlmorph_landmarks.py --type toe --stems 1359 1362
"""
import argparse
import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import dlib
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

COLOR_GT   = (0, 255, 0)      # green  — ground truth
COLOR_PRED = (0, 0, 255)      # red    — prediction
COLOR_LINE = (255, 255, 255)  # white  — error lines


def load_xml(xml_path):
    """Parse dlib XML. Returns list of (img_path, box, gt_pts) dicts."""
    tree = ET.parse(xml_path)
    records = []
    for img_el in tree.findall(".//image"):
        img_file = img_el.attrib["file"]
        box_el = img_el.find("box")
        if box_el is None:
            continue
        box = dlib.rectangle(
            left=int(box_el.attrib["left"]),
            top=int(box_el.attrib["top"]),
            right=int(box_el.attrib["left"]) + int(box_el.attrib["width"]),
            bottom=int(box_el.attrib["top"]) + int(box_el.attrib["height"]),
        )
        parts = sorted(box_el.findall("part"), key=lambda p: int(p.attrib["name"]))
        gt_pts = [(int(p.attrib["x"]), int(p.attrib["y"])) for p in parts]
        records.append({"file": img_file, "box": box, "gt": gt_pts})
    return records


def draw_landmarks(img, pts, color, radius=4, draw_line=True):
    """Draw small filled dots and a connecting polyline."""
    thickness = max(1, radius // 3)
    if draw_line and len(pts) > 1:
        poly = np.array([[x, y] for x, y in pts], dtype=np.int32)
        cv2.polylines(img, [poly], False, color, thickness)
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
        cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), 1)


def draw_error_lines(img, gt_pts, pred_pts):
    """Draw white lines between each GT and predicted landmark."""
    for (gx, gy), (px, py) in zip(gt_pts, pred_pts):
        cv2.line(img, (int(gx), int(gy)), (int(px), int(py)), COLOR_LINE, 1)


def visualize_record(record, predictor, pad_ratio=0.5, dot_radius=4):
    """Load image, run prediction, return cropped annotated image."""
    # Load via PIL and force uint8 RGB (handles 16-bit JPEGs dlib can't read)
    try:
        pil_img = Image.open(record["file"]).convert("RGB")
        img_rgb = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
    except Exception as e:
        print(f"  Could not read: {record['file']} ({e})")
        return None

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    box = record["box"]
    gt_pts = record["gt"]

    # Run dlib predictor on RGB image
    shape = predictor(img_rgb, box)
    pred_pts = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

    # Compute per-landmark pixel errors
    errors = [np.sqrt((gx-px)**2 + (gy-py)**2) for (gx,gy),(px,py) in zip(gt_pts, pred_pts)]
    mean_err = np.mean(errors)

    # Crop with padding around the bounding box
    h, w = img_bgr.shape[:2]
    bw = box.right() - box.left()
    bh = box.bottom() - box.top()
    pad = int(max(bw, bh) * pad_ratio)
    x1 = max(0, box.left() - pad)
    y1 = max(0, box.top() - pad)
    x2 = min(w, box.right() + pad)
    y2 = min(h, box.bottom() + pad)
    crop = img_bgr[y1:y2, x1:x2].copy()

    # Shift points to crop coords
    gt_crop   = [(x - x1, y - y1) for x, y in gt_pts]
    pred_crop = [(x - x1, y - y1) for x, y in pred_pts]

    draw_error_lines(crop, gt_crop, pred_crop)
    draw_landmarks(crop, gt_crop,   COLOR_GT,   radius=dot_radius)
    draw_landmarks(crop, pred_crop, COLOR_PRED, radius=dot_radius)

    # Legend + mean error
    ch, cw = crop.shape[:2]
    fs = max(0.5, cw / 800)
    th = max(1, int(fs * 1.5))
    stem = Path(record["file"]).stem
    cv2.putText(crop, f"{stem}  mean err: {mean_err:.1f}px",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th)
    cv2.putText(crop, "GT",   (8, 44), cv2.FONT_HERSHEY_SIMPLEX, fs, COLOR_GT,   th)
    cv2.putText(crop, "Pred", (8, 66), cv2.FONT_HERSHEY_SIMPLEX, fs, COLOR_PRED, th)

    return crop


def main():
    project_root = os.environ.get("PROJECT_ROOT", "/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads")
    ml_morph = f"{project_root}/ml-morph"

    BEST_MODELS = {
        "finger": f"{ml_morph}/hyperparam_results_finger_obb_aligned/depth3_cascade25_nu0.1_trees500_over30_fp500_splits20_tj0.dat",
        "toe":    f"{ml_morph}/hyperparam_results_toe/depth3_cascade12_nu0.1_trees500_over30_fp500_splits20_tj0.dat",
    }
    TEST_XMLS = {
        "finger": f"{ml_morph}/finger_test_yolo_obb_aligned.xml",
        "toe":    f"{ml_morph}/toe_test_yolo_obb_aligned.xml",
    }

    parser = argparse.ArgumentParser(description="Visualize ml-morph GT vs predicted landmarks")
    parser.add_argument("--type", dest="digit_type", default="both",
                        choices=["finger", "toe", "both"])
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Specific image stems to visualize (default: random)")
    parser.add_argument("--n", type=int, default=2,
                        help="Number of random images per digit type (if --stems not given)")
    parser.add_argument("--output-dir",
                        default=f"{project_root}/data/visualizations/mlmorph_landmarks",
                        help="Output directory")
    parser.add_argument("--width", type=int, default=600,
                        help="Output crop width in pixels")
    parser.add_argument("--dot-radius", type=int, default=3,
                        help="Landmark dot radius in pixels (before scaling)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    types = ["finger", "toe"] if args.digit_type == "both" else [args.digit_type]
    random.seed(args.seed)

    for dtype in types:
        print(f"\n--- {dtype} ---")
        predictor = dlib.shape_predictor(BEST_MODELS[dtype])
        records = load_xml(TEST_XMLS[dtype])

        if args.stems:
            stem_set = set(args.stems)
            selected = [r for r in records if Path(r["file"]).stem in stem_set]
        else:
            selected = random.sample(records, min(args.n, len(records)))

        for record in selected:
            stem = Path(record["file"]).stem
            crop = visualize_record(record, predictor, dot_radius=args.dot_radius)
            if crop is None:
                continue
            scale = args.width / crop.shape[1]
            out = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            out_path = out_dir / f"{stem}_{dtype}_gt_vs_pred.jpg"
            cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
