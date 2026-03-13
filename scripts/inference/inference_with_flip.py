#!/usr/bin/env python3
"""
Run OBB inference with flip strategy (no image splitting).

Two-pass approach:
1. Normal pass: run inference on original image → keep all classes
2. Flip pass: flip image vertically → run inference → keep finger/toe only
   → un-flip OBB coordinates back to original image space
3. Combine both passes with NMS (normal pass has priority)

Usage:
    python scripts/inference/inference_with_flip.py --config configs/H10_obb.yaml --quick-test
    python scripts/inference/inference_with_flip.py --config configs/H10_obb.yaml --source data/images/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


CLASS_COLORS = {
    'finger': (255, 0, 255),
    'toe': (0, 255, 0),
    'ruler': (255, 0, 0),
    'id': (0, 165, 255),
}


def obb_iou(corners1, corners2):
    """Compute IoU between two OBBs using polygon intersection."""
    from shapely.geometry import Polygon
    p1 = Polygon(corners1)
    p2 = Polygon(corners2)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.area + p2.area - inter
    return inter / union if union > 0 else 0.0


def cross_pass_nms(normal_dets, flip_dets, iou_threshold=0.3):
    """Normal pass has priority. Flip pass only adds non-overlapping detections."""
    keep = list(normal_dets)
    for fd in flip_dets:
        suppressed = False
        for kd in keep:
            if obb_iou(fd['corners'], kd['corners']) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(fd)
    return keep


def build_class_config(dataset_cfg):
    """Build class mapping. Returns (class_names, finger/toe IDs for flip pass)."""
    names = dataset_cfg.get('names', [])

    flip_keep_ids = set()
    for i, name in enumerate(names):
        if 'finger' in name.lower() or 'toe' in name.lower():
            flip_keep_ids.add(i)

    return names, flip_keep_ids


def run_flip_inference(model, img, conf=0.25, iou=0.4, imgsz=1280,
                       flip_keep_ids=None):
    """
    Two-pass inference on full image (no splitting):
    1. Normal pass: inference on original image, keep all classes
    2. Flip pass: flip vertically, inference, un-flip coords, keep finger/toe only
    3. Combine with NMS (normal pass has priority)
    """
    h, w = img.shape[:2]

    if flip_keep_ids is None:
        flip_keep_ids = {0, 1}

    normal_dets = []
    flip_dets = []

    # --- Pass 1: Normal inference on original image — keep all classes ---
    results_normal = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    if results_normal.obb is not None:
        for i in range(len(results_normal.obb)):
            corners = results_normal.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)
            normal_dets.append({
                'cls': int(results_normal.obb.cls[i]),
                'conf': float(results_normal.obb.conf[i]),
                'corners': corners,
                'source': 'normal'
            })

    # --- Pass 2: Flip vertically → inference → un-flip — keep finger/toe only ---
    img_flipped = cv2.flip(img, 0)  # vertical flip

    results_flip = model.predict(img_flipped, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    n_flip_raw = len(results_flip.obb) if results_flip.obb is not None else 0
    n_flip_kept = 0
    if results_flip.obb is not None:
        for i in range(len(results_flip.obb)):
            cls_id = int(results_flip.obb.cls[i])
            conf_i = float(results_flip.obb.conf[i])
            if cls_id not in flip_keep_ids:
                continue  # discard ruler/id from flipped pass
            n_flip_kept += 1

            corners_flipped = results_flip.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)
            # Un-flip y: y_original = (h - 1) - y_flipped
            corners = corners_flipped.copy()
            corners[:, 1] = (h - 1) - corners_flipped[:, 1]

            flip_dets.append({
                'cls': cls_id,
                'conf': conf_i,
                'corners': corners,
                'source': 'flip'
            })
    print(f"    [debug] flip raw:{n_flip_raw} kept(finger/toe):{n_flip_kept} after_nms:", end="")

    # --- Combine with NMS (normal pass has priority) ---
    final_detections = cross_pass_nms(normal_dets, flip_dets, iou_threshold=iou)
    n_flip_final = sum(1 for d in final_detections if d['source'] == 'flip')
    print(f"{n_flip_final}")

    return final_detections


def draw_detections(img, detections, class_names):
    vis_img = img.copy()
    for d in detections:
        cls_id = d['cls']
        corners = d['corners'].astype(np.int32)
        conf = d['conf']
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        color = CLASS_COLORS.get(name, (255, 255, 255))
        source_tag = " [flip]" if d['source'] == 'flip' else ""
        label = f"{name}{source_tag} {conf:.2f}"

        cv2.polylines(vis_img, [corners], True, color, 4)

        tx, ty = int(corners[:, 0].min()), int(corners[:, 1].min()) - 10
        (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        ty = max(ty, th_t + 10)

        cv2.rectangle(vis_img, (tx - 2, ty - th_t - 8), (tx + tw + 2, ty + 8), color, -1)
        cv2.putText(vis_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return vis_img


def main():
    parser = argparse.ArgumentParser(description="Run OBB inference with flip strategy")
    parser.add_argument('--config', default='configs/H10_obb.yaml',
                        help='Path to project YAML config')
    parser.add_argument('--model', help="Path to model weights (overrides config)")
    parser.add_argument('--source', help="Image file or directory (default: val set from config)")
    parser.add_argument('--output-dir', help="Output directory")
    parser.add_argument('--conf', type=float, help="Confidence threshold")
    parser.add_argument('--iou', type=float, help="NMS IoU threshold")
    parser.add_argument('--imgsz', type=int, help="Image size")
    parser.add_argument('--quick-test', action='store_true',
                        help="Quick test with 50 random images from val set")
    args = parser.parse_args()

    # Load config
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    train_cfg = cfg.get('train', {})
    inference_cfg = cfg.get('inference', {})
    dataset_cfg = cfg.get('dataset', {})

    class_names, flip_keep_ids = build_class_config(dataset_cfg)
    print(f"Classes: {class_names}")
    print(f"Flip pass keeps: {[class_names[i] for i in sorted(flip_keep_ids)]}")

    # Resolve model path
    if args.model:
        model_path = args.model
    else:
        task = train_cfg.get('task', 'obb')
        name = train_cfg.get('name', 'H10_obb')
        model_path = f"runs/{task}/{name}/weights/best.pt"

    # Resolve inference parameters
    conf = args.conf if args.conf is not None else inference_cfg.get('conf', 0.25)
    iou = args.iou if args.iou is not None else inference_cfg.get('iou', 0.4)
    imgsz = args.imgsz if args.imgsz is not None else inference_cfg.get('imgsz', 1280)

    # Resolve output directory
    name = train_cfg.get('name', 'obb')
    output_dir = Path(args.output_dir or inference_cfg.get('project', 'results')) / f"{name}_flip_inference"

    print(f"Config:     {args.config}")
    print(f"Model:      {model_path}")
    print(f"Conf: {conf}, IoU: {iou}, ImgSz: {imgsz}")
    print(f"Output dir: {output_dir}")

    model = YOLO(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve source
    if args.source:
        source_path = Path(args.source)
    else:
        val_rel = dataset_cfg.get('val', 'images/val')
        source_path = Path(dataset_cfg.get('path', 'data')) / val_rel
        print(f"Using val set: {source_path}")

    if source_path.is_dir():
        image_files = sorted(list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')))
    else:
        image_files = [source_path]

    if args.quick_test:
        import random
        random.seed(42)
        n = min(50, len(image_files))
        image_files = sorted(random.sample(image_files, n))
        print(f"Quick test: {n} images")

    print(f"Processing {len(image_files)} images...")

    # Single results file
    results_file = output_dir / "results.txt"
    with open(results_file, 'w') as rf:
        rf.write("image class class_name x1 y1 x2 y2 x3 y3 x4 y4 conf source\n")

        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            h, w = img.shape[:2]
            detections = run_flip_inference(model, img, conf=conf, iou=iou, imgsz=imgsz,
                                            flip_keep_ids=flip_keep_ids)

            # Save visualization
            vis_img = draw_detections(img, detections, class_names)
            out_path = output_dir / f"{img_file.stem}_flip_inf.jpg"
            cv2.imwrite(str(out_path), vis_img)

            # Append to results file (normalized coordinates)
            for d in detections:
                corners_norm = d['corners'].copy()
                corners_norm[:, 0] /= w
                corners_norm[:, 1] /= h
                coords = ' '.join(f"{c:.6f}" for c in corners_norm.flatten())
                cname = class_names[d['cls']] if d['cls'] < len(class_names) else str(d['cls'])
                rf.write(f"{img_file.stem} {d['cls']} {cname} {coords} {d['conf']:.4f} {d['source']}\n")

            n_normal = sum(1 for d in detections if d['source'] == 'normal')
            n_flip = sum(1 for d in detections if d['source'] == 'flip')
            print(f"  {img_file.name}: {len(detections)} detections (normal:{n_normal} flip:{n_flip})")

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
