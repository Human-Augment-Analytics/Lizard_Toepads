#!/usr/bin/env python3
"""
Run OBB inference with vertical flip strategy.

The flip strategy detects upper-view limbs by:
1. Standard pass → detect bot_finger, bot_toe (oriented bounding boxes)
2. Flip image vertically → upper limbs now look like bottom limbs
3. Run inference again → map flipped detections back as up_finger, up_toe

Usage:
    python scripts/inference/inference_with_flip.py --config configs/H8_obb_noflip.yaml --source data/images/
    python scripts/inference/inference_with_flip.py --model runs/obb/H8_obb_noflip/weights/best.pt --source img.jpg
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# Class definitions for flip inference output (4-class result)
# Standard pass: keep bot_finger(0) and bot_toe(1) from 2-class model,
# or bot_finger(2) and bot_toe(3) from 6-class model
# Flipped pass: bot → up mapping
CLASSES = {0: 'bot_finger', 1: 'bot_toe', 2: 'up_finger', 3: 'up_toe'}
COLORS = {0: (255, 0, 255), 1: (0, 255, 0), 2: (255, 165, 0), 3: (0, 255, 255)}


def flip_point(pt, h):
    return [pt[0], h - 1 - pt[1]]


def run_flip_inference(model, img, conf=0.25, iou=0.4, imgsz=1280):
    """
    Run inference with flip strategy.
    Returns a list of detections: {'cls': int, 'conf': float, 'corners': np.array}
    """
    h, w = img.shape[:2]

    # 1. Standard Inference — keep bot_finger and bot_toe
    results_orig = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]

    final_detections = []

    if results_orig.obb is not None:
        for i in range(len(results_orig.obb)):
            cls_id = int(results_orig.obb.cls[i])
            conf_score = float(results_orig.obb.conf[i])
            corners = results_orig.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)

            if cls_id in [0, 1]:  # bot_finger, bot_toe
                final_detections.append({
                    'cls': cls_id,
                    'conf': conf_score,
                    'corners': corners,
                    'source': 'original'
                })

    # 2. Flipped Inference — map bot → up
    flipped_img = cv2.flip(img, 0)
    results_flipped = model.predict(flipped_img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]

    if results_flipped.obb is not None:
        for i in range(len(results_flipped.obb)):
            cls_id = int(results_flipped.obb.cls[i])
            conf_score = float(results_flipped.obb.conf[i])

            # Map flipped bot -> original up
            target_cls = None
            if cls_id == 0:    # bot_finger -> up_finger
                target_cls = 2
            elif cls_id == 1:  # bot_toe -> up_toe
                target_cls = 3

            if target_cls is not None:
                corners_flipped = results_flipped.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)
                corners_orig = np.array([flip_point(p, h) for p in corners_flipped], dtype=np.float32)

                final_detections.append({
                    'cls': target_cls,
                    'conf': conf_score,
                    'corners': corners_orig,
                    'source': 'flipped'
                })

    return final_detections


def draw_detections(img, detections):
    vis_img = img.copy()
    for d in detections:
        cls_id = d['cls']
        corners = d['corners'].astype(np.int32)
        conf = d['conf']
        color = COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASSES.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.polylines(vis_img, [corners], True, color, 4)

        tx, ty = int(corners[:, 0].min()), int(corners[:, 1].min()) - 10
        (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        ty = max(ty, th_t + 10)

        cv2.rectangle(vis_img, (tx - 2, ty - th_t - 8), (tx + tw + 2, ty + 8), color, -1)
        cv2.putText(vis_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return vis_img


def main():
    parser = argparse.ArgumentParser(description="Run OBB inference with flip strategy")
    parser.add_argument('--config', default='configs/H8_obb_noflip.yaml',
                        help='Path to project YAML config (default: configs/H8_obb_noflip.yaml)')
    parser.add_argument('--model', help="Path to model weights (overrides config)")
    parser.add_argument('--source', required=True, help="Image file or directory")
    parser.add_argument('--output-dir', help="Output directory (default: results/<name>_flip_inference)")
    parser.add_argument('--conf', type=float, help="Confidence threshold (overrides config)")
    parser.add_argument('--iou', type=float, help="NMS IoU threshold (overrides config)")
    parser.add_argument('--imgsz', type=int, help="Image size (overrides config)")
    args = parser.parse_args()

    # Load config
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    train_cfg = cfg.get('train', {})
    inference_cfg = cfg.get('inference', {})

    # Resolve model path: CLI > runs/{task}/{name}/weights/best.pt
    if args.model:
        model_path = args.model
    else:
        task = train_cfg.get('task', 'obb')
        name = train_cfg.get('name', 'H8_obb_noflip')
        model_path = f"runs/{task}/{name}/weights/best.pt"

    # Resolve inference parameters: CLI > config > defaults
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

    source_path = Path(args.source)
    if source_path.is_dir():
        image_files = sorted(list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')))
    else:
        image_files = [source_path]

    print(f"Processing {len(image_files)} images...")

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        detections = run_flip_inference(model, img, conf=conf, iou=iou, imgsz=imgsz)

        vis_img = draw_detections(img, detections)
        out_path = output_dir / f"{img_file.stem}_flip_inf.jpg"
        cv2.imwrite(str(out_path), vis_img)
        print(f"  {img_file.name}: {len(detections)} detections -> {out_path}")


if __name__ == '__main__':
    main()
