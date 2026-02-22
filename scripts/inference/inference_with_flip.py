
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Class Definitions for 2-class model
# Model Output Classes (0-1): 0: bot_finger, 1: bot_toe
# Standard pass: keep bot_finger(0) and bot_toe(1)
# Flipped pass: bot_finger(0) -> up_finger, bot_toe(1) -> up_toe
CLASSES = {0: 'bot_finger', 1: 'bot_toe', 2: 'up_finger', 3: 'up_toe'}
COLORS = {0: (255,0,255), 1: (0,255,0), 2: (255,165,0), 3: (0,255,255)}

def flip_point(pt, h):
    return [pt[0], h - 1 - pt[1]]

def run_flip_inference(model, img, conf=0.25, iou=0.4):
    """
    Run inference with flip strategy.
    Returns a list of detections: {'cls': int, 'conf': float, 'corners': np.array}
    """
    h, w = img.shape[:2]
    
    # 1. Standard Inference
    # We keep bot_finger(2), bot_toe(3), ruler(4), id(5)
    # We IGNORE up_finger(0) and up_toe(1) from this pass as they are inaccurate axis-aligned boxes
    results_orig = model.predict(img, imgsz=1280, conf=conf, iou=iou, verbose=False)[0]
    
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

    # 2. Flipped Inference
    flipped_img = cv2.flip(img, 0)
    results_flipped = model.predict(flipped_img, imgsz=1280, conf=conf, iou=iou, verbose=False)[0]
    
    if results_flipped.obb is not None:
        for i in range(len(results_flipped.obb)):
            cls_id = int(results_flipped.obb.cls[i])
            conf_score = float(results_flipped.obb.conf[i])
            
            # Map flipped bot -> original up
            target_cls = None
            if cls_id == 0:  # bot_finger -> up_finger
                target_cls = 2
            elif cls_id == 1:  # bot_toe -> up_toe
                target_cls = 3
            
            if target_cls is not None:
                # Get corners in flipped space
                corners_flipped = results_flipped.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)
                
                # Flip back
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
        color = COLORS.get(cls_id, (255,255,255))
        label = f"{CLASSES.get(cls_id, str(cls_id))} {conf:.2f}"
        
        cv2.polylines(vis_img, [corners], True, color, 4)
        
        tx, ty = int(corners[:,0].min()), int(corners[:,1].min()) - 10
        (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        ty = max(ty, th_t + 10)
        
        cv2.rectangle(vis_img, (tx-2, ty-th_t-8), (tx+tw+2, ty+8), color, -1)
        cv2.putText(vis_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
    return vis_img

def main():
    parser = argparse.ArgumentParser(description="Run inference with flip strategy")
    project_root = os.environ.get("PROJECT_ROOT", "/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads")
    parser.add_argument('--model', default=f'{project_root}/runs/obb/H1_obb_2class2/weights/best.pt', help="Path to model")
    parser.add_argument('--source', required=True, help="Image file or directory")
    parser.add_argument('--output-dir', default='inference_outputs', help="Output directory")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")
    parser.add_argument('--iou', type=float, default=0.4, help="NMS IoU threshold")
    args = parser.parse_args()

    model = YOLO(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source)
    if source_path.is_dir():
        image_files = sorted(list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')))
    else:
        image_files = [source_path]

    print(f"Processing {len(image_files)} images...")

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None: continue
        
        detections = run_flip_inference(model, img, conf=args.conf, iou=args.iou)
        
        # Visualize
        vis_img = draw_detections(img, detections)
        out_path = output_dir / f"{img_file.stem}_flip_inf.jpg"
        cv2.imwrite(str(out_path), vis_img)
        print(f"Saved {out_path}")

        # Optional: Save text labels? (Format: class cx cy w h angle OR class x1 y1 ...?)
        # For now just visualization as requested originally.

if __name__ == '__main__':
    main()
