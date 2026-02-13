#!/usr/bin/env python3
"""
Visualize the progression: YOLO bbox -> YOLO-OBB -> Landmarks
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import dlib
import sys

# Class definitions
CLASSES = {0: 'bot_finger', 1: 'bot_toe', 2: 'up_finger', 3: 'up_toe'}
COLORS = {0: (255, 0, 255), 1: (0, 255, 0), 2: (255, 165, 0), 3: (0, 255, 255)}

def visualize_yolo_bbox(img_path, model_path, output_path):
    """Visualize standard YOLO bounding boxes"""
    model = YOLO(model_path)
    img = cv2.imread(str(img_path))
    
    results = model.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
    vis_img = img.copy()
    
    if results.boxes is not None:
        for i in range(len(results.boxes)):
            cls_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])
            x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy().astype(int)
            
            color = COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 8)
            
            label = f"{CLASSES.get(cls_id, str(cls_id))} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
            cv2.rectangle(vis_img, (x1, y1 - th - 20), (x1 + tw + 10, y1), color, -1)
            cv2.putText(vis_img, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    
    cv2.imwrite(str(output_path), vis_img)
    print(f"Saved YOLO bbox visualization to {output_path}")

def visualize_yolo_obb(img_path, model_path, output_path):
    """Visualize YOLO-OBB (oriented bounding boxes)"""
    model = YOLO(model_path)
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    vis_img = img.copy()
    
    # Standard inference
    results_orig = model.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
    
    if results_orig.obb is not None:
        for i in range(len(results_orig.obb)):
            cls_id = int(results_orig.obb.cls[i])
            conf = float(results_orig.obb.conf[i])
            corners = results_orig.obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
            
            if cls_id in [0, 1]:  # bot_finger, bot_toe
                color = COLORS.get(cls_id, (255, 255, 255))
                cv2.polylines(vis_img, [corners], True, color, 8)
                
                label = f"{CLASSES.get(cls_id, str(cls_id))} {conf:.2f}"
                tx, ty = int(corners[:, 0].min()), int(corners[:, 1].min()) - 20
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
                ty = max(ty, th + 20)
                cv2.rectangle(vis_img, (tx - 5, ty - th - 15), (tx + tw + 5, ty + 5), color, -1)
                cv2.putText(vis_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    
    # Flipped inference
    flipped_img = cv2.flip(img, 0)
    results_flipped = model.predict(flipped_img, imgsz=1280, conf=0.25, verbose=False)[0]
    
    if results_flipped.obb is not None:
        for i in range(len(results_flipped.obb)):
            cls_id = int(results_flipped.obb.cls[i])
            conf = float(results_flipped.obb.conf[i])
            corners = results_flipped.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)
            
            target_cls = None
            if cls_id == 0:  # bot_finger -> up_finger
                target_cls = 2
            elif cls_id == 1:  # bot_toe -> up_toe
                target_cls = 3
            
            if target_cls is not None:
                # Flip corners back
                corners_orig = np.array([[p[0], h - 1 - p[1]] for p in corners], dtype=np.int32)
                
                color = COLORS.get(target_cls, (255, 255, 255))
                cv2.polylines(vis_img, [corners_orig], True, color, 8)
                
                label = f"{CLASSES.get(target_cls, str(target_cls))} {conf:.2f}"
                tx, ty = int(corners_orig[:, 0].min()), int(corners_orig[:, 1].min()) - 20
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
                ty = max(ty, th + 20)
                cv2.rectangle(vis_img, (tx - 5, ty - th - 15), (tx + tw + 5, ty + 5), color, -1)
                cv2.putText(vis_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    
    cv2.imwrite(str(output_path), vis_img)
    print(f"Saved YOLO-OBB visualization to {output_path}")

def main():
    # Paths
    img_path = Path("/storage/ice-shared/cs8903onl/miami_fall_24_jpgs/1001.jpg")
    bbox_model = Path("/home/hice1/yloh30/scratch/Lizard_Toepads/yolo_bounding_box.pt")
    obb_model = Path("/home/hice1/yloh30/scratch/Lizard_Toepads/runs/obb/H1_obb_2class2/weights/best.pt")
    output_dir = Path("inference_results/progression")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. YOLO bounding boxes
    print("\n1. YOLO Bounding Boxes...")
    visualize_yolo_bbox(img_path, bbox_model, output_dir / "1001_yolo_bbox.jpg")
    
    # 2. YOLO-OBB
    print("\n2. YOLO-OBB (with flip strategy)...")
    visualize_yolo_obb(img_path, obb_model, output_dir / "1001_yolo_obb.jpg")
    
    # 3. Landmarks (already generated)
    print("\n3. Landmarks already generated at inference_results/1001.jpg")
    
    print("\n✓ All visualizations complete!")
    print(f"  - YOLO bbox: {output_dir / '1001_yolo_bbox.jpg'}")
    print(f"  - YOLO-OBB: {output_dir / '1001_yolo_obb.jpg'}")
    print(f"  - Landmarks: inference_results/1001.jpg")

if __name__ == "__main__":
    main()
