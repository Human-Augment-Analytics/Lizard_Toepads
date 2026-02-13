
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import dlib
import sys

# Class Definitions
CLASSES = {0: 'bot_finger', 1: 'bot_toe', 2: 'up_finger', 3: 'up_toe'}
COLORS = {0: (255,0,255), 1: (0,255,0), 2: (255,165,0), 3: (0,255,255)}

def get_dlib_rect(corners, img_shape, padding_ratio=0.3):
    # corners: 4x2 array or similar
    # Apply same padding as training (generate_yolo_bbox_xml.py expand_box)
    x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
    img_h, img_w = img_shape[:2]
    px = int(w * padding_ratio)
    py = int(h * padding_ratio)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(img_w, x + w + px)
    y2 = min(img_h, y + h + py)
    return dlib.rectangle(int(x1), int(y1), int(x2), int(y2))

def run_flip_inference_with_landmarks(model, finger_predictor, toe_predictor, img, conf=0.25):
    h, w = img.shape[:2]
    landmarks_list = []
    detections = []

    # 1. Standard Inference
    results_orig = model.predict(img, imgsz=1280, conf=conf, verbose=False)[0]
    
    if results_orig.obb is not None:
        for i in range(len(results_orig.obb)):
            cls_id = int(results_orig.obb.cls[i])
            corners = results_orig.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)
            
            if cls_id in [0, 1]:  # bot_finger, bot_toe
                rect = get_dlib_rect(corners, img.shape)
                
                # Ensure image is compatible with dlib (uint8, RGB, contiguous)
                if len(img.shape) == 3 and img.shape[2] == 3:
                     # Convert BGR to RGB if needed, or just ensure it's robust
                     # dlib works better with RGB usually
                     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                     img_rgb = img
                
                # Ensure contiguous
                img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

                predictor = finger_predictor if cls_id == 0 else toe_predictor
                shape = predictor(img_rgb, rect)
                
                points = []
                for k in range(shape.num_parts):
                    p = shape.part(k)
                    points.append((p.x, p.y))
                
                landmarks_list.append({
                    'cls': cls_id,
                    'points': points
                })
                detections.append({
                    'cls': cls_id,
                    'corners': corners
                })

    # 2. Flipped Inference
    flipped_img = cv2.flip(img, 0)
    results_flipped = model.predict(flipped_img, imgsz=1280, conf=conf, verbose=False)[0]
    
    if results_flipped.obb is not None:
        for i in range(len(results_flipped.obb)):
            cls_id = int(results_flipped.obb.cls[i])
            corners = results_flipped.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)
            
            # Map flipped bot -> original up
            target_cls = None
            if cls_id == 0:  # bot_finger -> up_finger
                target_cls = 2
            elif cls_id == 1:  # bot_toe -> up_toe
                target_cls = 3
            
            if target_cls is not None:
                if len(flipped_img.shape) == 3 and flipped_img.shape[2] == 3:
                     flipped_img_rgb = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB)
                else:
                     flipped_img_rgb = flipped_img
                flipped_img_rgb = np.ascontiguousarray(flipped_img_rgb, dtype=np.uint8)

                rect = get_dlib_rect(corners, flipped_img.shape)
                predictor = finger_predictor if cls_id == 0 else toe_predictor
                shape = predictor(flipped_img_rgb, rect)
                
                # Flip points back
                points = []
                for k in range(shape.num_parts):
                    p = shape.part(k)
                    # Flip Y coordinate back
                    points.append((p.x, h - 1 - p.y))
                
                # Flip corners back for visualization
                corners_orig = np.array([[p[0], h - 1 - p[1]] for p in corners], dtype=np.float32)
                
                landmarks_list.append({
                    'cls': target_cls,
                    'points': points
                })
                detections.append({
                    'cls': target_cls,
                    'corners': corners_orig
                })
                
    return detections, landmarks_list

def draw_results(img, detections, landmarks_list):
    vis_img = img.copy()
    
    # Draw detections
    for d in detections:
        cls_id = d['cls']
        corners = d['corners'].astype(np.int32)
        color = COLORS.get(cls_id, (255,255,255))
        cv2.polylines(vis_img, [corners], True, color, 4)
        
    # Draw landmarks
    for l in landmarks_list:
        cls_id = l['cls']
        points = l['points']
        color = COLORS.get(cls_id, (255,255,255))
        
        for (x, y) in points:
            cv2.circle(vis_img, (int(x), int(y)), 20, (0, 0, 255), -1) # Red dots for landmarks, larger size for HD image
            
    return vis_img

def main():
    parser = argparse.ArgumentParser(description="Run inference with flip strategy + landmarks")
    parser.add_argument('--model', default='/home/hice1/yloh30/scratch/Lizard_Toepads/runs/obb/H1_obb_2class2/weights/best.pt', help="Path to YOLO model")
    parser.add_argument('--finger-predictor', default='/home/hice1/yloh30/scratch/Lizard_Toepads/ml-morph/finger_predictor_yolo_bbox.dat', help="Path to finger shape predictor")
    parser.add_argument('--toe-predictor', default='/home/hice1/yloh30/scratch/Lizard_Toepads/ml-morph/toe_predictor_yolo_bbox.dat', help="Path to toe shape predictor")
    parser.add_argument('--source', required=True, help="Image file")
    parser.add_argument('--output-dir', default='inference_results', help="Output directory")
    args = parser.parse_args()

    # Load YOLO
    print(f"Loading YOLO model from {args.model}")
    model = YOLO(args.model)

    # Load Predictors
    print(f"Loading Finger Predictor from {args.finger_predictor}")
    finger_predictor = dlib.shape_predictor(args.finger_predictor)
    
    print(f"Loading Toe Predictor from {args.toe_predictor}")
    toe_predictor = dlib.shape_predictor(args.toe_predictor)

    # Resolve source
    source_path = Path(args.source)
    
    # Collect files to process
    files_to_process = []
    if source_path.is_dir():
        all_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        # filter out result files to avoid infinite loop or processing output
        files_to_process = [f for f in all_files if "_landmarks" not in f.name]
    else:
        files_to_process = [source_path]

    print(f"Found {len(files_to_process)} files to process.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for current_file in files_to_process:
        # Determine cleaned model name first to find original texture
        stem = current_file.stem
        # Remove specific patterns as requested to get model name
        stem = stem.replace('.1001', '')
        stem = stem.replace('diffuse', '')
        stem = stem.replace('Diffuse', '')
        stem = stem.replace('_flip_inf', '')
        stem = stem.replace('_landmarks', '')
        # Cleanup formatting
        stem = stem.replace('..', '.')
        stem = stem.replace('__', '_')
        stem = stem.strip('._ ')
        
        # Try to find original clean texture
        # Search paths for the clean texture file (e.g. modelname.jpg)
        # Search paths for the clean texture file (e.g. modelname.jpg)
        potential_paths = [
            Path(f"/storage/ice-shared/cs8903onl/miami_fall_24_jpgs/{stem}.jpg"),
            Path(f"data/processed_obb/images/{stem}.jpg"),
            Path(f"data/dataset_obb/images/train/{stem}.jpg"),
            Path(f"data/miami_fall_24_jpgs/{stem}.jpg"),
            # Also check if the current file itself is the clean texture
            current_file
        ]
        
        texture_file = None
        for p in potential_paths:
            if p.exists() and "_flip_inf" not in p.name and "_landmarks" not in p.name:
                texture_file = p
                break
        
        if texture_file:
            print(f"Processing texture file: {texture_file}")
            try:
                img = cv2.imread(str(texture_file))
                if img is None:
                    print(f"Error: Could not read image {texture_file}")
                    continue
                
                detections, landmarks = run_flip_inference_with_landmarks(model, finger_predictor, toe_predictor, img)
                vis_img = draw_results(img, detections, landmarks)
                
                out_name = stem + ".jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), vis_img)
                print(f"Saved result to {out_path}")
            except Exception as e:
                print(f"Error processing {texture_file}: {e}")
        else:
            print(f"Warning: Texture file for model '{stem}' is missing.")
            print(f"  Please provide it at: /Users/leyangloh/Downloads/3D_Fish/mussels_31 (or verify filename matches model name).")



if __name__ == "__main__":
    main()
