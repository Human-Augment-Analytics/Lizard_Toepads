import os
import shutil
import glob
import random
import argparse
from pathlib import Path

import yaml
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def merge_datasets(processed_dir, upper_dir, output_dir, viz_count=50):
    """
    Merges 'processed' (bottom view) and 'upper_dataset_roboflow' (upper view) datasets.
    
    Args:
        processed_dir: Path to data/processed
        upper_dir: Path to data/upper_dataset_roboflow/train
        output_dir: Path to data/final_processed_id_bilateral
        viz_count: Number of images to visualize
    """
    
    # Define paths
    proc_images_dir = os.path.join(processed_dir, 'images')
    proc_labels_dir = os.path.join(processed_dir, 'labels')
    
    upper_labels_dir = os.path.join(upper_dir, 'labels')
    
    out_images_dir = os.path.join(output_dir, 'images')
    out_labels_dir = os.path.join(output_dir, 'labels')
    out_viz_dir = os.path.join(output_dir, 'visualizations')
    
    # Create directories
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_viz_dir, exist_ok=True)
    
    # Define class mapping
    # Target order: up_finger, up_toe, bot_finger, bot_toe, ruler, id
    # Indices: 0, 1, 2, 3, 4, 5
    
    # Processed (Bottom) mapping:
    # 0 (Finger) -> 2 (bot_finger)
    # 1 (Toe) -> 3 (bot_toe)
    # 2 (Ruler) -> 4 (ruler)
    # 3 (ID) -> 5 (id)
    bottom_map = {0: 2, 1: 3, 2: 4, 3: 5}
    
    # Upper mapping:
    # 0 (up_finger) -> 0 (up_finger)
    # 1 (up_toe) -> 1 (up_toe)
    upper_map = {0: 0, 1: 1}
    
    class_names = ['up_finger', 'up_toe', 'bot_finger', 'bot_toe', 'ruler', 'id']
    
    # Colors for visualization (R, G, B)
    colors = [
        (255, 0, 0),    # up_finger: Red
        (0, 255, 0),    # up_toe: Green
        (0, 0, 255),    # bot_finger: Blue
        (255, 255, 0),  # bot_toe: Yellow
        (255, 0, 255),  # ruler: Magenta
        (0, 255, 255)   # id: Cyan
    ]
    
    # Get list of IDs from processed labels
    label_files = glob.glob(os.path.join(proc_labels_dir, "*.txt"))
    ids = [os.path.splitext(os.path.basename(f))[0] for f in label_files if f != "classes.txt"]
    
    print(f"Found {len(ids)} samples to process.")
    
    viz_counter = 0
    
    for file_id in tqdm(ids):
        # 1. Copy Image
        src_img_path = os.path.join(proc_images_dir, f"{file_id}.jpg")
        dst_img_path = os.path.join(out_images_dir, f"{file_id}.jpg")
        
        if not os.path.exists(src_img_path):
            print(f"Warning: Image for ID {file_id} not found at {src_img_path}. Skipping.")
            continue
            
        shutil.copy2(src_img_path, dst_img_path)
        
        # 2. Merge Labels
        merged_lines = []
        
        # Read Bottom Labels
        bot_label_path = os.path.join(proc_labels_dir, f"{file_id}.txt")
        if os.path.exists(bot_label_path):
            with open(bot_label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cls_idx = int(parts[0])
                    if cls_idx in bottom_map:
                        new_idx = bottom_map[cls_idx]
                        merged_lines.append(f"{new_idx} {' '.join(parts[1:])}")
        
        # Read Upper Labels
        up_label_path = os.path.join(upper_labels_dir, f"{file_id}.txt")
        if os.path.exists(up_label_path):
            with open(up_label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cls_idx = int(parts[0])
                    if cls_idx in upper_map:
                        new_idx = upper_map[cls_idx]
                        merged_lines.append(f"{new_idx} {' '.join(parts[1:])}")
        
        # Write Merged Labels
        dst_label_path = os.path.join(out_labels_dir, f"{file_id}.txt")
        with open(dst_label_path, 'w') as f:
            f.write('\n'.join(merged_lines))
            
        # 3. Visualization (First N images)
        if viz_counter < viz_count:
            try:
                img = Image.open(dst_img_path).convert("RGB")
                draw = ImageDraw.Draw(img)
                width, height = img.size
                
                # Use default font for visualization labels
                font = ImageFont.load_default()

                for line in merged_lines:
                    parts = line.strip().split()
                    cls_idx = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = (cx - w / 2) * width
                    y1 = (cy - h / 2) * height
                    x2 = (cx + w / 2) * width
                    y2 = (cy + h / 2) * height
                    
                    color = colors[cls_idx % len(colors)]
                    label_name = class_names[cls_idx]
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw label background
                    text_bbox = draw.textbbox((x1, y1), label_name, font=font)
                    draw.rectangle([text_bbox[0], text_bbox[1] - 20, text_bbox[2], text_bbox[1]], fill=color)
                    
                    # Draw text
                    draw.text((x1, y1 - 20), label_name, fill=(255, 255, 255), font=font)
                
                viz_path = os.path.join(out_viz_dir, f"{file_id}_viz.jpg")
                img.save(viz_path)
                viz_counter += 1
            except Exception as e:
                print(f"Error visualizing {file_id}: {e}")

    # Create classes.txt
    with open(os.path.join(out_labels_dir, "classes.txt"), "w") as f:
        f.write('\n'.join(class_names))
        
    print("Merge complete!")
    print(f"Output directory: {output_dir}")
    print(f"Visualizations saved to: {out_viz_dir}")



def parse_args():
    parser = argparse.ArgumentParser(description='Merge bilateral datasets.')
    parser.add_argument('--config', default='configs/H4.yaml', help='Path to YAML config file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load config
    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            
    # Get paths from config
    preprocessing = cfg.get('preprocessing', {})
    
    # 1. processed_dir: The output of the previous step (process_tps_files)
    # In H4.yaml this is 'bottom-view-processed-dir': data/processed_bottom
    processed_dir = preprocessing.get('bottom-view-processed-dir', 'data/processed')
    
    # 2. upper_dir: The additional upper side dataset
    upper_dir = preprocessing.get('addtional-upper-side-data-dir')
    
    # 3. output_dir: The final destination for the merged dataset
    # In H4.yaml this is 'merged-processed-dir'
    output_dir = preprocessing.get('merged-processed-dir', 'data/final_processed_id_bilateral')
    
    if not upper_dir:
        raise ValueError("addtional-upper-side-data-dir not found in config")
        
    print(f"Merging datasets:")
    print(f"  Source (Bottom): {processed_dir}")
    print(f"  Source (Upper):  {upper_dir}")
    print(f"  Destination:     {output_dir}")
    
    merge_datasets(processed_dir, upper_dir, output_dir)
