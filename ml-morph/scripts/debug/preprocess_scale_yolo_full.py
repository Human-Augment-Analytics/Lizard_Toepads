#!/usr/bin/env python3
"""
Preprocess consolidated_scale.tps using YOLO bounding boxes for full images (no cropping).

This script:
1. Uses YOLO model to detect bounding boxes (ruler/scale) in full images
2. Keeps full images (no cropping)
3. Uses YOLO bounding box as the dlib training bounding box
4. Generates train.xml and test.xml for dlib training on full images
"""

import argparse
import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import utils
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Increase PIL image size limit for large images
Image.MAX_IMAGE_PIXELS = None

# YOLO class mapping: 0=finger, 1=toe, 2=ruler
YOLO_CLASSES = ['finger', 'toe', 'ruler']


def read_tps(input_file):
    """Read TPS file and return landmarks and image names."""
    tps_file = open(input_file, 'r')
    tps = tps_file.read().splitlines()
    tps_file.close()
    
    lm, im, sc, coords_array = [], [], [], []
    
    for i, ln in enumerate(tps):
        if ln.startswith("LM"):
            lm_num = int(ln.split('=')[1])
            lm.append(lm_num)
            coords_mat = []
            for j in range(i + 1, i + 1 + lm_num):
                coords_mat.append(tps[j].split(' '))
            coords_mat = np.array(coords_mat, dtype=float)
            coords_array.append(coords_mat)
        
        if ln.startswith("IMAGE"):
            im.append(ln.split('=')[1])
        
        if ln.startswith("SCALE"):
            sc.append(ln.split('=')[1])
    
    return {'lm': lm, 'im': im, 'scl': sc, 'coords': coords_array}


def detect_bounding_box_yolo(image_path, yolo_model, conf_threshold=0.25, target_class='ruler'):
    """
    Use YOLO to detect bounding box for the target class (ruler/scale).
    
    Args:
        image_path: Path to original image
        yolo_model: Loaded YOLO model
        conf_threshold: Confidence threshold for detection
        target_class: Class to detect ('ruler', 'toe', or 'finger')
    
    Returns:
        bbox: (x1, y1, x2, y2) bounding box coordinates, or None if not found
    """
    # Load image
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil)
    img_height, img_width = img_array.shape[:2]
    
    # Run YOLO detection
    yolo_results = yolo_model.predict(
        img_pil,
        conf=conf_threshold,
        verbose=False
    )
    
    # Find target class detections
    target_class_id = YOLO_CLASSES.index(target_class) if target_class in YOLO_CLASSES else 2  # default to ruler
    
    detections = []
    if yolo_results and yolo_results[0].boxes is not None:
        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0])
            if class_id == target_class_id:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence
                })
    
    if not detections:
        print(f"Warning: No {target_class} detections found for {Path(image_path).name}!")
        # Use full image as fallback
        return (0, 0, img_width, img_height)
    
    # Use the first (highest confidence) detection
    detection = detections[0]
    return detection['bbox']


def generate_dlib_xml_with_bbox(images_data, sizes, bboxes, folder='train', out_file='output.xml',
                                padding_x=50, padding_y_top=20, padding_y_bottom=200):
    """
    Generate dlib XML file with custom bounding boxes (from YOLO).
    
    Args:
        images_data: Dict with 'im' (image names) and 'coords' (landmarks)
        sizes: Dict mapping image names to [height, width]
        bboxes: Dict mapping image names to (x1, y1, x2, y2) bounding boxes in top-left origin
        folder: Folder name for images
        out_file: Output XML file path
    """
    root = ET.Element('dataset')
    root.append(ET.Element('name'))
    root.append(ET.Element('comment'))
    
    images_e = ET.Element('images')
    root.append(images_e)
    
    for i in range(len(images_data['im'])):
        name = os.path.splitext(images_data['im'][i])[0] + '.jpg'
        path = os.path.join(folder, name)
        
        if name in sizes:
            sz = sizes[name][0]  # height
            img_width = sizes[name][1]  # width
            coords = images_data['coords'][i]  # Landmarks in bottom-left origin (from TPS)
            
            # Get YOLO bounding box for this image
            if name in bboxes:
                bbox_yolo = bboxes[name]
                x1, y1, x2, y2 = bbox_yolo
                
                # Expand bounding box with padding (passed as parameters)
                x1_padded = max(0, int(x1 - padding_x))
                y1_padded = max(0, int(y1 - padding_y_top))
                x2_padded = min(img_width, int(x2 + padding_x))
                y2_padded = min(sz, int(y2 + padding_y_bottom))  # Extra padding at bottom
                
                # Convert padded YOLO bbox (top-left origin) to dlib XML format
                # dlib XML format: top (from bottom), left (from left), width, height
                # YOLO: (x1, y1) top-left, (x2, y2) bottom-right in top-left origin
                # dlib: top is distance from bottom, so top = img_height - y2
                left = max(1, int(x1_padded))
                top = max(1, int(sz - y2_padded))  # Convert from top to bottom origin
                width = max(1, int(x2_padded - x1_padded))
                height = max(1, int(y2_padded - y1_padded))
            else:
                # Fallback: use bounding box around landmarks
                # Landmarks are in bottom-left origin
                min_x = coords[:, 0].min()
                max_x = coords[:, 0].max()
                min_y = coords[:, 1].min()  # bottom-left origin
                max_y = coords[:, 1].max()  # bottom-left origin
                
                padding = 20
                width = max_x - min_x + 2 * padding
                height = max_y - min_y + 2 * padding
                left = max(1, int(min_x - padding))
                top = max(1, int(sz - max_y - padding))  # Convert to bottom origin
            
            # Create image element
            image_e = ET.Element('image')
            image_e.set('file', path)
            
            # Create box element with YOLO bounding box
            box = ET.Element('box')
            box.set('top', str(top))
            box.set('left', str(left))
            box.set('width', str(width))
            box.set('height', str(height))
            
            # Add landmarks as parts (in bottom-left origin, as expected by dlib)
            for j in range(len(coords)):
                part = ET.Element('part')
                part.set('name', str(j))
                part.set('x', str(int(coords[j, 0])))
                part.set('y', str(int(coords[j, 1])))  # Already in bottom-left origin from TPS
                box.append(part)
            
            image_e.append(box)
            images_e.append(image_e)
    
    # Write XML file
    et = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
    with open(out_file, "w") as f:
        f.write(xmlstr)


def process_tps_to_full_images(tps_file, image_dir, yolo_model_path, output_base_dir='scale_full', 
                               train_ratio=0.8, seed=42, conf_threshold=0.25, target_class='ruler',
                               padding_x=50, padding_y_top=20, padding_y_bottom=200):
    """
    Process TPS file to create XML files for full images using YOLO bounding boxes.
    
    Args:
        tps_file: Path to consolidated TPS file
        image_dir: Directory containing original images
        yolo_model_path: Path to YOLO model (.pt file)
        output_base_dir: Base directory for output (will create train/ and test/ subdirectories)
        train_ratio: Ratio of images for training
        seed: Random seed for train/test split
        conf_threshold: YOLO confidence threshold
        target_class: Class to detect ('ruler', 'toe', or 'finger')
    """
    # Load YOLO model
    print(f"Loading YOLO model: {yolo_model_path}")
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model not found: {yolo_model_path}")
        sys.exit(1)
    
    yolo_model = YOLO(yolo_model_path)
    
    # Read TPS file
    print(f"Reading TPS file: {tps_file}")
    tps_data = read_tps(tps_file)
    
    num_specimens = len(tps_data['im'])
    print(f"Found {num_specimens} specimens in TPS file")
    
    # Create output directories (for copying images, though we'll use original paths in XML)
    train_dir = os.path.join(output_base_dir, 'train')
    test_dir = os.path.join(output_base_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Prepare data for train/test split
    all_items = list(range(num_specimens))
    random.seed(seed)
    random.shuffle(all_items)
    
    split_idx = int(train_ratio * len(all_items))
    train_indices = all_items[:split_idx]
    test_indices = all_items[split_idx:]
    
    print(f"Train set: {len(train_indices)} images")
    print(f"Test set: {len(test_indices)} images")
    
    # Process each specimen
    train_data = {'im': [], 'coords': []}
    test_data = {'im': [], 'coords': []}
    train_sizes = {}
    test_sizes = {}
    train_bboxes = {}
    test_bboxes = {}
    
    failed_images = []
    
    for idx in range(num_specimens):
        image_name = tps_data['im'][idx]
        landmarks = tps_data['coords'][idx]
        
        # Find original image
        image_path = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            potential_path = os.path.join(image_dir, image_name.replace('.jpg', ext))
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image for {image_name}, skipping...")
            failed_images.append(image_name)
            continue
        
        try:
            # Get image dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image {image_path}, skipping...")
                failed_images.append(image_name)
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Detect bounding box using YOLO
            bbox = detect_bounding_box_yolo(image_path, yolo_model, conf_threshold, target_class)
            
            # Determine output directory
            is_train = idx in train_indices
            output_dir = train_dir if is_train else test_dir
            split_name = 'train' if is_train else 'test'
            
            # Copy image to output directory (for consistency, though XML will reference original)
            output_image_name = os.path.basename(image_name)
            if not output_image_name.lower().endswith('.jpg'):
                output_image_name = os.path.splitext(output_image_name)[0] + '.jpg'
            
            output_path = os.path.join(output_dir, output_image_name)
            cv2.imwrite(output_path, img)
            
            # Store data for XML generation
            # Landmarks are already in bottom-left origin from TPS
            landmarks_for_xml = landmarks.copy()
            
            if is_train:
                train_data['im'].append(output_image_name)
                train_data['coords'].append(landmarks_for_xml)
                train_sizes[output_image_name] = [img_height, img_width]
                train_bboxes[output_image_name] = bbox
            else:
                test_data['im'].append(output_image_name)
                test_data['coords'].append(landmarks_for_xml)
                test_sizes[output_image_name] = [img_height, img_width]
                test_bboxes[output_image_name] = bbox
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{num_specimens} images...")
        
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            failed_images.append(image_name)
            continue
    
    print(f"\nSuccessfully processed {len(train_data['im']) + len(test_data['im'])} images")
    if failed_images:
        print(f"Failed to process {len(failed_images)} images: {failed_images[:10]}...")
    
    # Generate XML files with YOLO bounding boxes
    train_folder_rel = os.path.join(output_base_dir, 'train')
    test_folder_rel = os.path.join(output_base_dir, 'test')
    
    print("\nGenerating train.xml with YOLO bounding boxes (with padding)...")
    generate_dlib_xml_with_bbox(train_data, train_sizes, train_bboxes, 
                                folder=train_folder_rel, out_file='train.xml',
                                padding_x=padding_x, padding_y_top=padding_y_top, 
                                padding_y_bottom=padding_y_bottom)
    
    print("Generating test.xml with YOLO bounding boxes (with padding)...")
    generate_dlib_xml_with_bbox(test_data, test_sizes, test_bboxes, 
                                folder=test_folder_rel, out_file='test.xml',
                                padding_x=padding_x, padding_y_top=padding_y_top, 
                                padding_y_bottom=padding_y_bottom)
    
    print(f"\nDone! Images saved to:")
    print(f"  Train: {train_dir}")
    print(f"  Test: {test_dir}")
    print(f"  XML files: train.xml, test.xml")
    print(f"  Note: XML files use YOLO bounding boxes for dlib training")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess consolidated_scale.tps for full images using YOLO bounding boxes'
    )
    parser.add_argument(
        '-t', '--tps-file',
        type=str,
        default='consolidated_scale.tps',
        help='Path to consolidated TPS file (default: consolidated_scale.tps)'
    )
    parser.add_argument(
        '-i', '--image-dir',
        type=str,
        required=True,
        help='Directory containing original images'
    )
    parser.add_argument(
        '-y', '--yolo-model',
        type=str,
        default='yolo_bounding_box.pt',
        help='Path to YOLO model file (default: yolo_bounding_box.pt)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='scale_full',
        help='Base output directory (default: scale_full)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of images for training (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/test split (default: 42)'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='YOLO confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--target-class',
        type=str,
        default='ruler',
        choices=['ruler', 'toe', 'finger'],
        help='YOLO class to detect for bounding box (default: ruler)'
    )
    parser.add_argument(
        '--padding-x',
        type=int,
        default=50,
        help='Horizontal padding for bounding box (default: 50)'
    )
    parser.add_argument(
        '--padding-y-top',
        type=int,
        default=20,
        help='Top padding for bounding box (default: 20)'
    )
    parser.add_argument(
        '--padding-y-bottom',
        type=int,
        default=200,
        help='Bottom padding for bounding box (default: 200, for scale lines)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.tps_file):
        print(f"Error: TPS file not found: {args.tps_file}")
        sys.exit(1)
    
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Process
    process_tps_to_full_images(
        tps_file=args.tps_file,
        image_dir=args.image_dir,
        yolo_model_path=args.yolo_model,
        output_base_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        conf_threshold=args.conf_threshold,
        target_class=args.target_class,
        padding_x=args.padding_x,
        padding_y_top=args.padding_y_top,
        padding_y_bottom=args.padding_y_bottom
    )


if __name__ == '__main__':
    main()
