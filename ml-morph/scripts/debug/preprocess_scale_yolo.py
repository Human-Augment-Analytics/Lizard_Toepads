#!/usr/bin/env python3
"""
Preprocess consolidated_scale.tps to create cropped images using YOLO bounding box detection.

This script:
1. Uses YOLO model to detect bounding boxes (ruler/scale) in images
2. Crops images based on YOLO detections with padding
3. Adjusts landmark coordinates from consolidated_scale.tps to the cropped region
4. Saves cropped images to train/ and test/ directories
5. Generates train.xml and test.xml for dlib training
"""

import argparse
import os
import sys
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import utils

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


def crop_image_around_yolo_bbox(image_path, bbox, landmarks, padding_ratio=0.3, extend_bottom=False):
    """
    Crop image around YOLO bounding box with padding, and adjust landmarks.

    Args:
        image_path: Path to original image
        bbox: (x1, y1, x2, y2) bounding box from YOLO
        landmarks: numpy array of shape (N, 2) with landmark coordinates in TPS format (bottom-left origin)
        padding_ratio: Ratio of padding to add around bounding box (default 0.3 = 30%)
        extend_bottom: If True, extend crop to bottom of image (useful for scale/ruler to capture measurements)

    Returns:
        cropped_image: Cropped image as numpy array
        adjusted_landmarks: Landmarks adjusted to cropped coordinates
        crop_info: Dictionary with crop bounds
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    # YOLO bbox is in top-left origin (OpenCV format)
    x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
    
    # TPS coordinates use bottom-left origin (y=0 at bottom)
    # OpenCV uses top-left origin (y=0 at top)
    # Convert TPS coordinates to OpenCV coordinate system: y_flipped = img_height - y
    landmarks_converted = landmarks.copy()
    landmarks_converted[:, 1] = img_height - landmarks[:, 1]
    
    # Calculate padding around YOLO bbox
    bbox_width = x2_bbox - x1_bbox
    bbox_height = y2_bbox - y1_bbox
    
    # Use larger padding for narrow crops and ensure minimum crop sizes
    min_crop_width = 200  # Minimum crop width
    min_crop_height = 200  # Minimum crop height
    
    # Calculate proportional padding
    padding_x_prop = int(bbox_width * padding_ratio)
    padding_y_prop = int(bbox_height * padding_ratio)
    
    # Calculate absolute minimum padding needed
    padding_x_min = max(0, (min_crop_width - bbox_width) // 2)
    padding_y_min = max(0, (min_crop_height - bbox_height) // 2)
    
    # Use the larger of proportional or minimum padding
    padding_x = max(padding_x_prop, padding_x_min, 50)  # At least 50px padding
    padding_y = max(padding_y_prop, padding_y_min, 50)  # At least 50px padding
    
    # Calculate crop bounds with padding
    crop_x1 = max(0, x1_bbox - padding_x)
    crop_y1 = max(0, y1_bbox - padding_y)
    crop_x2 = min(img_width, x2_bbox + padding_x)

    # If extend_bottom is True, add extra padding below (not all the way to bottom)
    # This captures cm measurements below the scale without creating huge crops
    if extend_bottom:
        # Add 3x the normal padding below to capture measurements
        crop_y2 = min(img_height, y2_bbox + padding_y * 3)
    else:
        crop_y2 = min(img_height, y2_bbox + padding_y)
    
    # Ensure minimum crop size
    if (crop_x2 - crop_x1) < min_crop_width:
        center_x = (crop_x1 + crop_x2) // 2
        crop_x1 = max(0, center_x - min_crop_width // 2)
        crop_x2 = min(img_width, crop_x1 + min_crop_width)
        crop_x1 = max(0, crop_x2 - min_crop_width)
    
    if (crop_y2 - crop_y1) < min_crop_height:
        center_y = (crop_y1 + crop_y2) // 2
        crop_y1 = max(0, center_y - min_crop_height // 2)
        crop_y2 = min(img_height, crop_y1 + min_crop_height)
        crop_y1 = max(0, crop_y2 - min_crop_height)
    
    # Crop image
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Adjust landmarks to cropped coordinates
    adjusted_landmarks = landmarks_converted.copy()
    adjusted_landmarks[:, 0] = adjusted_landmarks[:, 0] - crop_x1
    adjusted_landmarks[:, 1] = adjusted_landmarks[:, 1] - crop_y1
    
    # Ensure landmarks are within cropped image bounds
    adjusted_landmarks[:, 0] = np.clip(adjusted_landmarks[:, 0], 0, crop_x2 - crop_x1 - 1)
    adjusted_landmarks[:, 1] = np.clip(adjusted_landmarks[:, 1], 0, crop_y2 - crop_y1 - 1)
    
    crop_info = {
        'x1': crop_x1,
        'y1': crop_y1,
        'x2': crop_x2,
        'y2': crop_y2,
        'width': crop_x2 - crop_x1,
        'height': crop_y2 - crop_y1
    }
    
    return cropped_img, adjusted_landmarks, crop_info


def process_tps_to_cropped_images(tps_file, image_dir, yolo_model_path, output_base_dir='cropped_scale',
                                   padding_ratio=0.3, train_ratio=0.8, seed=42, conf_threshold=0.25,
                                   target_class='ruler'):
    """
    Process TPS file to create cropped images using YOLO detection and prepare for training.

    Args:
        tps_file: Path to consolidated TPS file
        image_dir: Directory containing original images
        yolo_model_path: Path to YOLO model (.pt file)
        output_base_dir: Base directory for output (will create train/ and test/ subdirectories)
        padding_ratio: Padding ratio for cropping
        train_ratio: Ratio of images for training
        seed: Random seed for train/test split
        conf_threshold: YOLO confidence threshold
        target_class: Class to detect ('ruler', 'toe', or 'finger')
                      Note: For 'ruler' class, crop extends to bottom of image to capture measurements
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
    
    # Create output directories
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
            # Detect bounding box using YOLO
            bbox = detect_bounding_box_yolo(image_path, yolo_model, conf_threshold, target_class)
            
            # Crop image around YOLO bbox
            # For scale/ruler, extend to bottom of image to capture cm measurements
            extend_to_bottom = (target_class == 'ruler')
            cropped_img, adjusted_landmarks, crop_info = crop_image_around_yolo_bbox(
                image_path, bbox, landmarks, padding_ratio, extend_bottom=extend_to_bottom
            )
            
            # Determine output directory
            is_train = idx in train_indices
            output_dir = train_dir if is_train else test_dir
            split_name = 'train' if is_train else 'test'
            
            # Save cropped image
            output_image_name = os.path.basename(image_name)
            if not output_image_name.lower().endswith('.jpg'):
                output_image_name = os.path.splitext(output_image_name)[0] + '.jpg'
            
            output_path = os.path.join(output_dir, output_image_name)
            cv2.imwrite(output_path, cropped_img)
            
            # Store data for XML generation
            # Note: generate_dlib_xml's add_bbox_element expects bottom-left origin coordinates
            # and will flip the y-axis. Since we already flipped for cropping, we need to flip back
            # to bottom-left origin before passing to generate_dlib_xml
            cropped_img_height = cropped_img.shape[0]
            cropped_img_width = cropped_img.shape[1]
            landmarks_for_xml = adjusted_landmarks.copy()
            landmarks_for_xml[:, 1] = cropped_img_height - landmarks_for_xml[:, 1]

            if is_train:
                train_data['im'].append(output_image_name)
                train_data['coords'].append(landmarks_for_xml)
                # CRITICAL: Use actual cropped image dimensions, not crop_info
                train_sizes[output_image_name] = [cropped_img_height, cropped_img_width]
            else:
                test_data['im'].append(output_image_name)
                test_data['coords'].append(landmarks_for_xml)
                # CRITICAL: Use actual cropped image dimensions, not crop_info
                test_sizes[output_image_name] = [cropped_img_height, cropped_img_width]
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{num_specimens} images...")
        
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            failed_images.append(image_name)
            continue
    
    print(f"\nSuccessfully processed {len(train_data['im']) + len(test_data['im'])} images")
    if failed_images:
        print(f"Failed to process {len(failed_images)} images: {failed_images[:10]}...")
    
    # Generate XML files
    # Use relative paths - XML files will be in current directory, images in output_base_dir
    train_folder_rel = os.path.join(output_base_dir, 'train')
    test_folder_rel = os.path.join(output_base_dir, 'test')
    
    print("\nGenerating train.xml...")
    utils.generate_dlib_xml(train_data, train_sizes, folder=train_folder_rel, out_file='train.xml')

    print("Generating test.xml...")
    # Debug: Check dimensions for first test image
    test_img_name = test_data['im'][0] if test_data['im'] else None
    if test_img_name:
        idx = 0
        print(f"DEBUG: For {test_img_name}:")
        print(f"  Size in dictionary: {test_sizes.get(test_img_name)}")
        print(f"  Landmarks (bottom-left): {test_data['coords'][idx]}")
        test_img_path = os.path.join(test_folder_rel, test_img_name)
        if os.path.exists(test_img_path):
            import cv2
            debug_img = cv2.imread(test_img_path)
            print(f"  Actual image size: {debug_img.shape[1]}x{debug_img.shape[0]}")
    utils.generate_dlib_xml(test_data, test_sizes, folder=test_folder_rel, out_file='test.xml')
    
    print(f"\nDone! Cropped images saved to:")
    print(f"  Train: {train_dir}")
    print(f"  Test: {test_dir}")
    print(f"  XML files: train.xml, test.xml")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess consolidated_scale.tps to create cropped images using YOLO and XML files'
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
        default='cropped_scale',
        help='Base output directory for cropped images (default: cropped_scale)'
    )
    parser.add_argument(
        '-p', '--padding-ratio',
        type=float,
        default=0.3,
        help='Padding ratio around YOLO bounding box (default: 0.3 = 30%%)'
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
        help='YOLO class to detect for cropping (default: ruler)'
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
    process_tps_to_cropped_images(
        tps_file=args.tps_file,
        image_dir=args.image_dir,
        yolo_model_path=args.yolo_model,
        output_base_dir=args.output_dir,
        padding_ratio=args.padding_ratio,
        train_ratio=args.train_ratio,
        seed=args.seed,
        conf_threshold=args.conf_threshold,
        target_class=args.target_class
    )


if __name__ == '__main__':
    main()
