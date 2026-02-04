#!/usr/bin/env python3
"""
Preprocess consolidated_finger.tps to create cropped images and train/test XML files.

This script:
1. Reads consolidated_finger.tps
2. For each image, crops around the landmarks with padding
3. Adjusts landmark coordinates to the cropped region
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
import utils

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


def crop_image_around_landmarks(image_path, landmarks, padding_ratio=0.3):
    """
    Crop image around landmarks with padding.
    Automatically scales coordinates if they don't match image dimensions.
    
    Args:
        image_path: Path to original image
        landmarks: numpy array of shape (N, 2) with landmark coordinates
        padding_ratio: Ratio of padding to add around bounding box (default 0.3 = 30%)
    
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
    
    # TPS coordinates use bottom-left origin (y=0 at bottom)
    # OpenCV uses top-left origin (y=0 at top)
    # Convert TPS coordinates to OpenCV coordinate system: y_flipped = img_height - y
    landmarks_converted = landmarks.copy()
    landmarks_converted[:, 1] = img_height - landmarks[:, 1]
    
    # Calculate bounding box around landmarks (using converted coordinates)
    min_x = np.min(landmarks_converted[:, 0])
    max_x = np.max(landmarks_converted[:, 0])
    min_y = np.min(landmarks_converted[:, 1])
    max_y = np.max(landmarks_converted[:, 1])
    
    # Use converted landmarks for all subsequent operations
    landmarks = landmarks_converted
    
    # Check if coordinates need scaling (if they exceed image bounds)
    # Image coordinates are 0-indexed, so valid range is [0, width-1] and [0, height-1]
    needs_scaling = (max_x > img_width - 1 or max_y > img_height - 1 or min_x < 0 or min_y < 0)
    
    # Debug: Check if this is a problematic image
    debug_image = '1359.jpg' in image_path or '1026.jpg' in image_path
    if debug_image:
        print(f"\nDEBUG {Path(image_path).name}:")
        print(f"  Image: {img_width}×{img_height}")
        print(f"  Landmarks after y-flip: X=[{min_x:.0f}, {max_x:.0f}], Y=[{min_y:.0f}, {max_y:.0f}]")
        print(f"  Needs scaling: {needs_scaling}")
        print(f"    max_x ({max_x}) > img_width-1 ({img_width-1}): {max_x > img_width - 1}")
        print(f"    max_y ({max_y}) > img_height-1 ({img_height-1}): {max_y > img_height - 1}")
    
    if needs_scaling:
        # Calculate scale factors based on absolute coordinate values vs image dimensions
        # This handles cases where coordinates are in a different coordinate system
        # Scale based on the maximum coordinate value relative to image size
        
        # Find the maximum coordinate values (could be negative or exceed image bounds)
        coord_max_x = np.max(landmarks[:, 0])
        coord_max_y = np.max(landmarks[:, 1])
        coord_min_x = np.min(landmarks[:, 0])
        coord_min_y = np.min(landmarks[:, 1])
        
        # Calculate the range of coordinates
        coord_range_x = coord_max_x - coord_min_x
        coord_range_y = coord_max_y - coord_min_y
        
        # Calculate scale factors to map coordinates to image space
        # We need to scale so that the coordinate range fits within the image
        # Use 95% of image dimensions to leave some margin
        target_width = img_width * 0.95
        target_height = img_height * 0.95
        
        # Calculate scale needed to fit range in each dimension
        if coord_range_x > 0:
            scale_x = target_width / coord_range_x
        else:
            scale_x = 1.0
        
        if coord_range_y > 0:
            scale_y = target_height / coord_range_y
        else:
            scale_y = 1.0
        
        # Use the smaller scale to preserve aspect ratio
        scale_factor = min(scale_x, scale_y)
        
        # Calculate the scaled range
        scaled_range_x = coord_range_x * scale_factor
        scaled_range_y = coord_range_y * scale_factor
        
        # Center the scaled landmarks in the image
        offset_x = (img_width - scaled_range_x) / 2 - coord_min_x * scale_factor
        offset_y = (img_height - scaled_range_y) / 2 - coord_min_y * scale_factor
        
        # Apply scaling and offset to landmarks
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, 0] = scaled_landmarks[:, 0] * scale_factor + offset_x
        scaled_landmarks[:, 1] = scaled_landmarks[:, 1] * scale_factor + offset_y
        
        # Clamp to image bounds
        scaled_landmarks[:, 0] = np.clip(scaled_landmarks[:, 0], 0, img_width - 1)
        scaled_landmarks[:, 1] = np.clip(scaled_landmarks[:, 1], 0, img_height - 1)
        
        landmarks = scaled_landmarks
        
        if debug_image:
            print(f"  Applied scaling: factor={scale_factor:.4f}")
            print(f"  Scaled landmarks: X=[{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}], Y=[{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")
        
        # Recalculate bounding box with scaled coordinates
        min_x = np.min(landmarks[:, 0])
        max_x = np.max(landmarks[:, 0])
        min_y = np.min(landmarks[:, 1])
        max_y = np.max(landmarks[:, 1])
    
    # Convert to integers for indexing
    min_x = int(min_x)
    max_x = int(max_x)
    min_y = int(min_y)
    max_y = int(max_y)
    
    # Calculate padding
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    # Use larger padding for narrow crops and ensure minimum crop sizes
    # For very narrow crops, use absolute minimum padding
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
    crop_x1 = max(0, min_x - padding_x)
    crop_y1 = max(0, min_y - padding_y)
    crop_x2 = min(img_width, max_x + padding_x)
    crop_y2 = min(img_height, max_y + padding_y)
    
    # Ensure minimum crop size (use larger minimums)
    min_crop_width = 200
    min_crop_height = 200
    
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
    
    if debug_image:
        print(f"  Final crop: ({crop_x1}, {crop_y1}, {crop_x2-crop_x1}, {crop_y2-crop_y1})")
        print(f"  Cropped image size: {cropped_img.shape[1]}×{cropped_img.shape[0]}")
    
    # Adjust landmarks to cropped coordinates
    adjusted_landmarks = landmarks.copy()
    adjusted_landmarks[:, 0] = adjusted_landmarks[:, 0] - crop_x1
    adjusted_landmarks[:, 1] = adjusted_landmarks[:, 1] - crop_y1
    
    if debug_image:
        print(f"  Adjusted landmarks in crop: X=[{adjusted_landmarks[:, 0].min():.1f}, {adjusted_landmarks[:, 0].max():.1f}], Y=[{adjusted_landmarks[:, 1].min():.1f}, {adjusted_landmarks[:, 1].max():.1f}]")
    
    # Ensure landmarks are within cropped image bounds
    adjusted_landmarks[:, 0] = np.clip(adjusted_landmarks[:, 0], 0, crop_x2 - crop_x1 - 1)
    adjusted_landmarks[:, 1] = np.clip(adjusted_landmarks[:, 1], 0, crop_y2 - crop_y1 - 1)
    
    crop_info = {
        'x1': crop_x1,
        'y1': crop_y1,
        'x2': crop_x2,
        'y2': crop_y2,
        'width': crop_x2 - crop_x1,
        'height': crop_y2 - crop_y1,
        'scaled': needs_scaling
    }
    
    return cropped_img, adjusted_landmarks, crop_info


def process_tps_to_cropped_images(tps_file, image_dir, output_base_dir='cropped_finger', 
                                   padding_ratio=0.3, train_ratio=0.8, seed=42):
    """
    Process TPS file to create cropped images and prepare for training.
    
    Args:
        tps_file: Path to consolidated TPS file
        image_dir: Directory containing original images
        output_base_dir: Base directory for output (will create train/ and test/ subdirectories)
        padding_ratio: Padding ratio for cropping
        train_ratio: Ratio of images for training
        seed: Random seed for train/test split
    """
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
            continue
        
        try:
            # Crop image around landmarks
            cropped_img, adjusted_landmarks, crop_info = crop_image_around_landmarks(
                image_path, landmarks, padding_ratio
            )
            
            if crop_info.get('scaled', False):
                if (idx + 1) % 20 == 0 or idx < 5:
                    print(f"  Note: Scaled coordinates for {image_name} (coords exceeded image bounds)")
            
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
            landmarks_for_xml = adjusted_landmarks.copy()
            landmarks_for_xml[:, 1] = cropped_img_height - landmarks_for_xml[:, 1]
            
            if is_train:
                train_data['im'].append(output_image_name)
                train_data['coords'].append(landmarks_for_xml)
                train_sizes[output_image_name] = [crop_info['height'], crop_info['width']]
            else:
                test_data['im'].append(output_image_name)
                test_data['coords'].append(landmarks_for_xml)
                test_sizes[output_image_name] = [crop_info['height'], crop_info['width']]
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{num_specimens} images...")
        
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(train_data['im']) + len(test_data['im'])} images")
    
    # Generate XML files
    # Use relative paths - XML files will be in current directory, images in output_base_dir
    train_folder_rel = os.path.join(output_base_dir, 'train')
    test_folder_rel = os.path.join(output_base_dir, 'test')
    
    print("\nGenerating train.xml...")
    utils.generate_dlib_xml(train_data, train_sizes, folder=train_folder_rel, out_file='train.xml')
    
    print("Generating test.xml...")
    utils.generate_dlib_xml(test_data, test_sizes, folder=test_folder_rel, out_file='test.xml')
    
    print(f"\nDone! Cropped images saved to:")
    print(f"  Train: {train_dir}")
    print(f"  Test: {test_dir}")
    print(f"  XML files: train.xml, test.xml")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess consolidated_finger.tps to create cropped images and XML files'
    )
    parser.add_argument(
        '-t', '--tps-file',
        type=str,
        default='consolidated_finger.tps',
        help='Path to consolidated TPS file (default: consolidated_finger.tps)'
    )
    parser.add_argument(
        '-i', '--image-dir',
        type=str,
        required=True,
        help='Directory containing original images'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='cropped_finger',
        help='Base output directory for cropped images (default: cropped_finger)'
    )
    parser.add_argument(
        '-p', '--padding-ratio',
        type=float,
        default=0.3,
        help='Padding ratio around landmarks (default: 0.3 = 30%%)'
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
        output_base_dir=args.output_dir,
        padding_ratio=args.padding_ratio,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

