#!/usr/bin/env python3
"""
Helper functions for preprocessing images for inference with the cropped toe model.

The model was trained with 30% padding around landmarks. This script provides
functions to match that preprocessing exactly.
"""

import cv2
import numpy as np


def crop_image_for_inference(image_path, landmarks_tps, padding_ratio=0.3, min_padding=50, min_crop_size=200):
    """
    Crop image around landmarks with the same padding used during training.
    
    Args:
        image_path: Path to original image
        landmarks_tps: numpy array of shape (N, 2) with landmark coordinates in TPS format (bottom-left origin)
        padding_ratio: Padding ratio around bounding box (default 0.3 = 30%, matching training)
        min_padding: Minimum padding in pixels (default 50, matching training)
        min_crop_size: Minimum crop size in pixels (default 200, matching training)
    
    Returns:
        cropped_image: Cropped image as numpy array
        crop_info: Dictionary with crop bounds and metadata
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    # TPS coordinates use bottom-left origin (y=0 at bottom)
    # OpenCV uses top-left origin (y=0 at top)
    # Convert TPS coordinates to OpenCV coordinate system
    landmarks_opencv = landmarks_tps.copy()
    landmarks_opencv[:, 1] = img_height - landmarks_tps[:, 1]
    
    # Calculate bounding box around landmarks
    min_x = np.min(landmarks_opencv[:, 0])
    max_x = np.max(landmarks_opencv[:, 0])
    min_y = np.min(landmarks_opencv[:, 1])
    max_y = np.max(landmarks_opencv[:, 1])
    
    # Calculate padding
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    # Use larger padding for narrow crops and ensure minimum crop sizes
    padding_x_prop = int(bbox_width * padding_ratio)
    padding_y_prop = int(bbox_height * padding_ratio)
    
    # Calculate absolute minimum padding needed
    padding_x_min = max(0, (min_crop_size - bbox_width) // 2)
    padding_y_min = max(0, (min_crop_size - bbox_height) // 2)
    
    # Use the larger of proportional or minimum padding
    padding_x = max(padding_x_prop, padding_x_min, min_padding)
    padding_y = max(padding_y_prop, padding_y_min, min_padding)
    
    # Calculate crop bounds with padding
    crop_x1 = max(0, int(min_x - padding_x))
    crop_y1 = max(0, int(min_y - padding_y))
    crop_x2 = min(img_width, int(max_x + padding_x))
    crop_y2 = min(img_height, int(max_y + padding_y))
    
    # Ensure minimum crop size
    if (crop_x2 - crop_x1) < min_crop_size:
        center_x = (crop_x1 + crop_x2) // 2
        crop_x1 = max(0, center_x - min_crop_size // 2)
        crop_x2 = min(img_width, crop_x1 + min_crop_size)
        crop_x1 = max(0, crop_x2 - min_crop_size)
    
    if (crop_y2 - crop_y1) < min_crop_size:
        center_y = (crop_y1 + crop_y2) // 2
        crop_y1 = max(0, center_y - min_crop_size // 2)
        crop_y2 = min(img_height, crop_y1 + min_crop_size)
        crop_y1 = max(0, crop_y2 - min_crop_size)
    
    # Crop image
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    crop_info = {
        'x1': crop_x1,
        'y1': crop_y1,
        'x2': crop_x2,
        'y2': crop_y2,
        'width': crop_x2 - crop_x1,
        'height': crop_y2 - crop_y1,
        'original_width': img_width,
        'original_height': img_height
    }
    
    return cropped_img, crop_info


def convert_predicted_to_original(predicted_landmarks, crop_info):
    """
    Convert predicted landmarks (in bottom-left origin, relative to cropped image)
    back to original image coordinates (in TPS format, bottom-left origin).
    
    Args:
        predicted_landmarks: numpy array of shape (N, 2) with landmarks in dlib format
                            (bottom-left origin, relative to cropped image)
        crop_info: Dictionary with crop bounds from crop_image_for_inference
    
    Returns:
        original_landmarks_tps: numpy array of shape (N, 2) with landmarks in TPS format
                               (bottom-left origin, relative to original image)
    """
    cropped_h = crop_info['height']
    cropped_w = crop_info['width']
    orig_h = crop_info['original_height']
    orig_w = crop_info['original_width']
    
    # Convert from dlib format (bottom-left origin, cropped image) to OpenCV (top-left, cropped)
    landmarks_opencv_cropped = predicted_landmarks.copy()
    landmarks_opencv_cropped[:, 1] = cropped_h - predicted_landmarks[:, 1]
    
    # Map to original image coordinates (top-left origin)
    landmarks_opencv_original = landmarks_opencv_cropped.copy()
    landmarks_opencv_original[:, 0] += crop_info['x1']
    landmarks_opencv_original[:, 1] += crop_info['y1']
    
    # Convert back to TPS format (bottom-left origin, original image)
    original_landmarks_tps = landmarks_opencv_original.copy()
    original_landmarks_tps[:, 1] = orig_h - landmarks_opencv_original[:, 1]
    
    return original_landmarks_tps


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess image for inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--landmarks', type=str, help='Path to landmarks file (optional, for testing)')
    parser.add_argument('--output', type=str, default='cropped_for_inference.jpg', help='Output cropped image')
    
    args = parser.parse_args()
    
    # Example: if you have landmarks
    if args.landmarks:
        # Load landmarks (example - adjust based on your format)
        landmarks = np.loadtxt(args.landmarks)
        cropped_img, crop_info = crop_image_for_inference(args.image, landmarks)
        cv2.imwrite(args.output, cropped_img)
        print(f"Cropped image saved to {args.output}")
        print(f"Crop info: {crop_info}")
    else:
        print("Usage: Provide --landmarks to crop image for inference")
        print("The cropped image will have 30% padding around landmarks (matching training)")


