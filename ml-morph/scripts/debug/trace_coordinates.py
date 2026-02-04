#!/usr/bin/env python3
"""Trace coordinate transformations step by step."""

import cv2
import numpy as np

# Original image
img_orig = cv2.imread('/Users/leyangloh/dev/Lizard_Toepads/data/miami_fall_24_jpgs/1017.jpg')
orig_h, orig_w = img_orig.shape[:2]

# Cropped image
img_crop = cv2.imread('cropped_scale_padded/test/1017.jpg')
crop_h, crop_w = img_crop.shape[:2]

print(f"Original image: {orig_w}x{orig_h}")
print(f"Cropped image: {crop_w}x{crop_h}")
print()

# TPS landmarks (bottom-left origin in ORIGINAL image)
tps_landmarks = np.array([[38.0, 628.0], [672.0, 625.0]])
print(f"Step 1: TPS landmarks (bottom-left origin in original):")
print(f"  {tps_landmarks}")
print()

# Convert to top-left (OpenCV) for ORIGINAL image
landmarks_opencv = tps_landmarks.copy()
landmarks_opencv[:, 1] = orig_h - tps_landmarks[:, 1]
print(f"Step 2: Converted to top-left in original image:")
print(f"  {landmarks_opencv}")
print(f"  Point 0 at y={landmarks_opencv[0,1]:.0f} = {landmarks_opencv[0,1]/orig_h*100:.1f}% from top")
print()

# YOLO detected bbox (from earlier analysis)
yolo_bbox = (3, 5359, 739, 5437)
x1_bbox, y1_bbox, x2_bbox, y2_bbox = yolo_bbox

# Calculate crop with 3x padding
bbox_width = x2_bbox - x1_bbox
bbox_height = y2_bbox - y1_bbox
padding_ratio = 0.3
padding_x = max(int(bbox_width * padding_ratio), 50)
padding_y = max(int(bbox_height * padding_ratio), 50)

crop_x1 = max(0, x1_bbox - padding_x)
crop_y1 = max(0, y1_bbox - padding_y)
crop_x2 = min(orig_w, x2_bbox + padding_x)
crop_y2 = min(orig_h, y2_bbox + padding_y * 3)  # 3x padding

print(f"Step 3: YOLO bbox and crop region:")
print(f"  YOLO bbox: {yolo_bbox}")
print(f"  Padding: x={padding_x}, y={padding_y}")
print(f"  Crop region: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
print(f"  Crop size: {crop_x2-crop_x1}x{crop_y2-crop_y1}")
print()

# Adjust landmarks to cropped coordinates (still top-left)
adjusted_landmarks = landmarks_opencv.copy()
adjusted_landmarks[:, 0] = adjusted_landmarks[:, 0] - crop_x1
adjusted_landmarks[:, 1] = adjusted_landmarks[:, 1] - crop_y1

print(f"Step 4: Landmarks adjusted to cropped image (top-left):")
print(f"  {adjusted_landmarks}")
print(f"  Point 0 at y={adjusted_landmarks[0,1]:.0f} = {adjusted_landmarks[0,1]/(crop_y2-crop_y1)*100:.1f}% from top of crop")
print()

# This is what preprocessing does: convert to bottom-left for XML
cropped_img_height = crop_y2 - crop_y1
landmarks_for_xml = adjusted_landmarks.copy()
landmarks_for_xml[:, 1] = cropped_img_height - adjusted_landmarks[:, 1]

print(f"Step 5: Convert to bottom-left for passing to utils.generate_dlib_xml:")
print(f"  {landmarks_for_xml}")
print()

# This is what utils.add_part_element does
xml_y = cropped_img_height - landmarks_for_xml[:, 1]

print(f"Step 6: utils.add_part_element flips back (sz - bbox[1]):")
print(f"  XML y values: {xml_y}")
print(f"  This should equal step 4 (top-left coords)")
print()

print("="*60)
print("ACTUAL XML VALUES:")
print("="*60)
print(f"XML has: x=({38}, {672}), y=({96}, {99})")
print(f"Expected: y ≈ {adjusted_landmarks[0,1]:.0f}")
print()

print("THE BUG:")
if abs(xml_y[0] - 96) > 10:
    print(f"  Expected XML y ≈ {xml_y[0]:.0f}, but got y=96")
    print(f"  Difference: {xml_y[0] - 96:.0f} pixels")
    print()
    print("  This suggests the cropped image height used in XML generation")
    print(f"  was {96 + landmarks_for_xml[0,1]:.0f} instead of {cropped_img_height}")
