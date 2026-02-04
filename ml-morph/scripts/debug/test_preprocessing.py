#!/usr/bin/env python3
"""Test preprocessing coordinate transformations."""

import cv2
import numpy as np

# Simulate what happens for image 1017.jpg
image_path = '/Users/leyangloh/dev/Lizard_Toepads/data/miami_fall_24_jpgs/1017.jpg'
img = cv2.imread(image_path)
img_height, img_width = img.shape[:2]

print(f"Original image: {img_width}x{img_height}")

# TPS landmarks (bottom-left origin)
tps_landmarks = np.array([[38.0, 628.0], [672.0, 625.0]])
print(f"\nTPS landmarks (bottom-left): \n{tps_landmarks}")

# Step 1: Convert to top-left (OpenCV) for processing
landmarks_opencv = tps_landmarks.copy()
landmarks_opencv[:, 1] = img_height - tps_landmarks[:, 1]
print(f"\nConverted to top-left (OpenCV):\n{landmarks_opencv}")
print(f"  These should be near bottom: y = {landmarks_opencv[0,1]:.0f} ({landmarks_opencv[0,1]/img_height*100:.1f}% from top)")

# Step 2: Simulate cropping with extended bottom
# For this test, let's say YOLO detected bbox and we crop with extend_bottom=True
# The crop extended all the way to bottom of image
# Let's say crop was from (0, 5300) to (1000, 6022) - just an example

# After cropping, image is 959x724
cropped_height, cropped_width = 724, 959
crop_y1 = 5298  # Top of crop in original image (top-left coords)

# Adjust landmarks to cropped coordinates (still top-left)
adjusted_landmarks = landmarks_opencv.copy()
adjusted_landmarks[:, 1] = adjusted_landmarks[:, 1] - crop_y1
print(f"\nAfter cropping (adjusted to crop, still top-left):\n{adjusted_landmarks}")
print(f"  Relative to cropped image: y = {adjusted_landmarks[0,1]:.0f} ({adjusted_landmarks[0,1]/cropped_height*100:.1f}% from top of crop)")

# Step 3: Convert to bottom-left for XML (as preprocessing does)
landmarks_for_xml = adjusted_landmarks.copy()
landmarks_for_xml[:, 1] = cropped_height - adjusted_landmarks[:, 1]
print(f"\nConverted to bottom-left for XML:\n{landmarks_for_xml}")

# Step 4: What add_part_element does (should convert back to top-left)
xml_y = cropped_height - landmarks_for_xml[:, 1]
print(f"\nAfter add_part_element flip (should be top-left for dlib):\n  y values: {xml_y}")
print(f"  Expected y ≈ {adjusted_landmarks[0,1]:.0f} (same as step 2)")

print("\n" + "="*60)
print("COMPARISON WITH ACTUAL XML:")
print("="*60)
print(f"Actual XML has: y=96")
print(f"Expected from correct processing: y ≈ {adjusted_landmarks[0,1]:.0f}")
print(f"\nIf XML has y=96, working backwards:")
print(f"  Before add_part_element flip: bottom-left y = {cropped_height - 96}")
print(f"  Before xml conversion: top-left y = {96}")
print(f"  This means scale is at 13% from top - WRONG! Should be near bottom")
