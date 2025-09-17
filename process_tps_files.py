import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import argparse
from tqdm import tqdm

# Increase maximum image size limit
Image.MAX_IMAGE_PIXELS = None  # Remove size limit - instead we prob want to resize image beforehand

def read_tps_file(tps_path):
    """Read landmarks from TPS file."""
    points = []
    with open(tps_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('LM'):
                continue
            if re.match(r'^-?\d', line):
                x, y = map(float, line.strip().split()[:2])
                points.append((x, y))
    return points

def create_yolo_label(landmarks, img_width, img_height, output_dir, image_name, class_id,
                      padding_ratio_x=0.2, padding_ratio_y=0.2):
    """Create YOLO label using tight bounding box with proportional padding."""
    valid_landmarks = landmarks[2:]  # Skip ruler
    converted_landmarks = [(x, img_height - y) for x, y in valid_landmarks]

    x_coords, y_coords = zip(*converted_landmarks)
    
    # Calculate tight bounding box
    landmark_width = max(x_coords) - min(x_coords)
    landmark_height = max(y_coords) - min(y_coords)
    
    # Use proportional padding based on landmark size
    padding_x = max(5, landmark_width * padding_ratio_x)  # Minimum 1px padding
    padding_y = max(5, landmark_height * padding_ratio_y)  # Minimum 1px padding
    
    x_min = max(0, min(x_coords) - padding_x)
    x_max = min(img_width, max(x_coords) + padding_x)
    y_min = max(0, min(y_coords) - padding_y)
    y_max = min(img_height, max(y_coords) + padding_y)

    box_width = (x_max - x_min) / img_width
    box_height = (y_max - y_min) / img_height
    center_x = (x_min + x_max) / (2 * img_width)
    center_y = (y_min + y_max) / (2 * img_height)

    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    label_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    mode = 'a' if os.path.exists(label_path) else 'w'
    with open(label_path, mode) as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

    return x_min, y_min, x_max, y_max

def create_ruler_label(ruler_points, img_width, img_height, output_dir, image_name, class_id,
                       padding_ratio_x=0.1, padding_ratio_y=0.1):
    """Create YOLO label and bounding box around ruler points with proportional padding."""
    converted = [(x, img_height - y) for x, y in ruler_points]
    x_coords, y_coords = zip(*converted)
    
    # Calculate ruler dimensions
    ruler_width = max(x_coords) - min(x_coords)
    ruler_height = max(y_coords) - min(y_coords)
    
    # Use proportional padding
    padding_x = max(5, ruler_width * padding_ratio_x)  # Minimum 5px padding
    padding_y = max(5, ruler_height * padding_ratio_y)  # Minimum 5px padding
    
    x_min = max(0, min(x_coords) - padding_x)
    x_max = min(img_width, max(x_coords) + padding_x)
    y_min = max(0, min(y_coords) - padding_y)
    y_max = min(img_height, max(y_coords) + padding_y)

    box_width = (x_max - x_min) / img_width
    box_height = (y_max - y_min) / img_height
    center_x = (x_min + x_max) / (2 * img_width)
    center_y = (y_min + y_max) / (2 * img_height)

    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    label_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    mode = 'a' if os.path.exists(label_path) else 'w'
    with open(label_path, mode) as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

    return x_min, y_min, x_max, y_max

def write_classes_txt(output_dir):
    """Write YOLO class names to classes.txt."""
    class_list = ['Finger', 'Toe', 'Ruler']
    with open(os.path.join(output_dir, 'labels/classes.txt'), 'w') as f:
        for c in class_list:
            f.write(f"{c}\n")

def resize_image_and_landmarks(img, landmarks_list, target_size=1024):
    """Resize image and scale landmark coordinates proportionally."""
    original_width, original_height = img.size
    
    # Calculate scaling to fit within target_size while maintaining aspect ratio
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Scale landmarks
    scaled_landmarks_list = []
    for landmarks in landmarks_list:
        scaled_landmarks = [(x * scale, y * scale) for x, y in landmarks]
        scaled_landmarks_list.append(scaled_landmarks)
    
    return resized_img, scaled_landmarks_list, new_width, new_height

def grayscale_image_if_needed(image):
    """
    Convert RGB image to grayscale if it's not already.
    YOLO-compatible grayscale is mode 'L'.
    """
    if image.mode == 'L':
        return image  # Already grayscale
    else:
        return image.convert('L')

def process_single_image(finger_tps_path, toe_tps_path, jpg_path, output_dir='output', point_size=10, add_points=False, target_size=1024):
    """Process an image with both finger and toe TPS files."""
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    if not os.path.exists(finger_tps_path):
        raise FileNotFoundError(f"Finger TPS file not found: {finger_tps_path}")
    if not os.path.exists(toe_tps_path):
        raise FileNotFoundError(f"Toe TPS file not found: {toe_tps_path}")
    if not os.path.exists(jpg_path):
        raise FileNotFoundError(f"JPG file not found: {jpg_path}")
    
    with Image.open(jpg_path) as img:
        # Resize image and scale landmarks
        finger_landmarks = read_tps_file(finger_tps_path)
        toe_landmarks = read_tps_file(toe_tps_path)
        
        resized_img, [finger_landmarks, toe_landmarks], img_width, img_height = resize_image_and_landmarks(
            img, [finger_landmarks, toe_landmarks], target_size
        )
        
        img_gray = grayscale_image_if_needed(resized_img)        # For YOLO input
        img_copy = img_gray.convert('RGB')

        draw = ImageDraw.Draw(img_copy)
        font = ImageFont.load_default()

        # Create YOLO format labels and get bounding boxes with tighter padding
        finger_box = create_yolo_label(
            finger_landmarks, img_width, img_height,
            output_dir, os.path.basename(jpg_path),
            class_id=0, padding_ratio_x=0.01, padding_ratio_y=0.01
        )

        toe_box = create_yolo_label(
            toe_landmarks, img_width, img_height,
            output_dir, os.path.basename(jpg_path),
            class_id=1, padding_ratio_x=0.01, padding_ratio_y=0.01
        )

        # Draw bounding boxes and labels with thinner lines
        draw.rectangle([finger_box[0], finger_box[1], finger_box[2], finger_box[3]], 
                       outline='blue', width=2)
        draw.text((finger_box[0], finger_box[1] - 10), "Finger", fill='blue', font=font)

        draw.rectangle([toe_box[0], toe_box[1], toe_box[2], toe_box[3]], 
                       outline='red', width=2)
        draw.text((toe_box[0], toe_box[1] - 10), "Toe", fill='red', font=font)

        # --- Shared Ruler Box ---
        ruler_points = finger_landmarks[:2]  # use only once
        ruler_box = create_ruler_label(
            ruler_points, img_width, img_height,
            output_dir, os.path.basename(jpg_path),
            class_id=2
        )

        draw.rectangle([ruler_box[0], ruler_box[1], ruler_box[2], ruler_box[3]],
                       outline='purple', width=2)
        draw.text((ruler_box[0], ruler_box[1] - 20), "Ruler", fill='purple', font=font)

        # TPS point visualization (optional based on --add-points flag)
        if add_points:
            # Draw finger landmarks
            for x, y in finger_landmarks:
                y = img_height - y
                draw.ellipse([(x - point_size - 2, y - point_size - 2),
                              (x + point_size + 2, y + point_size + 2)], fill='black')
                draw.ellipse([(x - point_size, y - point_size),
                              (x + point_size, y + point_size)], fill='lime')

            # Draw toe landmarks
            for x, y in toe_landmarks:
                y = img_height - y
                draw.ellipse([(x - point_size - 2, y - point_size - 2),
                              (x + point_size + 2, y + point_size + 2)], fill='black')
                draw.ellipse([(x - point_size, y - point_size),
                              (x + point_size, y + point_size)], fill='yellow')

        # Save processed image to images folder (without "marked_" prefix for YOLO training)
        output_path = os.path.join(output_dir, 'images', os.path.basename(jpg_path))
        img_gray.save(output_path)  # Save grayscale for YOLO

        # Optionally save visualization with bounding boxes
        if add_points or True:  # Always save visualization for checking
            vis_path = os.path.join(output_dir, f'marked_{os.path.basename(jpg_path)}')
            img_copy.save(vis_path)

        # Write classes.txt
        write_classes_txt(output_dir)


def batch_process_directory(image_dir, tps_dir, output_dir='output', point_size=10, add_points=False, target_size=1024):
    """
    Process all images in image_dir with corresponding TPS files in tps_dir.
    Matches by base filename (e.g., 001.jpg ↔ 001_finger.tps & 001_toe.tps)
    """
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

    # Count valid files first
    valid_files = []
    for jpg_file in jpg_files:
        base_name = os.path.splitext(jpg_file)[0]
        finger_tps = os.path.join(tps_dir, f"{base_name}_finger.TPS")
        toe_tps = os.path.join(tps_dir, f"{base_name}_toe.TPS")
        if os.path.exists(finger_tps) and os.path.exists(toe_tps):
            valid_files.append(jpg_file)

    print(f"Found {len(valid_files)} images with matching TPS files out of {len(jpg_files)} total images")

    # Process with progress bar
    for jpg_file in tqdm(valid_files, desc="Processing images", unit="image"):
        base_name = os.path.splitext(jpg_file)[0]
        finger_tps = os.path.join(tps_dir, f"{base_name}_finger.TPS")
        toe_tps = os.path.join(tps_dir, f"{base_name}_toe.TPS")
        jpg_path = os.path.join(image_dir, jpg_file)

        process_single_image(finger_tps, toe_tps, jpg_path, output_dir, point_size, add_points, target_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TPS and JPG image data.')
    
    # Batch mode
    parser.add_argument('--image-dir', help='Directory containing JPG images')
    parser.add_argument('--tps-dir', help='Directory containing TPS files')

    # Shared
    parser.add_argument('--output-dir', default='output', help='Output directory (default: output)')
    parser.add_argument('--point-size', type=int, default=10, help='Size of landmark points (default: 10)')
    parser.add_argument('--add-points', action='store_true', help='Add TPS landmark points to output images')
    parser.add_argument('--target-size', type=int, default=1024, help='Target size for resizing images (default: 1024)')

    args = parser.parse_args()

    batch_process_directory(
        image_dir=args.image_dir,
        tps_dir=args.tps_dir,
        output_dir=args.output_dir,
        point_size=args.point_size,
        add_points=args.add_points,
        target_size=args.target_size
    )