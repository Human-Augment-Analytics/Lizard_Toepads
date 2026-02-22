import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import argparse
from tqdm import tqdm
import yaml
from pathlib import Path
from multiprocessing import Pool, cpu_count
import functools

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

def compute_pca_angle(points):
    """Compute principal-axis angle (radians) of 2D points via PCA.
    Returns angle of the principal eigenvector relative to the positive x-axis."""
    pts = np.array(points, dtype=float)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    return np.arctan2(principal[1], principal[0])


def rotate_corners(cx, cy, half_w, half_h, angle_rad):
    """Rotate 4 axis-aligned bbox corners around center by angle_rad.
    Returns list of (x, y) tuples for the 4 corners."""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Corners relative to center: TL, TR, BR, BL
    offsets = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    corners = []
    for dx, dy in offsets:
        rx = cx + dx * cos_a - dy * sin_a
        ry = cy + dx * sin_a + dy * cos_a
        corners.append((rx, ry))
    return corners


def create_yolo_label(landmarks, img_width, img_height, output_dir, image_name, class_id,
                      padding_ratio_x=0.2, padding_ratio_y=0.2, obb=False):
    """Create YOLO label using tight bounding box with proportional padding.
    When obb=True, writes oriented bounding box (4-corner polygon) format."""
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

    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    label_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    mode = 'a' if os.path.exists(label_path) else 'w'
    with open(label_path, mode) as f:
        if obb:
            angle_rad = compute_pca_angle(converted_landmarks)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            half_w = (x_max - x_min) / 2
            half_h = (y_max - y_min) / 2
            corners = rotate_corners(cx, cy, half_w, half_h, angle_rad)
            # Normalize and clamp to [0, 1]
            norm_corners = []
            for px, py in corners:
                norm_corners.append(max(0.0, min(1.0, px / img_width)))
                norm_corners.append(max(0.0, min(1.0, py / img_height)))
            coords_str = " ".join(f"{c:.6f}" for c in norm_corners)
            f.write(f"{class_id} {coords_str}\n")
        else:
            box_width = (x_max - x_min) / img_width
            box_height = (y_max - y_min) / img_height
            center_x = (x_min + x_max) / (2 * img_width)
            center_y = (y_min + y_max) / (2 * img_height)
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

    return x_min, y_min, x_max, y_max

def create_ruler_label(ruler_points, img_width, img_height, output_dir, image_name, class_id,
                       padding_ratio_x=0.1, padding_ratio_y=0.1, obb=False):
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

    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    label_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    mode = 'a' if os.path.exists(label_path) else 'w'
    with open(label_path, mode) as f:
        if obb:
            # Ruler angle from the line between its 2 endpoints
            angle_rad = np.arctan2(converted[1][1] - converted[0][1],
                                   converted[1][0] - converted[0][0])
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            half_w = (x_max - x_min) / 2
            half_h = (y_max - y_min) / 2
            corners = rotate_corners(cx, cy, half_w, half_h, angle_rad)
            norm_corners = []
            for px, py in corners:
                norm_corners.append(max(0.0, min(1.0, px / img_width)))
                norm_corners.append(max(0.0, min(1.0, py / img_height)))
            coords_str = " ".join(f"{c:.6f}" for c in norm_corners)
            f.write(f"{class_id} {coords_str}\n")
        else:
            box_width = (x_max - x_min) / img_width
            box_height = (y_max - y_min) / img_height
            center_x = (x_min + x_max) / (2 * img_width)
            center_y = (y_min + y_max) / (2 * img_height)
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

def process_single_image(finger_tps_path, toe_tps_path, jpg_path, output_dir='output', point_size=10, add_points=False, target_size=1024, obb=False):
    """Process an image with both finger and toe TPS files."""
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
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
            class_id=0, padding_ratio_x=0.01, padding_ratio_y=0.01, obb=obb
        )

        toe_box = create_yolo_label(
            toe_landmarks, img_width, img_height,
            output_dir, os.path.basename(jpg_path),
            class_id=1, padding_ratio_x=0.01, padding_ratio_y=0.01, obb=obb
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
            class_id=2, obb=obb
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

        # Always save visualization with bounding boxes for verification
        vis_path = os.path.join(output_dir, 'visualizations', f'marked_{os.path.basename(jpg_path)}')
        img_copy.save(vis_path)

        # Write classes.txt
        write_classes_txt(output_dir)


def process_single_image_wrapper(args):
    """Wrapper function for multiprocessing - unpacks arguments"""
    return process_single_image(*args)


def batch_process_directory(image_dir, tps_dir, output_dir='data/processed', point_size=10, add_points=False, target_size=1024, num_workers=None, obb=False):
    """
    Process all images in image_dir with corresponding TPS files in tps_dir.
    Matches by base filename (e.g., 001.jpg ↔ 001_finger.tps & 001_toe.tps)

    Args:
        num_workers: Number of parallel workers. If None, uses cpu_count()
    """
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

    # Count valid files first and prepare arguments
    valid_args = []
    for jpg_file in jpg_files:
        base_name = os.path.splitext(jpg_file)[0]
        finger_tps = os.path.join(tps_dir, f"{base_name}_finger.TPS")
        toe_tps = os.path.join(tps_dir, f"{base_name}_toe.TPS")
        jpg_path = os.path.join(image_dir, jpg_file)

        if os.path.exists(finger_tps) and os.path.exists(toe_tps):
            valid_args.append((finger_tps, toe_tps, jpg_path, output_dir, point_size, add_points, target_size, obb))

    print(f"Found {len(valid_args)} images with matching TPS files out of {len(jpg_files)} total images")

    if num_workers is None:
        num_workers = min(cpu_count(), len(valid_args))

    print(f"Using {num_workers} parallel workers")

    # Process with multiprocessing and progress bar
    if num_workers == 1:
        # Single-threaded for debugging or small datasets
        for args in tqdm(valid_args, desc="Processing images", unit="image"):
            process_single_image_wrapper(args)
    else:
        # Multi-threaded processing
        with Pool(num_workers) as pool:
            # Use imap for progress tracking
            list(tqdm(
                pool.imap(process_single_image_wrapper, valid_args),
                total=len(valid_args),
                desc="Processing images",
                unit="image"
            ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process TPS and JPG image data.')

    parser.add_argument('--config', default='configs/H1.yaml', help='Path to YAML config file (default: configs/H1.yaml). CLI flags override config.')

    # Batch mode
    parser.add_argument('--image-dir', help='Directory containing JPG images')
    parser.add_argument('--tps-dir', help='Directory containing TPS files')

    # Shared
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--point-size', type=int, help='Size of landmark points')
    parser.add_argument('--add-points', action='store_true', help='Add TPS landmark points to output images')
    parser.add_argument('--target-size', type=int, help='Target size for resizing images')
    parser.add_argument('--num-workers', type=int, help='Number of parallel workers (default: auto-detect CPU cores)')
    parser.add_argument('--obb', action='store_true', help='Generate oriented bounding box (OBB) labels instead of axis-aligned')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load YAML config if provided
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    def get_opt(name: str, default):
        # CLI has highest precedence
        value = getattr(args, name.replace('-', '_'), None)
        if value not in (None, ""):
            return value
        # then config YAML
        # support nested under 'preprocessing' section
        if name in cfg and cfg[name] not in (None, ""):
            return cfg[name]
        if isinstance(cfg.get('preprocessing'), dict) and cfg['preprocessing'].get(name) not in (None, ""):
            return cfg['preprocessing'][name]
        return default

    image_dir = get_opt('image-dir', None)
    tps_dir = get_opt('tps-dir', None)
    output_dir = get_opt('output-dir', 'data/processed')
    point_size = int(get_opt('point-size', 10))
    add_points = bool(get_opt('add-points', False))
    target_size = int(get_opt('target-size', 1024))
    num_workers = args.num_workers or get_opt('num-workers', None)
    # args.obb from store_true defaults to False, so check it explicitly
    # to avoid shadowing the config value in get_opt
    if args.obb:
        obb = True
    else:
        obb_cfg = (cfg.get('preprocessing') or {}).get('obb', cfg.get('obb', False))
        obb = bool(obb_cfg)

    print(f"Using output_dir: {output_dir}")
    print(f"Using target_size: {target_size}")
    if obb:
        print("OBB mode: generating oriented bounding box labels")
    if num_workers is not None:
        num_workers = int(num_workers)

    if not image_dir or not tps_dir:
        raise ValueError("image-dir and tps-dir are required. Provide via --image-dir/--tps-dir or config file.")

    batch_process_directory(
        image_dir=image_dir,
        tps_dir=tps_dir,
        output_dir=output_dir,
        point_size=point_size,
        add_points=add_points,
        target_size=target_size,
        num_workers=num_workers,
        obb=obb
    )