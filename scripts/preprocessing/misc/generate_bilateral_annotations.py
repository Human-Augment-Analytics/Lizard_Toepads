"""
Semi-Automated Bilateral Annotation Generation (Approach 4)

Uses the H2_bilateral_preprocessed model with bilateral inference to automatically
generate complete bilateral annotations (2 fingers + 2 toes + 2 rulers per image).

This approach bridges the gap between Approach 3 (augmented training) and complete
bilateral detection by creating a dataset where each image has annotations for both
upper and lower toepads.

Usage:
    # Full model-based annotation (both sides from model)
    python scripts/preprocessing/generate_bilateral_annotations.py \
        --config configs/H2_bilateral_preprocessed.yaml \
        --source-images data/processed/images \
        --output-dir data/bilateral_annotated \
        --verify

    # Hybrid annotation (TPS for lower, model for upper)
    python scripts/preprocessing/generate_bilateral_annotations.py \
        --config configs/H2_bilateral_preprocessed.yaml \
        --source-images data/processed/images \
        --tps-dir /storage/ice-shared/cs8903onl/tps_files \
        --output-dir data/hybrid_bilateral \
        --use-tps-lower \
        --limit 100 \
        --verify
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import glob

# Add parent directory to path to import from inference scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference'))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate bilateral annotations using trained model'
    )
    parser.add_argument('--config', default='configs/H2_bilateral_preprocessed.yaml',
                       help='Path to config file')
    parser.add_argument('--model', help='Path to trained model (overrides config)')
    parser.add_argument('--source-images', required=True,
                       help='Directory containing images to annotate')
    parser.add_argument('--output-dir', default='data/bilateral_annotated',
                       help='Output directory for annotations')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--verify', action='store_true',
                       help='Generate verification visualizations')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of images to process (for testing)')

    # Hybrid annotation mode (TPS + Model)
    parser.add_argument('--tps-dir', default=None,
                       help='Directory containing TPS files (for hybrid mode)')
    parser.add_argument('--use-tps-lower', action='store_true',
                       help='Use TPS annotations for lower side, model for upper side')

    return parser.parse_args()


def load_model_and_config(args):
    """Load configuration and model"""
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    # Get model path
    if args.model:
        model_path = args.model
    else:
        train_name = cfg.get('train', {}).get('name', 'H2_bilateral_preprocessed')
        model_path = f"runs/detect/{train_name}/weights/best.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    return model, cfg


# ============================================================================
# TPS Annotation Functions (for Hybrid Mode)
# ============================================================================

def read_tps_file(tps_path):
    """
    Read landmarks from TPS file.
    Returns list of (x, y) tuples.
    """
    import re
    points = []
    try:
        with open(tps_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('LM'):
                    continue
                if re.match(r'^-?\d', line):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        points.append((x, y))
    except Exception as e:
        print(f"  Warning: Error reading TPS file {tps_path}: {e}")
        return []
    return points


def landmarks_to_bbox(landmarks, img_width, img_height, padding_ratio_x=0.01, padding_ratio_y=0.01):
    """
    Convert TPS landmarks to bounding box coordinates.
    TPS uses bottom-left origin, so we need to flip Y coordinates.

    Args:
        landmarks: List of (x, y) tuples in TPS coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        padding_ratio_x: Padding ratio for width
        padding_ratio_y: Padding ratio for height

    Returns:
        [x_min, y_min, x_max, y_max] in image coordinates
    """
    if not landmarks:
        return None

    # Convert TPS coordinates (bottom-left origin) to image coordinates (top-left origin)
    converted_landmarks = [(x, img_height - y) for x, y in landmarks]

    x_coords, y_coords = zip(*converted_landmarks)

    # Calculate tight bounding box
    landmark_width = max(x_coords) - min(x_coords)
    landmark_height = max(y_coords) - min(y_coords)

    # Add padding
    padding_x = max(5, landmark_width * padding_ratio_x)
    padding_y = max(5, landmark_height * padding_ratio_y)

    x_min = max(0, min(x_coords) - padding_x)
    x_max = min(img_width, max(x_coords) + padding_x)
    y_min = max(0, min(y_coords) - padding_y)
    y_max = min(img_height, max(y_coords) + padding_y)

    return [x_min, y_min, x_max, y_max]


def get_tps_paths(image_name, tps_dir):
    """
    Get paths to finger and toe TPS files for a given image.

    Args:
        image_name: Name of image file (e.g., "1001.jpg")
        tps_dir: Directory containing TPS files

    Returns:
        (finger_tps_path, toe_tps_path) or (None, None) if not found
    """
    base_name = Path(image_name).stem  # Remove extension
    finger_tps = Path(tps_dir) / f"{base_name}_finger.TPS"
    toe_tps = Path(tps_dir) / f"{base_name}_toe.TPS"

    if finger_tps.exists() and toe_tps.exists():
        return str(finger_tps), str(toe_tps)
    else:
        return None, None


# ============================================================================
# Model Detection Functions
# ============================================================================

def detect_upper_region_only(image, model, conf, iou, overlap=0.1):
    """
    Detect objects only in the upper region by flipping it.
    This is used in hybrid mode where lower side comes from TPS.

    Args:
        image: PIL Image
        model: YOLO model
        conf: Confidence threshold
        iou: IoU threshold
        overlap: Overlap ratio between regions

    Returns:
        List of detections in upper region with un-flipped coordinates
    """
    img_width, img_height = image.size

    # Calculate upper region with overlap
    mid_point = img_height // 2
    overlap_pixels = int(img_height * overlap)
    upper_region = (0, 0, img_width, mid_point + overlap_pixels)

    # Crop and flip upper region
    upper_half = image.crop(upper_region)
    upper_flipped = upper_half.transpose(Image.FLIP_TOP_BOTTOM)

    # Run detection on flipped upper region
    upper_results = model.predict(upper_flipped, conf=conf, iou=iou, verbose=False)

    upper_detections = []

    # Process detections and un-flip coordinates
    if upper_results and upper_results[0].boxes is not None:
        upper_height = upper_half.size[1]
        for box in upper_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_score = float(box.conf[0])
            cls = int(box.cls[0])

            # Un-flip Y coordinates
            y1_orig = upper_height - y2
            y2_orig = upper_height - y1

            upper_detections.append({
                'class': cls,
                'bbox': [x1, y1_orig, x2, y2_orig],
                'confidence': conf_score,
                'region': 'upper',
                'source': 'model'
            })

    return upper_detections


def split_and_flip_detection(image, model, conf, iou, overlap=0.1):
    """
    Run bilateral detection: split image + flip upper region
    Returns detections with their coordinates in original image space
    """
    img_width, img_height = image.size

    # Calculate regions with overlap
    mid_point = img_height // 2
    overlap_pixels = int(img_height * overlap)

    # Define regions
    upper_region = (0, 0, img_width, mid_point + overlap_pixels)
    lower_region = (0, mid_point - overlap_pixels, img_width, img_height)

    # Crop regions
    upper_half = image.crop(upper_region)
    lower_half = image.crop(lower_region)

    # Flip upper region vertically to make it look like lower region
    upper_flipped = upper_half.transpose(Image.FLIP_TOP_BOTTOM)

    # Run detection on both regions
    upper_results = model.predict(upper_flipped, conf=conf, iou=iou, verbose=False)
    lower_results = model.predict(lower_half, conf=conf, iou=iou, verbose=False)

    all_detections = []

    # Process flipped upper detections (un-flip coordinates)
    if upper_results and upper_results[0].boxes is not None:
        upper_height = upper_half.size[1]
        for box in upper_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_score = float(box.conf[0])
            cls = int(box.cls[0])

            # Un-flip y coordinates
            y1_orig = upper_height - y2
            y2_orig = upper_height - y1

            all_detections.append({
                'class': cls,
                'bbox': [x1, y1_orig, x2, y2_orig],
                'confidence': conf_score,
                'region': 'upper'
            })

    # Process lower detections
    if lower_results and lower_results[0].boxes is not None:
        for box in lower_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_score = float(box.conf[0])
            cls = int(box.cls[0])

            # Adjust y coordinates to full image space
            y1_adjusted = y1 + mid_point - overlap_pixels
            y2_adjusted = y2 + mid_point - overlap_pixels

            all_detections.append({
                'class': cls,
                'bbox': [x1, y1_adjusted, x2, y2_adjusted],
                'confidence': conf_score,
                'region': 'lower'
            })

    return all_detections


def classify_detections_by_position(detections, img_height, threshold=0.5):
    """
    Classify detections into upper and lower regions
    Returns dict with separated detections by class and region
    """
    classified = {
        'upper_finger': [],
        'upper_toe': [],
        'lower_finger': [],
        'lower_toe': [],
        'rulers': []
    }

    class_names = ['finger', 'toe', 'ruler']

    for det in detections:
        cls_id = det['class']
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'

        # Calculate center y in normalized coordinates
        bbox = det['bbox']
        center_y = ((bbox[1] + bbox[3]) / 2) / img_height

        # Classify by position
        if cls_name == 'ruler':
            # Only keep lower side ruler
            if center_y >= threshold:
                classified['rulers'].append(det)
        elif center_y < threshold:
            classified[f'upper_{cls_name}'].append(det)
        else:
            classified[f'lower_{cls_name}'].append(det)

    return classified


def select_best_detections(classified):
    """
    Select the best detection for each position (highest confidence)
    Ensures we get exactly 2 fingers, 2 toes, and 1-2 rulers
    """
    selected = {
        'upper_finger': None,
        'upper_toe': None,
        'lower_finger': None,
        'lower_toe': None,
        'rulers': []
    }

    # Select best for each position
    for key in ['upper_finger', 'upper_toe', 'lower_finger', 'lower_toe']:
        if classified[key]:
            # Sort by confidence and take the best
            best = max(classified[key], key=lambda x: x['confidence'])
            selected[key] = best

    # Select top 1 ruler (lower side only)
    if classified['rulers']:
        sorted_rulers = sorted(classified['rulers'], key=lambda x: x['confidence'], reverse=True)
        selected['rulers'] = sorted_rulers[:1]

    return selected


def convert_to_yolo_format(detections, img_width, img_height):
    """
    Convert detections to YOLO format
    Returns list of YOLO label lines
    """
    labels = []

    # Map position keys to class IDs (keeping original 3-class structure)
    class_mapping = {
        'upper_finger': 0,
        'lower_finger': 0,
        'upper_toe': 1,
        'lower_toe': 1,
        'ruler': 2
    }

    for key, det in detections.items():
        if key == 'rulers':
            for ruler_det in det:
                bbox = ruler_det['bbox']
                class_id = class_mapping['ruler']

                # Convert to YOLO format (normalized xywh)
                x_center = ((bbox[0] + bbox[2]) / 2) / img_width
                y_center = ((bbox[1] + bbox[3]) / 2) / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height

                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        elif det is not None:
            bbox = det['bbox']
            class_id = class_mapping[key]

            # Convert to YOLO format
            x_center = ((bbox[0] + bbox[2]) / 2) / img_width
            y_center = ((bbox[1] + bbox[3]) / 2) / img_height
            width = (bbox[2] - bbox[0]) / img_width
            height = (bbox[3] - bbox[1]) / img_height

            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return labels


def save_yolo_labels(image_name, labels, output_dir):
    """Save YOLO format labels to file"""
    labels_dir = Path(output_dir) / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_name).stem
    label_file = labels_dir / f"{base_name}.txt"

    with open(label_file, 'w') as f:
        for label in labels:
            f.write(label + '\n')


def create_verification_image(image, detections, output_dir, image_name):
    """Create visualization with all detections marked (clean style, no confidence labels)"""
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Create a copy for drawing
    img_copy = image.copy().convert('RGB')
    draw = ImageDraw.Draw(img_copy)

    # Color mapping
    colors = {
        'upper_finger': 'cyan',
        'lower_finger': 'blue',
        'upper_toe': 'yellow',
        'lower_toe': 'red',
        'ruler': 'purple'
    }

    # Draw detections (clean boxes, no labels)
    for key, det in detections.items():
        if key == 'rulers':
            for ruler_det in det:
                bbox = ruler_det['bbox']
                color = colors['ruler']
                # Ensure bbox coordinates are in correct order
                x_min, y_min, x_max, y_max = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        elif det is not None:
            bbox = det['bbox']
            color = colors[key]
            # Ensure bbox coordinates are in correct order
            x_min, y_min, x_max, y_max = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

    # Save
    base_name = Path(image_name).stem
    output_path = vis_dir / f"{base_name}_bilateral.jpg"
    img_copy.save(output_path, quality=95)


def process_image_hybrid(image_path, tps_finger, tps_toe, model, args, output_dir):
    """
    Process image using hybrid annotation:
    - Lower side: TPS annotations (high precision)
    - Upper side: Model detection (automated)
    - Ruler: TPS annotations
    """
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Step 1: Load TPS annotations for lower side
        finger_landmarks = read_tps_file(tps_finger)
        toe_landmarks = read_tps_file(tps_toe)

        if not finger_landmarks or not toe_landmarks:
            print(f"  Warning: Failed to read TPS files for {Path(image_path).name}, falling back to full model detection")
            return process_image(image_path, model, args, output_dir)

        # Step 2: Convert TPS to bounding boxes (lower side)
        # Ruler is first 2 points from finger landmarks
        ruler_points = finger_landmarks[:2]
        finger_points = finger_landmarks[2:]  # Skip ruler points
        toe_points = toe_landmarks[2:]

        ruler_bbox = landmarks_to_bbox(ruler_points, img_width, img_height, padding_ratio_x=0.05, padding_ratio_y=0.2)
        lower_finger_bbox = landmarks_to_bbox(finger_points, img_width, img_height)
        lower_toe_bbox = landmarks_to_bbox(toe_points, img_width, img_height)

        # Step 3: Detect upper side with model
        upper_detections = detect_upper_region_only(image, model, args.conf, args.iou)

        # Step 4: Select best upper detections
        upper_finger = None
        upper_toe = None
        for det in upper_detections:
            if det['class'] == 0:  # Finger
                if upper_finger is None or det['confidence'] > upper_finger['confidence']:
                    upper_finger = det
            elif det['class'] == 1:  # Toe
                if upper_toe is None or det['confidence'] > upper_toe['confidence']:
                    upper_toe = det

        # Step 5: Combine TPS (lower) + Model (upper)
        selected = {
            'lower_finger': {
                'bbox': lower_finger_bbox,
                'confidence': 1.0,  # TPS annotations have perfect confidence
                'source': 'tps'
            } if lower_finger_bbox else None,
            'lower_toe': {
                'bbox': lower_toe_bbox,
                'confidence': 1.0,
                'source': 'tps'
            } if lower_toe_bbox else None,
            'upper_finger': upper_finger,
            'upper_toe': upper_toe,
            'rulers': [{
                'bbox': ruler_bbox,
                'confidence': 1.0,
                'source': 'tps'
            }] if ruler_bbox else []
        }

        # Step 6: Convert to YOLO format
        yolo_labels = convert_to_yolo_format(selected, img_width, img_height)

        # Step 7: Save labels and copy image
        save_yolo_labels(Path(image_path).name, yolo_labels, output_dir)

        images_dir = Path(output_dir) / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        image.save(images_dir / Path(image_path).name)

        # Step 8: Create verification image with source annotations
        if args.verify:
            create_verification_image_hybrid(image, selected, output_dir, Path(image_path).name)

        # Step 9: Return statistics
        stats = {
            'total_detections': len(yolo_labels),
            'has_upper_finger': selected['upper_finger'] is not None,
            'has_lower_finger': selected['lower_finger'] is not None,
            'has_upper_toe': selected['upper_toe'] is not None,
            'has_lower_toe': selected['lower_toe'] is not None,
            'num_rulers': len(selected['rulers']),
            'mode': 'hybrid',
            'lower_source': 'tps',
            'upper_source': 'model'
        }

        return stats

    except Exception as e:
        print(f"  Error processing {Path(image_path).name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_verification_image_hybrid(image, detections, output_dir, image_name):
    """Create visualization with source annotations (TPS vs Model)"""
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Create a copy for drawing
    img_copy = image.copy().convert('RGB')
    draw = ImageDraw.Draw(img_copy)

    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font

    # Color mapping with source distinction
    colors = {
        'upper_finger': ('cyan', 2),      # Model detection
        'lower_finger': ('blue', 4),       # TPS annotation (thicker)
        'upper_toe': ('yellow', 2),        # Model detection
        'lower_toe': ('red', 4),           # TPS annotation (thicker)
        'ruler': ('purple', 4)             # TPS annotation (thicker)
    }

    # Draw detections with source labels
    for key, det in detections.items():
        if key == 'rulers':
            for i, ruler_det in enumerate(det):
                bbox = ruler_det['bbox']
                conf = ruler_det.get('confidence', 1.0)
                source = ruler_det.get('source', 'model')
                color, width = colors['ruler']

                # Ensure bbox coordinates are in correct order [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
                label = f"ruler {i+1}: {conf:.2f} ({source.upper()})"
                draw.text((x_min, y_min - 25), label, fill=color, font=font)
        elif det is not None:
            bbox = det['bbox']
            conf = det.get('confidence', 0.0)
            source = det.get('source', 'model')
            color, width = colors[key]

            # Ensure bbox coordinates are in correct order
            x_min, y_min, x_max, y_max = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
            label = f"{key}: {conf:.2f} ({source.upper()})"
            draw.text((x_min, y_min - 25), label, fill=color, font=font)

    # Add legend
    legend_y = 10
    draw.rectangle([10, legend_y, 200, legend_y + 60], fill='white', outline='black', width=2)
    draw.text((15, legend_y + 5), "Sources:", fill='black', font=small_font)
    draw.text((15, legend_y + 25), "Thick lines = TPS (precise)", fill='blue', font=small_font)
    draw.text((15, legend_y + 42), "Thin lines = Model (auto)", fill='cyan', font=small_font)

    # Save
    base_name = Path(image_name).stem
    output_path = vis_dir / f"{base_name}_hybrid.jpg"
    img_copy.save(output_path, quality=95)


def process_image(image_path, model, args, output_dir):
    """Process a single image and generate bilateral annotations"""
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Run bilateral detection
        detections = split_and_flip_detection(image, model, args.conf, args.iou)

        if not detections:
            print(f"  Warning: No detections found in {Path(image_path).name}")
            return None

        # Classify by position
        classified = classify_detections_by_position(detections, img_height)

        # Select best detections
        selected = select_best_detections(classified)

        # Convert to YOLO format
        yolo_labels = convert_to_yolo_format(selected, img_width, img_height)

        # Save labels
        save_yolo_labels(Path(image_path).name, yolo_labels, output_dir)

        # Copy image to output directory
        images_dir = Path(output_dir) / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        image.save(images_dir / Path(image_path).name)

        # Create verification image if requested
        if args.verify:
            create_verification_image(image, selected, output_dir, Path(image_path).name)

        # Return statistics
        stats = {
            'total_detections': len(yolo_labels),
            'has_upper_finger': selected['upper_finger'] is not None,
            'has_lower_finger': selected['lower_finger'] is not None,
            'has_upper_toe': selected['upper_toe'] is not None,
            'has_lower_toe': selected['lower_toe'] is not None,
            'num_rulers': len(selected['rulers']),
            'mode': 'full_model'
        }

        return stats

    except Exception as e:
        print(f"  Error processing {Path(image_path).name}: {str(e)}")
        return None


def main():
    args = parse_args()

    # Load model and config
    model, cfg = load_model_and_config(args)

    # Get image files
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(args.source_images, ext)))

    image_files.sort()

    if not image_files:
        print(f"No images found in {args.source_images}")
        return

    # Filter out flipud images (we only want original images)
    image_files = [f for f in image_files if '_flipud' not in Path(f).stem]

    if args.limit:
        image_files = image_files[:args.limit]

    # Check if hybrid mode is enabled
    hybrid_mode = args.use_tps_lower and args.tps_dir is not None

    print(f"Found {len(image_files)} images to process")
    print(f"Mode: {'HYBRID (TPS + Model)' if hybrid_mode else 'Full Model Detection'}")
    if hybrid_mode:
        print(f"TPS directory: {args.tps_dir}")
        print(f"  Lower side: TPS annotations (high precision)")
        print(f"  Upper side: Model detection (automated)")
    print(f"Output directory: {args.output_dir}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Verification images: {'Yes' if args.verify else 'No'}")
    print()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Process all images
    all_stats = []
    failed = []
    tps_missing = []

    for img_path in tqdm(image_files, desc="Generating annotations"):
        if hybrid_mode:
            # Try to get TPS files
            finger_tps, toe_tps = get_tps_paths(Path(img_path).name, args.tps_dir)
            if finger_tps and toe_tps:
                stats = process_image_hybrid(img_path, finger_tps, toe_tps, model, args, args.output_dir)
            else:
                tps_missing.append(Path(img_path).name)
                # Fallback to full model detection
                stats = process_image(img_path, model, args, args.output_dir)
        else:
            stats = process_image(img_path, model, args, args.output_dir)

        if stats:
            all_stats.append(stats)
        else:
            failed.append(Path(img_path).name)

    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Mode: {'HYBRID (TPS + Model)' if hybrid_mode else 'Full Model Detection'}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Successfully annotated: {len(all_stats)}")
    print(f"Failed: {len(failed)}")

    if hybrid_mode and tps_missing:
        print(f"\nTPS Status:")
        print(f"  Images with TPS files: {len(all_stats) - len(tps_missing)}")
        print(f"  Images missing TPS (used model fallback): {len(tps_missing)}")

    if all_stats:
        print("\nDetection Statistics:")
        print(f"  Images with upper finger: {sum(s['has_upper_finger'] for s in all_stats)} "
              f"({100*sum(s['has_upper_finger'] for s in all_stats)/len(all_stats):.1f}%)")
        print(f"  Images with lower finger: {sum(s['has_lower_finger'] for s in all_stats)} "
              f"({100*sum(s['has_lower_finger'] for s in all_stats)/len(all_stats):.1f}%)")
        print(f"  Images with upper toe: {sum(s['has_upper_toe'] for s in all_stats)} "
              f"({100*sum(s['has_upper_toe'] for s in all_stats)/len(all_stats):.1f}%)")
        print(f"  Images with lower toe: {sum(s['has_lower_toe'] for s in all_stats)} "
              f"({100*sum(s['has_lower_toe'] for s in all_stats)/len(all_stats):.1f}%)")

        avg_detections = sum(s['total_detections'] for s in all_stats) / len(all_stats)
        print(f"\n  Average detections per image: {avg_detections:.1f}")

        complete_bilateral = sum(
            s['has_upper_finger'] and s['has_lower_finger'] and
            s['has_upper_toe'] and s['has_lower_toe']
            for s in all_stats
        )
        print(f"  Complete bilateral annotations: {complete_bilateral} "
              f"({100*complete_bilateral/len(all_stats):.1f}%)")

        if hybrid_mode:
            hybrid_count = sum(1 for s in all_stats if s.get('mode') == 'hybrid')
            print(f"\n  Hybrid annotations (TPS+Model): {hybrid_count}")
            print(f"  Full model annotations: {len(all_stats) - hybrid_count}")

    if failed:
        print(f"\nFailed images ({len(failed)}):")
        for name in failed[:10]:
            print(f"  - {name}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

    print(f"\nOutput saved to: {args.output_dir}/")
    print(f"  - Images: {args.output_dir}/images/")
    print(f"  - Labels: {args.output_dir}/labels/")
    if args.verify:
        print(f"  - Visualizations: {args.output_dir}/visualizations/")

    print("\nNext steps:")
    print("1. Review verification images (if generated)")
    print("2. Manually correct any errors using LabelImg or similar tool")
    print("3. Split dataset for training:")
    print(f"   python scripts/preprocessing/split_dataset.py \\")
    print(f"     --images-dir {args.output_dir}/images \\")
    print(f"     --labels-dir {args.output_dir}/labels \\")
    print(f"     --output-dir data/dataset_bilateral")
    print("4. Train model with complete bilateral annotations")


if __name__ == "__main__":
    main()
