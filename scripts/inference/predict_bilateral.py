"""
Bilateral Toepad Detection Script
Implements Approach 2: Multi-scale detection without retraining
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import argparse
import yaml
from pathlib import Path
import glob

Image.MAX_IMAGE_PIXELS = None

def parse_args():
    parser = argparse.ArgumentParser(description='Run bilateral inference with multi-scale detection')
    parser.add_argument('--config', default='configs/H1.yaml', help='Path to project config file')
    parser.add_argument('--model', help='Path to trained model weights')
    parser.add_argument('--source', help='Path to image or directory for inference')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with first 10 images from val set')
    parser.add_argument('--conf', type=float, help='Confidence threshold (overrides config)')
    parser.add_argument('--iou', type=float, help='IoU threshold for NMS (overrides config)')
    parser.add_argument('--imgsz', type=int, help='Image size for inference (overrides config)')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--save-txt', action='store_true', help='Save results as txt files')
    parser.add_argument('--project', help='Results save directory (overrides config)')
    parser.add_argument('--method', choices=['split', 'flip', 'both'], default='both',
                       help='Detection method: split (region-based), flip (vertical flip upper), or both')
    parser.add_argument('--overlap', type=float, default=0.1,
                       help='Overlap ratio between regions (0-0.5)')
    
    add_rclone_args(parser)
    
    return parser.parse_args()


def get_model_from_config(cfg):
    """Get model path based on training name from config"""
    train_name = cfg.get('train', {}).get('name', 'H1')
    model_path = f"runs/detect/{train_name}/weights/best.pt"

    if os.path.exists(model_path):
        return model_path
    else:
        # Fallback to latest model
        model_paths = glob.glob("runs/detect/*/weights/best.pt")
        if not model_paths:
            raise FileNotFoundError("No trained models found")
        return max(model_paths, key=os.path.getmtime)


def split_image_detection(image_path, model, conf, iou, imgsz, overlap=0.1):
    """
    Split image into upper and lower regions with optional overlap
    """
    # Open image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Calculate split point with overlap
    mid_point = img_height // 2
    overlap_pixels = int(img_height * overlap)

    # Define regions
    upper_region = (0, 0, img_width, mid_point + overlap_pixels)
    lower_region = (0, mid_point - overlap_pixels, img_width, img_height)

    # Crop regions
    upper_half = img.crop(upper_region)
    lower_half = img.crop(lower_region)

    # Run inference on each region
    print(f"  - Detecting on upper region (with {overlap*100}% overlap)...")
    upper_results = model.predict(
        upper_half,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )

    print(f"  - Detecting on lower region (with {overlap*100}% overlap)...")
    lower_results = model.predict(
        lower_half,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )

    # Combine detections
    all_boxes = []
    all_confs = []
    all_classes = []

    # Process upper region detections
    if upper_results and upper_results[0].boxes is not None:
        for box in upper_results[0].boxes:
            # Adjust coordinates to full image space
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Upper region starts at y=0, no adjustment needed for y
            all_boxes.append([x1, y1, x2, y2])
            all_confs.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

    # Process lower region detections
    if lower_results and lower_results[0].boxes is not None:
        for box in lower_results[0].boxes:
            # Adjust coordinates to full image space
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Lower region starts at (mid_point - overlap_pixels)
            y1_adjusted = y1 + mid_point - overlap_pixels
            y2_adjusted = y2 + mid_point - overlap_pixels
            all_boxes.append([x1, y1_adjusted, x2, y2_adjusted])
            all_confs.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

    return all_boxes, all_confs, all_classes


def flip_detection(image_path, model, conf, iou, imgsz):
    """
    Detect on upper region by vertically flipping it (makes upper look like lower)
    """
    # Open image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Split into upper and lower
    upper_half = img.crop((0, 0, img_width, img_height // 2))
    lower_half = img.crop((0, img_height // 2, img_width, img_height))

    # Vertically flip upper half (upside down)
    upper_flipped = upper_half.transpose(Image.FLIP_TOP_BOTTOM)

    # Run inference
    print(f"  - Detecting on flipped upper region...")
    upper_flip_results = model.predict(
        upper_flipped,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )

    print(f"  - Detecting on lower region...")
    lower_results = model.predict(
        lower_half,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )

    # Combine detections
    all_boxes = []
    all_confs = []
    all_classes = []

    # Process flipped upper detections (need to un-flip coordinates)
    if upper_flip_results and upper_flip_results[0].boxes is not None:
        upper_width, upper_height = upper_half.size
        for box in upper_flip_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Un-flip coordinates (only flip y, not x)
            y1_orig = upper_height - y2
            y2_orig = upper_height - y1
            all_boxes.append([x1, y1_orig, x2, y2_orig])
            all_confs.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

    # Process lower detections
    if lower_results and lower_results[0].boxes is not None:
        for box in lower_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Adjust y coordinates
            y1_adjusted = y1 + img_height // 2
            y2_adjusted = y2 + img_height // 2
            all_boxes.append([x1, y1_adjusted, x2, y2_adjusted])
            all_confs.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

    return all_boxes, all_confs, all_classes


def combined_detection(image_path, model, conf, iou, imgsz, overlap=0.1):
    """
    Combine both split and flip detection methods
    """
    print(f"  - Running split detection...")
    split_boxes, split_confs, split_classes = split_image_detection(
        image_path, model, conf, iou, imgsz, overlap
    )

    print(f"  - Running flip detection...")
    flip_boxes, flip_confs, flip_classes = flip_detection(
        image_path, model, conf, iou, imgsz
    )

    # Combine all detections
    all_boxes = split_boxes + flip_boxes
    all_confs = split_confs + flip_confs
    all_classes = split_classes + flip_classes

    # Apply NMS to remove duplicates
    if all_boxes:
        all_boxes, all_confs, all_classes = apply_nms(
            all_boxes, all_confs, all_classes, iou
        )

    return all_boxes, all_confs, all_classes


def apply_nms(boxes, confs, classes, iou_threshold):
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    """
    if not boxes:
        return [], [], []

    # Convert to numpy arrays
    boxes = np.array(boxes)
    confs = np.array(confs)
    classes = np.array(classes)

    # Sort by confidence
    indices = np.argsort(confs)[::-1]

    keep = []
    while len(indices) > 0:
        # Take the first (highest confidence) detection
        i = indices[0]
        keep.append(i)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining boxes
        remaining_boxes = boxes[indices[1:]]
        ious = calculate_iou(boxes[i], remaining_boxes)

        # Only keep boxes with IoU below threshold
        indices = indices[1:][ious < iou_threshold]

    return boxes[keep].tolist(), confs[keep].tolist(), classes[keep].tolist()


def calculate_iou(box, boxes):
    """
    Calculate IoU between one box and multiple boxes
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - intersection

    return intersection / union


def run_bilateral_inference(model_path, source_path, conf, iou, imgsz,
                          method='both', overlap=0.1, save=True, save_txt=False,
                          project='results', name='bilateral_inference', uploader=None):
    """
    Run bilateral inference on images with saving functionality matching predict.py
    """
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running bilateral inference with method: {method}")
    print(f"Parameters: conf={conf}, iou={iou}, imgsz={imgsz}, overlap={overlap}")

    # Create output directory if saving
    output_dir = None
    if save:
        output_dir = Path(project) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / 'labels').mkdir(exist_ok=True) if save_txt else None
        print(f"Saving results to: {output_dir}")

    # Prepare source list
    if isinstance(source_path, list):
        source = source_path
    elif os.path.isdir(source_path):
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        source = []
        for ext in image_exts:
            source.extend(glob.glob(os.path.join(source_path, f"*{ext}")))
            source.extend(glob.glob(os.path.join(source_path, f"*{ext.upper()}")))
    else:
        source = [source_path]

    print(f"\nProcessing {len(source)} images...\n")

    # Process each image
    class_names = ['finger', 'toe', 'ruler']
    total_detections = 0
    results_summary = []

    for idx, img_path in enumerate(source):
        print(f"Image {idx+1}/{len(source)}: {os.path.basename(img_path)}")

        # Run detection based on method
        if method == 'split':
            boxes, confs, classes = split_image_detection(
                img_path, model, conf, iou, imgsz, overlap
            )
        elif method == 'flip':
            boxes, confs, classes = flip_detection(
                img_path, model, conf, iou, imgsz
            )
        else:  # both
            boxes, confs, classes = combined_detection(
                img_path, model, conf, iou, imgsz, overlap
            )

        # Save visualization if requested
        if save and boxes:
            save_visualization(img_path, boxes, confs, classes, class_names, output_dir)

        # Save txt labels if requested
        if save_txt and boxes:
            save_txt_labels(img_path, boxes, confs, classes, output_dir / 'labels')

        # Count detections by class
        class_counts = {}
        for cls in classes:
            cls_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        num_detections = len(boxes)
        total_detections += num_detections

        # Print results for this image
        if num_detections > 0:
            counts_str = ", ".join([f"{count} {name}" for name, count in class_counts.items()])
            print(f"  Found {num_detections} detections: {counts_str}")

            # Print detailed detection info (only first 5 to avoid clutter)
            for i, (box, conf_score, cls) in enumerate(zip(boxes[:5], confs[:5], classes[:5])):
                cls_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                print(f"    {i+1}. {cls_name}: {conf_score:.3f} at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            if num_detections > 5:
                print(f"    ... and {num_detections - 5} more detections")
        else:
            print(f"  No detections found")

        results_summary.append({
            'image': os.path.basename(img_path),
            'path': img_path,
            'detections': num_detections,
            'class_counts': class_counts,
            'boxes': boxes,
            'confidences': confs,
            'classes': classes
        })

        print()

    # Print summary
    print("=" * 50)
    print("BILATERAL DETECTION SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {len(source)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(source):.2f}")
    print(f"Detection method: {method}")

    # Class distribution
    total_class_counts = {}
    for result in results_summary:
        for cls_name, count in result['class_counts'].items():
            total_class_counts[cls_name] = total_class_counts.get(cls_name, 0) + count

    print("\nClass distribution:")
    for cls_name, count in total_class_counts.items():
        print(f"  {cls_name}: {count}")

    if save:
        print(f"\nResults saved to: {output_dir}/")
        # Save summary JSON
        save_summary(results_summary, output_dir)
        
    # Upload to cloud storage if uploader is provided and enabled
    if uploader and save:
        uploader.upload_results(output_dir, run_name=name)

    return results_summary


def save_visualization(img_path, boxes, confs, classes, class_names, output_dir):
    """
    Save image with detection boxes drawn
    """
    import cv2
    import numpy as np

    # Read image
    img = cv2.imread(img_path)

    # Define colors for each class (BGR format)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red

    # Draw boxes
    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls % len(colors)]
        cls_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{cls_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save image
    output_path = output_dir / Path(img_path).name
    cv2.imwrite(str(output_path), img)


def save_txt_labels(img_path, boxes, confs, classes, labels_dir):
    """
    Save detection results in YOLO txt format
    """
    img = Image.open(img_path)
    img_width, img_height = img.size

    # Create txt file
    txt_path = labels_dir / (Path(img_path).stem + '.txt')

    with open(txt_path, 'w') as f:
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box

            # Convert to normalized xywh format
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Write to file (class x_center y_center width height confidence)
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")


def save_summary(results_summary, output_dir):
    """
    Save detection summary as JSON
    """
    import json

    # Prepare summary data (remove non-serializable items)
    summary_data = {
        'total_images': len(results_summary),
        'total_detections': sum(r['detections'] for r in results_summary),
        'images': []
    }

    for result in results_summary:
        summary_data['images'].append({
            'name': result['image'],
            'path': result['path'],
            'detections': result['detections'],
            'class_counts': result['class_counts']
        })

    # Save as JSON
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary saved to: {summary_path}")


def main():
    args = parse_args()

    # Load config
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    # Get inference parameters from config
    inference_cfg = cfg.get('inference', {})

    # Determine parameters with priority: command line > config > defaults
    conf = args.conf if args.conf is not None else inference_cfg.get('conf', 0.25)
    iou = args.iou if args.iou is not None else inference_cfg.get('iou', 0.45)
    imgsz = args.imgsz if args.imgsz is not None else inference_cfg.get('imgsz', 1024)
    
    # Create uploader from args and config
    uploader = get_uploader_from_args(args, cfg)
    
    if uploader.enabled:
        print(f"  - Upload remote: {uploader.remote_name}")
        print(f"  - Upload path: {uploader.base_path}")

    # Handle save parameter
    if args.no_save:
        save = False
    elif args.save:
        save = True
    else:
        save = inference_cfg.get('save', True)

    save_txt = args.save_txt if args.save_txt else inference_cfg.get('save_txt', False)
    project = args.project if args.project else inference_cfg.get('project', 'results')

    # Get model path
    if args.model:
        model_path = args.model
    else:
        try:
            model_path = get_model_from_config(cfg)
            print(f"Using model from config: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    # Determine source
    if args.quick_test:
        val_dir = "data/dataset/images/val"
        if not os.path.exists(val_dir):
            print(f"Error: Validation directory not found: {val_dir}")
            return

        # Get first 10 images for quick test
        image_exts = ['.jpg', '.jpeg', '.png']
        all_images = []
        for ext in image_exts:
            all_images.extend(glob.glob(os.path.join(val_dir, f"*{ext}")))

        all_images.sort()
        source_path = all_images[:10]  # First 10 images
        print(f"Quick test mode: Testing with {len(source_path)} validation images")
    elif args.source:
        source_path = args.source
    else:
        print("Error: Please specify source with --source or use --quick-test")
        return

    # Run bilateral inference
    train_name = cfg.get('train', {}).get('name', 'H1')
    run_bilateral_inference(
        model_path=model_path,
        source_path=source_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        method=args.method,
        overlap=args.overlap,
        save=save,
        save_txt=save_txt,
        project=project,
        name=f"{train_name}_bilateral"
        uploader=uploader
    )


if __name__ == "__main__":
    main()