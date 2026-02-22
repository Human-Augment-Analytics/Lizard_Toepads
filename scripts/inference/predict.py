from ultralytics import YOLO
from PIL import Image
import os
import argparse
import yaml
from pathlib import Path
import glob

Image.MAX_IMAGE_PIXELS = None

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained YOLO model')
    parser.add_argument('--config', default='configs/H1.yaml', help='Path to project config file')
    parser.add_argument('--model', help='Path to trained model weights')
    parser.add_argument('--source', help='Path to image or directory for inference')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with first 50 images from processed/images')
    parser.add_argument('--conf', type=float, help='Confidence threshold (overrides config)')
    parser.add_argument('--iou', type=float, help='IoU threshold for NMS (overrides config)')
    parser.add_argument('--imgsz', type=int, help='Image size for inference (overrides config)')
    parser.add_argument('--save', action='store_true', help='Save results (overrides config)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results (overrides config)')
    parser.add_argument('--save-txt', action='store_true', help='Save results as txt files (overrides config)')
    parser.add_argument('--project', help='Results save directory (overrides config)')
    parser.add_argument('--task', choices=['detect', 'obb'], help='YOLO task type (default: detect)')
    return parser.parse_args()


def get_model_from_config(cfg):
    """Get model path based on training name from config"""
    train_name = cfg.get('train', {}).get('name', 'H1')
    model_path = f"runs/detect/{train_name}/weights/best.pt"

    if os.path.exists(model_path):
        return model_path
    else:
        # Fallback to latest model
        return get_latest_model()


def get_latest_model():
    """Find the latest best.pt model in runs/detect/"""
    model_paths = glob.glob("runs/detect/*/weights/best.pt")
    if not model_paths:
        raise FileNotFoundError("No trained models found in runs/detect/*/weights/best.pt")

    # Get the most recent model by modification time
    latest_model = max(model_paths, key=os.path.getmtime)
    return latest_model


def preprocess_for_inference(img_path, target_size=1024):
    """
    Preprocess image to exactly match the training pipeline.
    This ensures the model sees the same type of input it was trained on.
    """
    with Image.open(img_path) as img:
        print(f"Original image: {img.size}, mode: {img.mode}")

        # Step 1: resize with same method as training
        original_width, original_height = img.size
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # resize with same resampling method as training
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Step 2: convert to grayscale (matching training preprocessing)
        if resized_img.mode != 'L':
            resized_img = resized_img.convert('L')

        # Step 3: convert back to RGB for YOLO inference (3 channels expected)
        resized_img = resized_img.convert('RGB')

        print(f"Processed image: {resized_img.size}, mode: {resized_img.mode}")
        return resized_img


def run_inference(model_path, source_path, conf=0.25, iou=0.45, imgsz=1024, save=True, save_txt=False, project="results", name="inference"):
    """Run inference on images"""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running inference on: {source_path}")

    # Check if source is a directory, list of files, or single file
    if isinstance(source_path, list):
        # List of files (from quick-test mode)
        source = source_path
        print(f"Processing {len(source)} selected images")
    elif os.path.isdir(source_path):
        # Process directory
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_exts:
            image_files.extend(glob.glob(os.path.join(source_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(source_path, f"*{ext.upper()}")))

        if not image_files:
            print(f"No images found in {source_path}")
            return

        print(f"Found {len(image_files)} images to process")
        source = image_files
    else:
        # Single file
        source = source_path

    # Run inference with optimized settings to avoid OOM
    results = model.predict(
        source=source,
        save=save,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        show_labels=True,
        show_conf=True,
        save_txt=save_txt,
        project=project,
        name=name,
        exist_ok=True,
        stream=True,      # Use streaming to avoid memory accumulation
        augment=False,    # Disable Test Time Augmentation to save memory
        batch=1,          # Process one image at a time to minimize GPU memory usage
    )

    # Print results summary
    total_detections = 0
    class_names = model.names
    image_count = 0

    print(f"Processing {len(source) if isinstance(source, list) else 'all'} images...")
    for i, r in enumerate(results):
        image_count += 1
        if image_count % 50 == 0:
            print(f"  Processed {image_count} images...")

        # OBB results use r.obb, standard detection uses r.boxes
        detections = r.obb if r.obb is not None and len(r.obb) > 0 else r.boxes
        if detections is not None:
            num_detections = len(detections)
            total_detections += num_detections

            for j in range(num_detections):
                class_id = int(detections.cls[j])
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                confidence = float(detections.conf[j])
                print(f"  {j+1}. {class_name}: {confidence:.3f} confidence")

                if hasattr(detections, 'xywhr') and detections.xywhr is not None:
                    # OBB: print center, size, and rotation angle
                    cx, cy, w, h, angle = detections.xywhr[j].tolist()
                    import math
                    print(f"     OBB: center=({cx:.1f}, {cy:.1f}) size=({w:.1f}x{h:.1f}) angle={math.degrees(angle):.1f}°")
                else:
                    x1, y1, x2, y2 = detections.xyxy[j].tolist()
                    print(f"     Box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")

    print(f"\n=== Summary ===")
    print(f"Total images processed: {image_count}")
    print(f"Total detections: {total_detections}")
    print(f"Results saved to: {project}/{name}/")


def main():
    args = parse_args()

    # Load config if provided
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    # Get inference parameters from config, with defaults
    inference_cfg = cfg.get('inference', {})

    # Determine parameters with priority: command line > config > defaults
    conf = args.conf if args.conf is not None else inference_cfg.get('conf', 0.25)
    iou = args.iou if args.iou is not None else inference_cfg.get('iou', 0.45)
    imgsz = args.imgsz if args.imgsz is not None else inference_cfg.get('imgsz', 1024)

    # Handle save parameter with --save and --no-save flags
    if args.no_save:
        save = False
    elif args.save:
        save = True
    else:
        save = inference_cfg.get('save', True)

    save_txt = args.save_txt if args.save_txt else inference_cfg.get('save_txt', False)
    project = args.project if args.project else inference_cfg.get('project', 'results')

    print(f"Inference parameters from {'command line/config' if args.config else 'defaults'}:")
    print(f"  - Confidence threshold: {conf}")
    print(f"  - IoU threshold: {iou}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Save results: {save}")
    print(f"  - Save txt: {save_txt}")
    print(f"  - Project directory: {project}")

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        try:
            model_path = get_model_from_config(cfg)
            print(f"Using model from config: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify model path with --model or train a model first")
            return

    # Determine source path
    if args.quick_test:
        # Quick test mode: use first 50 images from dataset/images/val
        val_dir = "data/dataset/images/val"
        if not os.path.exists(val_dir):
            print(f"Error: Validation images directory not found: {val_dir}")
            return

        # Get first 50 images
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        all_images = []
        for ext in image_exts:
            all_images.extend(glob.glob(os.path.join(val_dir, f"*{ext}")))
            all_images.extend(glob.glob(os.path.join(val_dir, f"*{ext.upper()}")))

        all_images.sort()
        source_path = all_images[:50]  # First 50 images
        if not source_path:
            print(f"No images found in {val_dir}")
            return
        print(f"Quick test mode: Testing with {len(source_path)} validation images from {val_dir}")

    elif args.source:
        source_path = args.source
    else:
        # Try to use original data directory as default
        default_sources = [
            "/storage/ice-shared/cs8903onl/miami_fall_24_jpgs",
            "data/miami_fall_24_jpgs",
            "data/processed/images"
        ]
        source_path = None
        for src in default_sources:
            if os.path.exists(src):
                source_path = src
                break

        if not source_path:
            print("Error: No source images found. Please specify with --source or use --quick-test")
            return

    # Run inference with better output organization
    train_name = cfg.get('train', {}).get('name', 'H1')
    run_inference(
        model_path=model_path,
        source_path=source_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=save,
        save_txt=save_txt,
        project=project,
        name=f"{train_name}_inference"
    )


if __name__ == "__main__":
    main()