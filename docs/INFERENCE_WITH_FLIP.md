# Inference with Flip Strategy

This document explains the **Flip-Predict-FlipBack** inference strategy used to accurately detect **upper-view** limbs (`up_finger`, `up_toe`) by leveraging the model's high performance on **bottom-view** limbs (`bot_finger`, `bot_toe`).

## The Strategy

The model is highly trained to detect bottom-view limbs with precise Oriented Bounding Boxes (OBB). Since the lizard's upper-view limbs are morphologically similar (and symmetric) to the bottom-view limbs when flipped, we can use a two-pass inference approach:

1.  **Standard Pass**: Run inference on the original image.
    *   Keep highly confident `bot_finger`, `bot_toe`, `ruler`, and `id` detections.
    *   Ignore `up_finger` and `up_toe` detections (which are often less accurate or axis-aligned).
2.  **Flipped Pass**: Flip the image vertically (mirror across X-axis).
    *   The "upper" limbs now look like "bottom" limbs to the model.
    *   Run inference to detect `bot_finger` and `bot_toe` on this flipped image.
3.  **Flip Back & Merge**:
    *   Map the flipped `bot_finger` detections → `up_finger`.
    *   Map the flipped `bot_toe` detections → `up_toe`.
    *   Flip the coordinates back to the original image space.
    *   Combine with the results from the Standard Pass.

This results in a full set of accurate OBBs for all limb classes without requiring a separately trained upper-limb model.

## Prerequisites

### 1. Model Checkpoint
You need a trained YOLOv11-OBB model that performs well on `bot_finger` and `bot_toe`.
*   **Default Path**: `$PROJECT_ROOT/runs/obb/H6_obb_6class3/weights/best.pt`

### 2. Environment
Ensure your Python environment has the following installed:
*   `ultralytics` (YOLOv11)
*   `opencv-python`
*   `numpy`

(The `lizard` conda environment is already configured with these).

## Image Preprocessing

The script handles resizing automatically (YOLO default size 1280), but your source images should meet these criteria:

1.  **Orientation**: The images must be **dorsal views** (standard top-down view of the lizard).
    *   The "Flip" strategy assumes that flipping the image vertically brings the upper limbs into a position/orientation that resembles the bottom limbs' usual appearance.
    *   If images are rotated by 90 degrees or inverted, the flip logic may need adjustment.
2.  **Format**: Standard image formats (`.jpg`, `.png`, `.jpeg`).
3.  **Clarity**: Ensure limbs are visible. The model is robust, but occlusion will affect detection.

No manual cropping or resizing is required before running the script.

## Usage

Run the `inference_with_flip.py` script provided in `scripts/inference/`.

### Single Image
```bash
python3 scripts/inference/inference_with_flip.py \
  --source path/to/image.jpg \
  --output-dir inference_results
```

### Entire Directory
```bash
python3 scripts/inference/inference_with_flip.py \
  --source path/to/image_folder/ \
  --output-dir inference_results
```

### Custom Model or Parameters
You can override the model path and confidence thresholds:
```bash
python3 scripts/inference/inference_with_flip.py \
  --source data/test_images/ \
  --model runs/obb/H6_obb_6class3/weights/best.pt \
  --conf 0.25 \
  --iou 0.4
```
*   `--conf`: Confidence threshold (default 0.25). Higher values reduce false positives.
*   `--iou`: NMS IoU threshold (default 0.4). Lower values help suppress duplicate overlapping boxes.

## Output

The script saves visualized images (with drawn OBBs) to the specified `--output-dir`. Filenames are appended with `_flip_inf.jpg`.
