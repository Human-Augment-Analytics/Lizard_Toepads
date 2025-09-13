## Lizard Toepads – Codebase Overview

### What this repository does

This repository provides a complete pipeline to train and run a YOLOv11 object detector to localize lizard toepads (and related structures) in high‑resolution scans.

- Converts digitized TPS landmark files into YOLO detection labels
- Trains a detector with Ultralytics YOLOv11 using a standard dataset layout
- Runs inference and saves predictions for visual inspection and downstream measurement

### Key components

- `process_tps_files.py`

  - Purpose: Convert TPS landmark points to YOLO `.txt` detection labels and generate verification images.
  - Assumptions:
    - Each image has two TPS files: `<basename>_finger.TPS` and `<basename>_toe.TPS`.
    - The first two landmarks in each TPS are treated as the Ruler endpoints; remaining points define the biological region.
    - Images are resized to a target size (default 1024) while preserving aspect ratio; landmarks are scaled accordingly.
    - Grayscale conversion is applied for consistency, then converted back to RGB for YOLO.
  - Outputs:
    - `output/labels/<basename>.txt` in YOLO format: one line per detected class (`class cx cy w h`, normalized 0–1)
    - `output/marked_<basename>.jpg` with drawn boxes for manual QA
    - `output/labels/classes.txt` listing class names (`Finger`, `Toe`, `Ruler`)
  - CLI example (PowerShell / single line):
    ```bash
    python process_tps_files.py --image-dir data/Miami_Fall_2024_Toepads --tps-dir data/archive/24fall_raw_tps --output-dir output --add-points
    ```

- `yolo_detection.py`

  - Purpose: Train a YOLOv11 detector.
  - Reads dataset configuration from `data/data.yaml`.
  - Uses larger image size (1024), optional multi‑GPU, and common speedups (AMP, caching).
  - Produces training runs under `runs/detect/<experiment_name>` with checkpoints and metrics.

- `yolo_predict.py`

  - Purpose: Run inference on image(s) with a trained model and save predicted images/labels.
  - Applies preprocessing aligned with training (resize to 1024, grayscale → RGB) before prediction.
  - Saves results under `runs/detect/predict/`.

- `data/data.yaml`
  - YOLO dataset configuration. Example in repo:
    ```yaml
    path: /home/hice1/dherbig3/lizard-toepads/dataset_new
    train: images/train
    val: images/val
    nc: 3
    names: ["finger", "toe", "ruler"]
    ```

### Expected dataset layout

Under the directory specified by `path` in `data/data.yaml`:

```
dataset_root/
  images/
    train/           # training images (.jpg/.png)
    val/             # validation images
  labels/
    train/           # YOLO label files (.txt), one per image
    val/
```

Naming must match per split, e.g. `images/train/1001.jpg` ↔ `labels/train/1001.txt`.

Label file format (one object per line):

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to 0–1. Class indices follow `names` order in `data.yaml`.

### End‑to‑end workflow

1. Prepare TPS → YOLO labels

   - Place images in `data/Miami_Fall_2024_Toepads`
   - Place TPS files in `data/archive/24fall_raw_tps`, with paired names:
     - `1234.jpg` ↔ `1234_finger.TPS` and `1234_toe.TPS`
   - Run `process_tps_files.py` (see example above). Review `output/marked_*.jpg`.

2. Assemble the YOLO dataset

   - Copy/organize images and generated labels into `dataset_root/images/{train,val}` and `dataset_root/labels/{train,val}`.
   - Update `data/data.yaml` to point to `dataset_root` and set correct `names` and `nc`.

3. Train

   - Example (Ultralytics CLI):
     ```bash
     yolo task=detect mode=train model=yolov11n.pt data=data/data.yaml epochs=50 imgsz=640
     ```
   - Or run `python yolo_detection.py` (uses 1024 and additional options).

4. Inference
   - Example (script): `python yolo_predict.py`
   - Or via CLI, setting `model` and `source` images.

### Assumptions and limitations

- TPS first two points are treated as Ruler endpoints. If your TPS lacks ruler points, the current script will misinterpret the first two biological landmarks as ruler. In that case, either add ruler points to TPS or adapt the script to a two‑class setup (no Ruler).
- Both `*_finger.TPS` and `*_toe.TPS` must exist for an image to be processed; otherwise, it is skipped.
- Images are processed as grayscale for training consistency, then converted back to RGB for the YOLO API.
- The pipeline currently expects `.jpg` files in the image directory.

### Customization tips

- Change classes:
  - Update `data/data.yaml` `names` and `nc` accordingly.
  - If removing the Ruler class, adjust training to two classes and modify the TPS‑to‑YOLO conversion to stop generating the Ruler label.
- Image size and performance:
  - `imgsz` in training (`640`–`1280`) balances accuracy and speed. Scripts show examples with `1024`.
  - Use AMP (`amp=True`) and `cache=True` to speed up training if memory allows.

### Repository highlights

- Data/config:
  - `data/data.yaml` – dataset config for YOLO
  - `data/Miami_Fall_2024_Toepads/` – source images (not tracked by git except `data.yaml`)
- Training & inference:
  - `yolo_detection.py` – training script
  - `yolo_predict.py` – inference script
- Preprocessing:
  - `process_tps_files.py` – TPS → YOLO labels + visual QA

If you need a variant without Ruler or with separate TPS folders for finger/toe, it can be added with minor changes.
