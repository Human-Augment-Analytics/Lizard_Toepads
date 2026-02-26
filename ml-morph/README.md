# ml-morph

Machine learning tools for landmark-based morphometrics.

## Overview

ml-morph predicts precise landmark positions on lizard toepad images using a two-stage pipeline:

1. **YOLO Detection** (Stage 1) — A pre-trained YOLOv11 model detects the region of interest (toe, finger, ruler) and produces bounding boxes
2. **dlib Shape Prediction** (Stage 2) — An ensemble of regression trees predicts (x, y) landmark coordinates within each detected bounding box

Three bounding box strategies are available for shape predictor training:

| Strategy | Config | Description |
|---|---|---|
| **Baseline** | `toe_training_yolo_baseline.yaml` | YOLO axis-aligned boxes, no rotation |
| **YOLO Bbox** | `toe_training_yolo_bbox.yaml` | YOLO axis-aligned boxes + 30% padding |
| **OBB** | `toe_training_yolo_obb.yaml` | YOLO oriented bounding boxes, crops rotated upright |

**Note:** The dlib shape predictor is very sensitive to bounding box size at inference time. Training must use the same bbox distribution (YOLO-derived) that will be used during inference.

> For a full walkthrough from environment setup to inference, see the [PACE Experiments Guide](../docs/ML_MORPH_PACE_EXPERIMENTS.md).
> For experimental comparisons, see [Baseline vs OBB](../docs/COMPARISON_BASELINE_VS_OBB.md) and [Crop & Rotate OBB](../docs/EXPERIMENT_CROP_ROTATE_OBB.md).

## Installation

Dependencies are managed by the parent project's `pyproject.toml`:

```bash
# From repository root
uv sync

# dlib is required (may need cmake + boost as system dependencies)
uv pip install dlib
```

**Note**: When using `uv run`, add `--no-sync` (or set `UV_NO_SYNC=1`) to prevent re-syncing which can replace CUDA PyTorch wheels with CPU versions.

## Quick Start

```bash
cd ml-morph

# Step 1: Convert TPS landmarks to dlib XML
uv run --no-sync python scripts/preprocessing/tps_to_xml.py \
  --image-dir /path/to/images \
  --tps-file consolidated_toe.tps \
  --train-ratio 0.8

# Step 2: Replace tight bboxes with YOLO-derived bboxes
uv run --no-sync python scripts/preprocessing/generate_yolo_bbox_xml.py \
  --config configs/toe_training_yolo_bbox.yaml

# Step 3: Train shape predictor on YOLO-sized bboxes
uv run --no-sync python shape_trainer.py \
  -d train_yolo_bbox.xml -t test_yolo_bbox.xml \
  -o toe_predictor_yolo_bbox -th 8

# Or use the unified workflow for the legacy dlib-only pipeline:
uv run --no-sync python scripts/train_workflow.py --config configs/toe_training.yaml
```

## Pipeline

### Step 1: Preprocessing (TPS to XML)

Convert TPS landmark files into dlib's XML annotation format:

```bash
uv run --no-sync python scripts/preprocessing/tps_to_xml.py \
  --image-dir /path/to/images \
  --tps-file consolidated_toe.tps \
  --train-ratio 0.8
```

This reads TPS files (flipping Y coordinates from TPS bottom-left origin to image top-left origin), computes tight bounding boxes around landmarks, and writes `train.xml` / `test.xml`.

**Optional:** Add `--normalize` to apply PCA orientation normalization, which aligns specimens to a canonical orientation before training. Normalized images are saved to `images_normalized/` and rotation metadata to JSON files.

### Step 2: Generate YOLO Bounding Boxes

Replace the tight landmark-derived bboxes with YOLO detection boxes so the shape predictor trains on realistic inference-time bounding boxes:

```bash
uv run --no-sync python scripts/preprocessing/generate_yolo_bbox_xml.py \
  --config configs/toe_training_yolo_bbox.yaml
```

This runs the pre-trained YOLO model on each image, matches detections to ground-truth landmarks by centroid proximity, and replaces the XML bounding boxes with YOLO boxes + configurable padding. Falls back to landmark-derived boxes with extra padding when YOLO misses a detection.

**OBB variant:** When using an OBB YOLO model (`toe_training_yolo_obb.yaml`), the script crops and rotates each image around the oriented bounding box, transforms ground-truth landmarks into the rotated crop coordinate space, and writes tighter upright bounding boxes.

### Step 3: Train Shape Predictor

Train the dlib cascade regression tree shape predictor:

```bash
uv run --no-sync python shape_trainer.py \
  -d train_yolo_bbox.xml \
  -t test_yolo_bbox.xml \
  -o toe_predictor_yolo_bbox \
  -th 8
```

Outputs `toe_predictor_yolo_bbox.dat` (the trained model) and reports average pixel error on train and test sets.

### Step 4: Inference

At inference time, the two stages run in sequence:
1. YOLO detects the region of interest in new images
2. The dlib shape predictor predicts landmarks within each YOLO bounding box

For the legacy dlib-only pipeline (which uses a HOG+SVM detector instead of YOLO):
```bash
python prediction.py \
  -i test_images \
  -d detector.svm \
  -p predictor.dat \
  -o predictions.xml
```

### Step 5: Evaluate

```bash
# Test shape predictor accuracy
python shape_tester.py -t test_yolo_bbox.xml -p toe_predictor_yolo_bbox.dat
```

## Configuration

### YOLO Bbox Config (`yolo_bbox:` section)

| Parameter | Default | Description |
|---|---|---|
| `yolo_model` | — | Path to pre-trained YOLO weights |
| `padding_ratio` | 0.3 | Padding added around YOLO detection box |
| `fallback_padding_ratio` | 0.5 | Padding when YOLO misses a detection |
| `conf_threshold` | 0.25 | YOLO confidence threshold |
| `target_class` | — | Class to match (e.g., `toe`, `finger`) |

### Shape Predictor Config (`shape:` section)

| Parameter | Default | Description |
|---|---|---|
| `threads` | 8 | CPU threads |
| `tree_depth` | 4 | Depth of each regression tree |
| `cascade_depth` | 15 | Number of cascades |
| `nu` | 0.1 | Regularization parameter |
| `oversampling` | 20 | Training oversampling amount |
| `test_splits` | 20 | Splits per tree node |
| `feature_pool_size` | 500 | Features sampled per split |
| `num_trees` | 500 | Trees per cascade level |

## Utilities

### Preprocessing (`scripts/preprocessing/`)

```bash
# Consolidate multiple TPS files into one
python scripts/preprocessing/consolidate_all_tps.py

# Merge specific TPS files
python scripts/preprocessing/merge_tps_files.py

# Remove specific landmarks from a TPS file
python scripts/preprocessing/remove_landmarks_from_tps.py

# Extract scale information from TPS files
python scripts/preprocessing/extract_scale_tps.py

# Generate oriented bounding boxes from TPS landmarks
python scripts/preprocessing/generate_obb_from_tps.py

# Split dlib XML into train/validation/test sets (for hyperparameter tuning)
python scripts/preprocessing/split_train_val_test.py \
  --train-xml train.xml --test-xml test.xml --val-ratio 0.2
```

### Training (`scripts/training/`)

```bash
# Grid search over shape predictor hyperparameters
python scripts/training/hyperparameter_search.py --config configs/toe_training_yolo_bbox.yaml

# Visualize hyperparameter search results
python scripts/plot_hyperparam_results.py --results-dir hyperparam_results/
```

## Directory Structure

```
ml-morph/
├── preprocessing.py          # TPS→XML conversion, train/test split
├── detector_trainer.py       # Train HOG+SVM detector (legacy)
├── detector_tester.py        # Evaluate detector (legacy)
├── shape_trainer.py          # Train dlib shape predictor
├── shape_tester.py           # Evaluate shape predictor
├── prediction.py             # Run inference (legacy dlib-only)
├── configs/
│   ├── toe_training.yaml               # Legacy dlib-only config
│   ├── toe_training_yolo_baseline.yaml  # YOLO baseline bbox config
│   ├── toe_training_yolo_bbox.yaml      # YOLO bbox + padding config
│   ├── toe_training_yolo_obb.yaml       # YOLO OBB config
│   ├── finger_training_yolo_*.yaml      # Finger variants
│   └── default.yaml                     # Example config
├── scripts/
│   ├── train_workflow.py     # Unified dlib-only workflow
│   ├── evaluate.py           # Evaluation script
│   ├── preprocessing/        # TPS and bbox utilities
│   └── training/             # Hyperparameter search
└── utils/                    # Shared helpers (XML/TPS I/O)
```

## Integration with Main Project

ml-morph is **Stage 2** of the full pipeline described in the [root README](../README.md):

```
Raw Images → YOLO Detection (Stage 1) → Cropped Regions → ml-morph Landmarks (Stage 2)
```

The YOLO model is trained separately (see root project). Its detections feed into ml-morph via `generate_yolo_bbox_xml.py`, which substitutes YOLO bounding boxes into the training XML so the shape predictor learns to work with YOLO-sized boxes.

## Troubleshooting

**"No module named 'dlib'"** — Install dlib: `uv pip install dlib`. May require `cmake` and `boost` as system dependencies.

**High landmark error at inference** — Likely a bbox mismatch. The shape predictor must be trained on the same bounding box distribution it sees at inference. If using YOLO detections at inference, train with a `*_yolo_bbox.yaml` or `*_yolo_obb.yaml` config, not the legacy tight-bbox config.

**YOLO misses detections** — The `generate_yolo_bbox_xml.py` script falls back to landmark-derived boxes with `fallback_padding_ratio` padding. Check `conf_threshold` if too many detections are missed.
