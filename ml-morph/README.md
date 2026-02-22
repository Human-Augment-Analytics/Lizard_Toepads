# ml-morph

Machine learning tools for landmark-based morphometrics

## Overview

ml-morph provides **two approaches** for automated landmarking:

1. **Classic dlib workflow** - Traditional HOG detectors + shape predictors (proven, stable)
2. **PyTorch Keypoint Detection** - Modern deep learning with ResNet (see [README_pytorch.md](README_pytorch.md))

This README covers the **classic dlib approach**, which is the original ml-morph method. Both workflows are integrated with the main project's `uv` workflow.

## Quick Start

```bash
cd ml-morph

# Complete workflow: preprocessing + detector + shape predictor
uv run --no-sync python scripts/train_workflow.py --config configs/toe_training.yaml
```

## Installation

Dependencies are managed by the parent project's `pyproject.toml`. From the repository root:

```bash
# Install all dependencies
uv sync

# Install dlib (required for classic workflow)
uv pip install dlib

# Note: dlib may require system dependencies (cmake, boost)
# See README_classic_dlib.md for installation troubleshooting
```

## Directory Structure

```
ml-morph/
├── preprocessing.py, detector_trainer.py, etc.  # Core dlib scripts (top-level)
├── configs/
│   ├── toe_training.yaml                        # Dlib workflow config
│   ├── toe_pytorch.yaml                         # PyTorch config
│   └── default.yaml                             # Example dlib config
├── scripts/
│   ├── train_workflow.py                        # Unified dlib workflow script
│   ├── train_pytorch_workflow.py                # PyTorch workflow script
│   ├── evaluate.py                              # PyTorch evaluation
│   ├── preprocessing/                           # TPS utilities
│   └── debug/                                   # Experimental scripts
├── pytorch_keypoint/                            # PyTorch Lightning module
└── utils/                                       # Utility functions
```

## Classic dlib Workflow

### 1. Prepare Data

You need:
- Images (e.g., `train/*.jpg`)
- TPS file with landmarks (e.g., `consolidated_toe.tps`)

### 2. Configure Training

Edit `configs/toe_training.yaml`:

```yaml
preprocessing:
  input_dir: train
  tps_file: consolidated_toe.tps
  train_ratio: 0.8

detector:
  dataset: train.xml
  test: test.xml
  output: toe_detector.svm
  n_threads: 7
  epsilon: 0.01
  c_param: 5

shape:
  dataset: train.xml
  test: test.xml
  output: toe_predictor.dat
  threads: 7
  tree_depth: 4
  cascade_depth: 15
  nu: 0.1
  oversampling: 20
```

### 3. Run Training

```bash
# Complete workflow (preprocessing + detector + shape predictor)
uv run --no-sync python scripts/train_workflow.py --config configs/toe_training.yaml

# Skip preprocessing if XML files exist
uv run --no-sync python scripts/train_workflow.py --config configs/toe_training.yaml --skip-preprocessing

# Skip inference
uv run --no-sync python scripts/train_workflow.py --config configs/toe_training.yaml --skip-inference
```

The workflow will:
1. Convert TPS → XML annotations
2. Train HOG + SVM object detector → `toe_detector.svm`
3. Train regression tree shape predictor → `toe_predictor.dat`
4. (Optional) Run inference on new images

### 4. Run Predictions

After training, use the models for inference:

```bash
# Run predictions on new images
python prediction.py \
  -i test_images \
  -d toe_detector.svm \
  -p toe_predictor.dat \
  -o predictions.xml
```

### 5. Test Models

```bash
# Test detector performance
python detector_tester.py -t test.xml -d toe_detector.svm

# Test shape predictor performance
python shape_tester.py -t test.xml -p toe_predictor.dat
```

## Model Architecture

### Object Detector (HOG + SVM)
- **Method**: Histogram of Oriented Gradients (HOG) with Support Vector Machine (SVM)
- **Purpose**: Detects and localizes the object in images
- **Output**: Bounding box around the object

### Shape Predictor (Regression Trees)
- **Method**: Ensemble of regression trees
- **Purpose**: Predicts precise landmark positions within detected bounding box
- **Output**: (x, y) coordinates for all keypoints

## Configuration Options

### Preprocessing Config
- `input_dir`: Directory containing images
- `tps_file`: Path to TPS landmark file
- `train_ratio`: Train/test split ratio (default: 0.8)

### Detector Config
- `n_threads`: Number of CPU threads (default: 7)
- `epsilon`: Insensitivity parameter (default: 0.01)
- `c_param`: Soft margin parameter C (default: 5)
- `upsample`: Upsample limit (default: 0)
- `symmetrical`: Whether objects are bilaterally symmetrical (default: false)
- `window_size`: Detection window size (optional)

### Shape Predictor Config
- `threads`: Number of CPU threads (default: 7)
- `tree_depth`: Regression tree depth (default: 4)
- `cascade_depth`: Number of cascades (default: 15)
- `nu`: Regularization parameter (default: 0.1)
- `oversampling`: Oversampling amount (default: 20)
- `test_splits`: Number of test splits (default: 20)
- `feature_pool_size`: Feature pool size (default: 500)
- `num_trees`: Number of trees (default: 500)

## PyTorch Alternative Workflow

For modern deep learning approach using ResNet, see the PyTorch workflow:

### Quick Start (PyTorch)

```bash
# Install PyTorch dependencies
uv pip install torch torchvision pytorch-lightning tensorboard

# Complete workflow: preprocessing + training + evaluation
uv run --no-sync python scripts/train_pytorch_workflow.py --config configs/toe_pytorch.yaml

# Skip individual steps as needed
uv run --no-sync python scripts/train_pytorch_workflow.py \
  --config configs/toe_pytorch.yaml \
  --skip-preprocessing  # If XML files already exist

# Standalone evaluation
uv run --no-sync python scripts/evaluate.py --config configs/toe_pytorch.yaml
```

### Config: `configs/toe_pytorch.yaml`

The config controls:
- **preprocessing**: TPS → XML conversion and train/test split
- **data**: Image size, augmentation, batch loading
- **model**: ResNet18 backbone, learning rate, regularization
- **trainer**: Batch size, epochs, GPU/MPS acceleration

**Advantages of PyTorch**:
- GPU/MPS acceleration (faster training)
- Transfer learning with pretrained ImageNet weights
- Modern deep learning techniques
- Better accuracy in many cases

**Disadvantages**:
- Requires more compute resources
- Larger model size
- More dependencies

For detailed documentation, see [README_pytorch.md](README_pytorch.md).

## Utilities

### TPS Manipulation
```bash
# Consolidate multiple TPS files
python scripts/preprocessing/consolidate_all_tps.py

# Merge TPS files
python scripts/preprocessing/merge_tps_files.py

# Remove specific landmarks
python scripts/preprocessing/remove_landmarks_from_tps.py
```

### Hyperparameter Search
```bash
python scripts/training/hyperparameter_search.py --config configs/toe_training.yaml
```

## Integration with Main Project

ml-morph is the **Stage 2** of the two-stage pipeline:

1. **Stage 1 (YOLO)**: Detect regions (fingers, toes, rulers) → Crop images
2. **Stage 2 (ml-morph)**: Predict precise landmarks on cropped regions

To use cropped regions from YOLO:
```bash
# After YOLO inference crops regions
cd ml-morph
uv run --no-sync python scripts/train_workflow.py --config configs/toe_training.yaml
```

## Troubleshooting

### Import Errors
If you see "No module named 'dlib'", you're trying to use the classic workflow. Either:
- Install dlib: `uv pip install dlib` (may be difficult)
- Use PyTorch workflow instead (recommended)

### GPU Not Detected
- **Mac**: Should use MPS automatically
- **Linux/Windows**: Check CUDA installation and PyTorch version

### Out of Memory
Reduce batch size or image size in config:
```yaml
data:
  image_size: 128  # Reduce from 256
trainer:
  batch_size: 8  # Reduce from 16
```