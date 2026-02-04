# ml-morph - PyTorch Keypoint Detection

Modern deep learning approach for landmark-based morphometrics using PyTorch Lightning.

## Overview

This is the **PyTorch alternative** to ml-morph's classic dlib workflow. It uses ResNet18 with transfer learning for direct keypoint regression.

**When to use PyTorch over dlib:**
- You have GPU/MPS acceleration available
- You need better accuracy and are willing to trade compute time
- You want to leverage transfer learning from ImageNet
- You have larger datasets (100+ images)

**When to use dlib instead:**
- CPU-only environment
- Smaller datasets
- Need proven, stable method
- Prefer traditional computer vision approaches

## Installation

From the repository root:

```bash
# Install base dependencies
uv sync

# Install PyTorch (choose based on your hardware)
# Mac (MPS):
uv pip install --python .venv torch torchvision torchaudio

# Linux/Windows with CUDA 12.4:
uv pip install --python .venv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU-only:
uv pip install --python .venv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Optional: TensorBoard for visualization
uv pip install tensorboard
```

## Quick Start

```bash
cd ml-morph

# Complete workflow: preprocessing + training + evaluation
uv run --no-sync python scripts/train_pytorch_workflow.py --config configs/toe_pytorch.yaml
```

## PyTorch Workflow

### 1. Prepare Data

You need:
- Images (e.g., `train/*.jpg`)
- TPS file with landmarks (e.g., `consolidated_toe.tps`)

### 2. Configure Training

Edit `configs/toe_pytorch.yaml`:

```yaml
preprocessing:
  tps_file: consolidated_toe.tps
  image_dir: train
  train_ratio: 0.8

data:
  image_size: 256
  num_workers: 4
  augment: true
  persistent_workers: true

model:
  pretrained: true
  lr: 0.0001
  weight_decay: 0.0001

trainer:
  batch_size: 16
  max_epochs: 50
  accelerator: auto  # MPS on Mac, CUDA on GPU, CPU otherwise
```

### 3. Run Training

```bash
# Complete workflow (preprocessing + training + evaluation)
uv run --no-sync python scripts/train_pytorch_workflow.py --config configs/toe_pytorch.yaml

# Skip preprocessing if XML files exist
uv run --no-sync python scripts/train_pytorch_workflow.py --config configs/toe_pytorch.yaml --skip-preprocessing

# Skip evaluation (only train)
uv run --no-sync python scripts/train_pytorch_workflow.py --config configs/toe_pytorch.yaml --skip-evaluation
```

The workflow will:
1. Convert TPS → XML annotations
2. Train ResNet18 keypoint detector
3. Evaluate on both training and test sets
4. Save checkpoints to `runs/keypoints/`

### 4. Monitor Training (Optional)

```bash
# In a separate terminal
uv run tensorboard --logdir runs/keypoints/

# Open browser to http://localhost:6006
```

### 5. Evaluate Results

**Option 1: Automatic evaluation during training**
```bash
# Evaluation runs automatically after training
uv run --no-sync python scripts/train_pytorch_workflow.py --config configs/toe_pytorch.yaml
```

**Option 2: Standalone evaluation**
```bash
# Evaluate after training completes (auto-finds best checkpoint)
uv run --no-sync python scripts/evaluate.py --config configs/toe_pytorch.yaml

# Evaluate specific checkpoint
uv run --no-sync python scripts/evaluate.py \
  --config configs/toe_pytorch.yaml \
  --checkpoint runs/keypoints/toe/lightning_logs/version_0/checkpoints/best.ckpt

# Evaluate only test set
uv run --no-sync python scripts/evaluate.py --config configs/toe_pytorch.yaml --test-only
```

The evaluation compares:
- Training set performance (to check for underfitting)
- Test set performance (actual generalization)
- Per-keypoint error breakdown

### 6. Run Predictions on New Images

```bash
# Find the best checkpoint (lowest validation loss)
uv run --no-sync python pytorch_keypoint/predict_keypoints.py \
  --config configs/toe_pytorch.yaml \
  --checkpoint runs/keypoints/toe/lightning_logs/version_0/checkpoints/best.ckpt \
  --xml your_images.xml \
  --output predictions.csv

# Evaluate against ground truth
uv run --no-sync python pytorch_keypoint/evaluate_predictions.py \
  --csv predictions.csv \
  --xml ground_truth.xml \
  --root .
```

## Model Architecture

- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Output**: Flattened (x, y) coordinates for all keypoints
- **Loss**: Smooth L1 Loss
- **Optimizer**: AdamW with Cosine Annealing LR

## Configuration Options

### Data Config
- `image_size`: Input image size (default: 256)
- `num_workers`: Data loading workers (default: 4)
- `augment`: Enable data augmentation (default: true)
- `persistent_workers`: Keep workers alive between epochs (default: true)

### Model Config
- `pretrained`: Use ImageNet pretrained weights (default: true)
- `lr`: Learning rate (default: 1e-4)
- `weight_decay`: L2 regularization (default: 1e-4)
- `freeze_backbone_until`: Freeze first N layers (optional)

### Trainer Config
- `batch_size`: Batch size (default: 16)
- `max_epochs`: Training epochs (default: 50)
- `accelerator`: Hardware accelerator (auto/mps/cuda/cpu)
- `precision`: Training precision ("32" or "16-mixed")

## Troubleshooting

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

### Import Errors
Make sure PyTorch Lightning is installed:
```bash
uv pip install pytorch-lightning
```

## Performance Comparison

Example results on toe dataset (9 landmarks, 160 training images):

| Method | Training Error | Test Error | Training Time |
|--------|---------------|------------|---------------|
| **PyTorch** | 29.9 pixels | 72.9 pixels | ~10 min (MPS) |
| **dlib** | Variable | Variable | ~30-60 min (CPU) |

PyTorch shows faster training with GPU but may overfit on small datasets. The generalization gap (training vs test error) can be reduced with more data or stronger regularization.

## Citation

If you use ml-morph in your research:

```
Porto, A. and Voje, K.L., 2020. ML‐morph: A fast, accurate and general
approach for automated detection and landmarking of biological structures
in images. Methods in Ecology and Evolution, 11(4), pp.500-512.
```
