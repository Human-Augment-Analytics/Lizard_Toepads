# Preprocessing Augmentation Pipeline Usage

## Overview

This pipeline implements **preprocessing-time vertical flip augmentation** for bilateral toepad detection. Unlike training-time augmentation (H2_bilateral), this approach generates augmented images during the preprocessing stage, resulting in:

- ✅ **Doubled training data** (original + flipped images)
- ✅ **Faster training** (no real-time augmentation overhead)
- ✅ **Deterministic augmentation** (100% coverage vs 50% random)
- ✅ **No data leakage** (grouped splitting ensures base image and augmented image stay together)

## File Structure

### New Files Created

1. **`configs/H2_bilateral_preprocessed.yaml`**
   - Configuration for preprocessing augmentation pipeline
   - Key settings:
     - `preprocessing.flipud-augmentation: true`
     - `split.group-by-suffix: _flipud`
     - No training-time augmentation parameters

2. **`scripts/preprocessing/process_tps_files_flipud.py`**
   - Preprocessing script with vertical flip augmentation
   - Generates: `001.jpg` → `001.jpg` + `001_flipud.jpg`
   - Automatically transforms label coordinates

3. **`scripts/preprocessing/split_dataset.py`** (modified)
   - Added group-aware splitting
   - Ensures `001.jpg` and `001_flipud.jpg` stay in the same split
   - Prevents data leakage between train/val sets

## Complete Pipeline

### Step 1: Preprocessing with Augmentation

```bash
uv run scripts/preprocessing/process_tps_files_flipud.py \
  --config configs/H2_bilateral_preprocessed.yaml
```

**What it does:**
- Reads TPS files and images from source directories
- Generates original processed images in `data/processed_augmented/images/`
- Generates vertically flipped images with `_flipud` suffix
- Creates corresponding labels with flipped coordinates (`center_y = 1.0 - center_y`)
- Expected output: ~2x images (e.g., 100 → 200)

**Output structure:**
```
data/processed_augmented/
├── images/
│   ├── 001.jpg
│   ├── 001_flipud.jpg
│   ├── 002.jpg
│   ├── 002_flipud.jpg
│   └── ...
├── labels/
│   ├── 001.txt
│   ├── 001_flipud.txt
│   ├── 002.txt
│   ├── 002_flipud.txt
│   └── ...
└── visualizations/
    ├── marked_001.jpg
    ├── marked_001_flipud.jpg
    └── ...
```

### Step 2: Group-Aware Dataset Split

```bash
uv run scripts/preprocessing/split_dataset.py \
  --config configs/H2_bilateral_preprocessed.yaml
```

**What it does:**
- Groups images by base name (removes `_flipud` suffix)
- Shuffles **groups** (not individual images)
- Splits groups into train (80%) and val (20%)
- Creates symlinks in `data/dataset_augmented/`

**Critical feature:** Prevents data leakage
- If `001.jpg` is in train → `001_flipud.jpg` is also in train
- If `100.jpg` is in val → `100_flipud.jpg` is also in val

**Output structure:**
```
data/dataset_augmented/
├── images/
│   ├── train/
│   │   ├── 001.jpg -> ../../processed_augmented/images/001.jpg
│   │   ├── 001_flipud.jpg -> ../../processed_augmented/images/001_flipud.jpg
│   │   └── ...
│   └── val/
│       ├── 100.jpg -> ../../processed_augmented/images/100.jpg
│       ├── 100_flipud.jpg -> ../../processed_augmented/images/100_flipud.jpg
│       └── ...
└── labels/
    ├── train/
    └── val/
```

### Step 3: Training

```bash
uv run scripts/training/train_yolo_aug.py \
  --config configs/H2_bilateral_preprocessed.yaml
```

**What it does:**
- Trains YOLOv11n on augmented dataset
- **No training-time augmentation** (data already augmented)
- Faster training compared to H2_bilateral
- Results saved to `runs/detect/H2_bilateral_preprocessed/`

## Comparison with H2_bilateral

| Aspect | H2_bilateral | H2_bilateral_preprocessed |
|--------|--------------|---------------------------|
| **Augmentation timing** | Training-time | Preprocessing-time |
| **Data quantity** | Same | Doubled |
| **Augmentation probability** | 50% (random) | 100% (deterministic) |
| **Training speed** | Slower (real-time flip) | Faster (pre-generated) |
| **Disk usage** | Low | 2x (doubled images) |
| **Data leakage risk** | N/A | Prevented by grouping |
| **Config file** | `configs/H2_bilateral.yaml` | `configs/H2_bilateral_preprocessed.yaml` |
| **Output directory** | `data/dataset/` | `data/dataset_augmented/` |
| **Model name** | `H2_bilateral` | `H2_bilateral_preprocessed` |

## Quick Start

Run all steps sequentially:

```bash
# 1. Preprocess with vertical flip augmentation
uv run scripts/preprocessing/process_tps_files_flipud.py \
  --config configs/H2_bilateral_preprocessed.yaml

# 2. Split dataset with grouping
uv run scripts/preprocessing/split_dataset.py \
  --config configs/H2_bilateral_preprocessed.yaml

# 3. Train model
uv run scripts/training/train_yolo_aug.py \
  --config configs/H2_bilateral_preprocessed.yaml
```

## Validation

### After Preprocessing
```bash
# Check number of generated images
find data/processed_augmented/images -name "*.jpg" | wc -l
# Should be ~2x the number of original images

# Verify flipped images exist
ls data/processed_augmented/images/*_flipud.jpg | head -5
```

### After Split
```bash
# Check grouping worked correctly
# Example: if 001.jpg is in train, 001_flipud.jpg should also be in train
ls data/dataset_augmented/images/train/001*
# Expected: 001.jpg, 001_flipud.jpg (both present or both absent)

# Check split ratio
echo "Train images:" $(find data/dataset_augmented/images/train -name "*.jpg" | wc -l)
echo "Val images:" $(find data/dataset_augmented/images/val -name "*.jpg" | wc -l)
```

### Verify Label Transformation
```bash
# Check that flipped labels have inverted center_y
echo "Original label (001.txt):"
cat data/processed_augmented/labels/001.txt

echo "Flipped label (001_flipud.txt):"
cat data/processed_augmented/labels/001_flipud.txt

# center_y values should be: flipped_y ≈ 1.0 - original_y
```

## Troubleshooting

### Issue: No flipped images generated
**Solution:** Check `preprocessing.flipud-augmentation: true` in config

### Issue: Data leakage (001.jpg in train, 001_flipud.jpg in val)
**Solution:** Check `split.group-by-suffix: _flipud` in config

### Issue: Different number of images/labels
**Solution:** Check preprocessing logs for errors, ensure all TPS files exist

## Configuration Parameters

### Key Config Settings (H2_bilateral_preprocessed.yaml)

```yaml
preprocessing:
  flipud-augmentation: true          # Enable vertical flip
  output-dir: data/processed_augmented

split:
  images-dir: data/processed_augmented/images
  labels-dir: data/processed_augmented/labels
  output-dir: data/dataset_augmented
  group-by-suffix: _flipud           # Group augmented images

train:
  name: H2_bilateral_preprocessed
  # No augmentation parameters (already augmented in preprocessing)

dataset:
  path: data/dataset_augmented
```

## Expected Results

After training, compare results:

```bash
# H2_bilateral (training-time augmentation)
tensorboard --logdir runs/detect/H2_bilateral

# H2_bilateral_preprocessed (preprocessing-time augmentation)
tensorboard --logdir runs/detect/H2_bilateral_preprocessed
```

**Metrics to compare:**
- mAP@0.5
- Precision/Recall
- Training time per epoch
- Bilateral toepad detection accuracy (upper vs lower)

## Next Steps

1. ✅ Run preprocessing pipeline
2. ✅ Verify data integrity and grouping
3. ✅ Train model
4. 🔄 Compare results with H2_bilateral
5. 🔄 Evaluate bilateral detection performance
