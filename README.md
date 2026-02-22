# Lizard Toepad Detection Pipeline

**Authors**: Dylan Herbig, Junling Zhuang, Leyang Shen
**Cluster**: Georgia Tech PACE/ICE (also applicable to Phoenix and Hive)

## Overview

A two-stage pipeline for lizard toepad analysis:

1. **Stage 1 — YOLO Detection**: Detect toepad regions (fingers, toes, ruler, ID) using YOLOv11, with support for both standard bounding boxes (`detect`) and oriented bounding boxes (`obb`).
2. **Stage 2 — ml-morph Landmark Prediction**: Crop detected regions and predict anatomical landmarks using dlib shape predictors. See [docs/ML_MORPH_PIPELINE.md](docs/ML_MORPH_PIPELINE.md).

### Class Mapping (6 classes, shared across detect and OBB)

| ID | Class      | Source              |
|----|------------|---------------------|
| 0  | up_finger  | Upper view dataset  |
| 1  | up_toe     | Upper view dataset  |
| 2  | bot_finger | TPS landmark files  |
| 3  | bot_toe    | TPS landmark files  |
| 4  | ruler      | TPS landmark files  |
| 5  | id         | TPS landmark files  |

---

## Prerequisites

- Active PACE account with GPU allocation access
- Python 3.10+ (tested with 3.13)
- [uv](https://docs.astral.sh/uv/) package manager

### Data Sources (ICE Cluster)

| Data | Path |
|------|------|
| Images | `/storage/ice-shared/cs8903onl/miami_fall_24_jpgs/` |
| TPS Landmarks | `/storage/ice-shared/cs8903onl/tps_files/` |
| Upper View Dataset | `/storage/ice-shared/cs8903onl/miami_fall_24_upper_dataset_roboflow/` |

---

## Setup

### 1. Install uv and dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~/Lizard_Toepads
uv sync
```

### 2. Install PyTorch with CUDA

```bash
# CUDA 12.4 (must rerun after every uv sync)
uv pip install --python .venv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU-only alternative:
# uv pip install --python .venv torch torchvision torchaudio
```

> **Note**: PyTorch is excluded from `pyproject.toml` to avoid CPU/GPU version conflicts. After `uv sync`, rerun the CUDA install command. When using `uv run`, either pass `--no-sync` or set `UV_NO_SYNC=1` to prevent overwriting CUDA wheels.

### 3. Download pre-trained models

```bash
uv run python scripts/utils/download_models.py
```

This downloads all YOLOv11 detect + OBB models to `models/base_models/`.

### 4. Using shared data on PACE (optional)

```bash
# Symlink shared data (saves quota)
mkdir -p ~/Lizard_Toepads/data
ln -s /storage/ice-shared/cs8903onl/miami_fall_24_jpgs ~/Lizard_Toepads/data/miami_fall_24_jpgs
ln -s /storage/ice-shared/cs8903onl/tps_files ~/Lizard_Toepads/data/tps_files
```

---

## Pipeline Overview

### Shared Steps

1. Install dependencies and download models (see [Setup](#setup))
2. Generate bottom view labels from TPS landmarks

```bash
uv run python scripts/preprocessing/detect/generate_bottom_view_labels.py --config configs/H5.yaml
```

### YOLO Detect Pipeline

```bash
# 3a. Merge upper + bottom views (6-class bbox)
uv run python scripts/preprocessing/detect/merge_upper_bottom_views.py --config configs/H5.yaml

# 4a. Create train/val split
uv run python scripts/preprocessing/create_train_val_split.py --config configs/H5.yaml  # shared

# 5a. Allocate GPU and train
salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:H200:1
srun --pty bash

uv run python scripts/training/train_yolo.py --config configs/H5.yaml

# 6a. Inference
uv run python scripts/inference/predict.py --config configs/H5.yaml --quick-test
```

### YOLO OBB Pipeline

```bash
# 3b. Generate OBB labels from TPS
uv run python scripts/preprocessing/obb/generate_obb_from_tps.py --config configs/H8_obb_noflip.yaml

# 4b. Create merged OBB dataset (bbox → OBB conversion)
uv run python scripts/preprocessing/obb/create_merged_obb_dataset.py --config configs/H8_obb_noflip.yaml

# 5b. Create no-flip OBB dataset (bottom-only, fair comparison against detect baseline)
uv run python scripts/preprocessing/obb/create_noflip_obb_dataset.py --config configs/H8_obb_noflip.yaml

# 6b. Allocate GPU and train
salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:H200:1
srun --pty bash

uv run --no-sync python scripts/training/train_yolo.py --config configs/H8_obb_noflip.yaml

# 7b. Inference (flip strategy for upper-view detection)
uv run --no-sync python scripts/inference/inference_with_flip.py --config configs/H8_obb_noflip.yaml --source data/images/
```

### Hyperparameter Tuning (either pipeline)

```bash
# Detect tuning
uv run python scripts/tuning/tune_hyperparams.py --config configs/H5.yaml --num-samples 50

# OBB tuning
uv run python scripts/tuning/tune_hyperparams.py --config configs/H8_obb_noflip.yaml --num-samples 50
```

### Stage 2: ml-morph (Landmark Prediction)

After YOLO detections, crop regions and run dlib shape predictors:
```bash
cd ml-morph
# See docs/ML_MORPH_PIPELINE.md for full instructions
```

---

## Configuration

All training parameters live in YAML configs under `configs/`. The training script passes every key in the `train:` section directly to YOLO — **adding a new parameter requires zero code changes**, just add it to the YAML.

### Available Configs

| Config | Task | Description |
|--------|------|-------------|
| `H5.yaml` | detect | Bilateral detection with augmentation |
| `H6.yaml` | detect | H5 + Ray Tune best hyperparameters |
| `H7_obb_6class.yaml` | obb | OBB with merged upper+bottom views |
| `H8_obb_noflip.yaml` | obb | OBB bottom-only (fair baseline comparison) |

### Config Structure

```yaml
train:
  task: detect          # or obb
  model: yolo11m.pt     # auto-downloaded by YOLO
  epochs: 300
  batch: 32
  imgsz: 1280
  # ... any YOLO train() parameter works here

dataset:
  path: data/dataset
  train: images/train
  val: images/val
  nc: 6
  names: ["up_finger", "up_toe", "bot_finger", "bot_toe", "ruler", "id"]

inference:
  conf: 0.2
  iou: 0.2
```

---

## Model Selection Guide

This project uses two model families from [Ultralytics](https://docs.ultralytics.com/):

- **[YOLOv11](https://docs.ultralytics.com/models/yolo11/) (detect)** — standard axis-aligned bounding boxes. Good when objects are roughly upright.
- **[YOLO26](https://docs.ultralytics.com/models/yolo26/) (OBB)** — oriented (rotated) bounding boxes with the latest architecture. NMS-free end-to-end inference, ~43% faster CPU speed vs YOLO11.

### YOLOv11 Detection Models (`task: detect`)

Used by configs: `H5.yaml`, `H6.yaml`

| Model | Filename | Params | Speed | Use Case |
|-------|----------|--------|-------|----------|
| YOLOv11n | yolov11n.pt | 2.6M | Fastest | Quick experiments |
| YOLOv11s | yolov11s.pt | 9.4M | Fast | Good balance |
| **YOLOv11m** | **yolov11m.pt** | 20.1M | Medium | **Recommended** |
| YOLOv11l | yolov11l.pt | 25.3M | Slow | High accuracy |
| YOLOv11x | yolov11x.pt | 56.9M | Slowest | Maximum accuracy |

### YOLO26-OBB Models (`task: obb`)

Used by configs: `H7_obb_6class.yaml`, `H8_obb_noflip.yaml`

| Model | Filename | Params | Speed | Use Case |
|-------|----------|--------|-------|----------|
| YOLO26n-OBB | yolo26n-obb.pt | 2.5M | Fastest | Quick experiments |
| YOLO26s-OBB | yolo26s-obb.pt | 9.0M | Fast | Good balance |
| **YOLO26m-OBB** | **yolo26m-obb.pt** | 19.3M | Medium | **Recommended** |
| YOLO26l-OBB | yolo26l-obb.pt | 24.4M | Slow | High accuracy |
| YOLO26x-OBB | yolo26x-obb.pt | 57.6M | Slowest | Maximum accuracy |

> **Why different versions?** YOLO26 is the latest Ultralytics architecture with NMS-free inference and improved speed. We use it for OBB where the new rotation head benefits most. Detect stays on YOLOv11 which has proven results on our dataset. Both are swappable via config — just change `model:` in the YAML.
>
> **When to use OBB?** If toepad specimens are scanned at various angles, OBB produces tighter bounding boxes and cleaner crops for downstream landmark prediction. See [docs/COMPARISON_BASELINE_VS_OBB.md](docs/COMPARISON_BASELINE_VS_OBB.md) for a quantitative comparison.

---

## SLURM Allocation Examples

```bash
# Single H200 GPU (recommended for training)
salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:H200:1

# Single A100 GPU
salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:A100:1

# Multi-GPU for hyperparameter tuning
salloc -N1 --ntasks-per-node=8 -t12:00:00 --gres=gpu:H200:4
```

See `sbatch/` for pre-built SLURM batch scripts.

---

## Project Structure

```
Lizard_Toepads/
├── configs/                          # YOLO training configs (H5, H6, H7, H8)
├── scripts/
│   ├── preprocessing/
│   │   ├── detect/                   # Detect pipeline: TPS→bbox, merge views
│   │   ├── obb/                      # OBB pipeline: TPS→OBB, merge datasets
│   │   ├── create_train_val_split.py # Shared: train/val split
│   │   └── consolidate_tps_by_category.py
│   ├── training/                     # train_yolo.py (unified detect + OBB)
│   ├── inference/                    # predict.py, predict_bilateral.py, inference_with_flip.py
│   ├── tuning/                       # tune_hyperparams.py (Ray Tune + Optuna)
│   ├── visualization/
│   └── utils/                        # download_models.py, setup_gpu.py, extract_id
├── ml-morph/                         # Stage 2: dlib landmark prediction (self-contained)
├── sbatch/                           # SLURM batch scripts
├── docs/                             # Additional documentation
├── data/                             # Datasets (gitignored)
├── models/                           # Pre-trained models (gitignored)
└── runs/                             # Training outputs (gitignored)
```

---

## Additional Documentation

- [Baseline vs OBB Comparison](docs/COMPARISON_BASELINE_VS_OBB.md)
- [OBB Crop & Rotate Experiment](docs/EXPERIMENT_CROP_ROTATE_OBB.md)
- [Inference with Flip Strategy](docs/INFERENCE_WITH_FLIP.md)
- [ml-morph Pipeline](docs/ML_MORPH_PIPELINE.md)
- [Step-by-Step Experiments](docs/RUN_EXPERIMENTS_STEP_BY_STEP.md)

## Resources

- [PACE Documentation](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096)
- [Ultralytics YOLO](https://www.ultralytics.com/yolo)
