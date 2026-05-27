# Lizard Toepads

**[🚀 Newcomer? Read the Guide to Lizard Toepads & LizardMorph here!](docs/NEWCOMER_GUIDE.md)**

**Authors**: Dylan Herbig, Junling Zhuang, Leyang Shen
**Cluster**: Georgia Tech PACE/ICE (also applicable to Phoenix and Hive)
**Last Updated**: Junling Zhuang, 2026-03-06

## Overview

A two-stage pipeline for lizard toepad analysis:

1. **Stage 1 — YOLO Detection**: Detect toepad regions (fingers, toes, ruler, ID) using YOLOv11, with support for both standard bounding boxes (`detect`) and oriented bounding boxes (`obb`).
2. **Stage 2 — ml-morph Landmark Prediction**: Crop detected regions and predict anatomical landmarks using dlib shape predictors. See [docs/ML_MORPH_PACE_EXPERIMENTS.md](docs/ML_MORPH_PACE_EXPERIMENTS.md).

### Class Mapping

**OBB pipeline (H10+) — 4 classes:**

| ID | Class   | Source             |
|----|---------|--------------------|
| 0  | finger  | TPS landmark files |
| 1  | toe     | TPS landmark files |
| 2  | ruler   | TPS landmark files |
| 3  | id      | TPS landmark files |

**Detect pipeline (H5) — 6 classes (legacy):**

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

### YOLO OBB Pipeline (4-class)

```bash
# 3b. Generate OBB labels + resized images from TPS (--visualize N to verify)
uv run python scripts/preprocessing/obb/generate_obb_from_tps.py --config configs/H10_obb.yaml --visualize 10

# 4b. Create train/val split
uv run python scripts/preprocessing/create_train_val_split.py --config configs/H10_obb.yaml

# 5b. Allocate GPU and train
salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:H200:1
srun --pty bash

uv run python scripts/training/train_yolo.py --config configs/H10_obb.yaml

# 6b. Inference
uv run python scripts/inference/predict.py --config configs/H10_obb.yaml --quick-test
```

<details>
<summary>Legacy OBB Pipeline (6-class, H8/H9)</summary>

```bash
uv run python scripts/preprocessing/obb/generate_obb_from_tps.py --config configs/H8_obb_botonly.yaml --visualize 10
uv run python scripts/preprocessing/obb/create_merged_obb_dataset.py --config configs/H8_obb_botonly.yaml
uv run python scripts/preprocessing/obb/create_noflip_obb_dataset.py --config configs/H8_obb_botonly.yaml --visualize 10
uv run python scripts/preprocessing/create_train_val_split.py --config configs/H8_obb_botonly.yaml
uv run python scripts/training/train_yolo.py --config configs/H8_obb_botonly.yaml
uv run python scripts/inference/inference_with_flip.py --config configs/H8_obb_botonly.yaml --quick-test
```

</details>

### Publish Model to GitHub Releases

```bash
# Dry run (preview release notes and assets)
uv run python scripts/deployment/publish_release.py --config configs/H11_obb.yaml --version v1.0.0-obb --dry-run

# Publish (uploads best.pt, best_fp16.onnx, best_fp32.onnx, metadata.json)
uv run python scripts/deployment/publish_release.py --config configs/H11_obb.yaml --version v1.0.0-obb
```

Versioning: `v{MAJOR}.{MINOR}.{PATCH}-{task}` — MAJOR for architecture changes, MINOR for training improvements, PATCH for export changes.

### Hyperparameter Tuning (either pipeline)

```bash
# Detect tuning
uv run python scripts/tuning/tune_hyperparams.py --config configs/H5.yaml --num-samples 50

# OBB tuning
uv run python scripts/tuning/tune_hyperparams.py --config configs/H8_obb_botonly.yaml --num-samples 50
```

### Stage 2: ml-morph (Landmark Prediction)

After YOLO detections, crop regions and run dlib shape predictors:
```bash
cd ml-morph
# See docs/ML_MORPH_PACE_EXPERIMENTS.md for full instructions
```

---

## Configuration

All training parameters live in YAML configs under `configs/`. The training script passes every key in the `train:` section directly to YOLO — **adding a new parameter requires zero code changes**, just add it to the YAML.

### Available Configs

| Config | Task | Classes | Description |
|--------|------|---------|-------------|
| **`H10_obb.yaml`** | **obb** | **4** | **OBB 4-class (finger, toe, ruler, id)** |
| `H5.yaml` | detect | 6 | Bilateral detection with augmentation |
| `H6.yaml` | detect | 6 | H5 + Ray Tune best hyperparameters |
| `H7_obb_6class.yaml` | obb | 6 | OBB with merged upper+bottom views |
| `H8_obb_botonly.yaml` | obb | 6 | OBB bottom-only (legacy) |
| `H9_obb_botonly.yaml` | obb | 6 | H8 + stronger augmentation (legacy) |

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
  path: data/obb/dataset_4class_split
  train: images/train
  val: images/val
  nc: 4
  names: ["finger", "toe", "ruler", "id"]

inference:
  conf: 0.2
  iou: 0.2
```

---

## Model Selection Guide

This project uses two model families from [Ultralytics](https://docs.ultralytics.com/):

- **[YOLOv11](https://docs.ultralytics.com/models/yolo11/)** — used for both detect and OBB tasks. Proven architecture with strong results on our dataset.

### YOLOv11 Detection Models (`task: detect`)

Used by configs: `H5.yaml`, `H6.yaml`

| Model | Filename | Params | Speed | Use Case |
|-------|----------|--------|-------|----------|
| YOLOv11n | yolov11n.pt | 2.6M | Fastest | Quick experiments |
| YOLOv11s | yolov11s.pt | 9.4M | Fast | Good balance |
| **YOLOv11m** | **yolov11m.pt** | 20.1M | Medium | **Recommended** |
| YOLOv11l | yolov11l.pt | 25.3M | Slow | High accuracy |
| YOLOv11x | yolov11x.pt | 56.9M | Slowest | Maximum accuracy |

### YOLOv11-OBB Models (`task: obb`)

Used by configs: `H10_obb.yaml`, `H7_obb_6class.yaml`, `H8_obb_botonly.yaml`

| Model | Filename | Params | Speed | Use Case |
|-------|----------|--------|-------|----------|
| YOLOv11n-OBB | yolo11n-obb.pt | 2.7M | Fastest | Quick experiments |
| YOLOv11s-OBB | yolo11s-obb.pt | 9.6M | Fast | Good balance |
| **YOLOv11m-OBB** | **yolo11m-obb.pt** | 20.4M | Medium | **Recommended** |
| YOLOv11l-OBB | yolo11l-obb.pt | 25.5M | Slow | High accuracy |
| YOLOv11x-OBB | yolo11x-obb.pt | 57.5M | Slowest | Maximum accuracy |

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

- **[NEWCOMER_GUIDE.md](docs/NEWCOMER_GUIDE.md)** - Guide to Lizard Toepads & LizardMorph
- [Baseline vs OBB Comparison](docs/COMPARISON_BASELINE_VS_OBB.md)
- [OBB Crop & Rotate Experiment](docs/EXPERIMENT_CROP_ROTATE_OBB.md)
- [Inference with Flip Strategy](docs/INFERENCE_WITH_FLIP.md)
- [ml-morph Pipeline](docs/ML_MORPH_PACE_EXPERIMENTS.md)
- [Step-by-Step Experiments](docs/RUN_EXPERIMENTS_STEP_BY_STEP.md)
- [MLflow Deployment Plan](docs/PLAN_MLFLOW_DEPLOYMENT.md)

## Resources

- [PACE Documentation](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096)
- [Ultralytics YOLO](https://www.ultralytics.com/yolo)
