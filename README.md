# PACE Documentation Contribution Template

## Document Metadata
* **Authors**: Dylan Herbig, Junling Zhuang
* **Date Created**: 2025-07-18
* **Last Updated**: 2026-01-23 By Junling Zhuang
* **Applicable Clusters**: Only utilized on ICE, but also applicable for Phoenix and Hive

## Bilateral Toepad Detection Pipeline

### Overview
This documentation describes the setup and usage of a YOLOv8-based object detection pipeline used to identify lizard toepads from scanned images. The pipeline now supports **bilateral detection** by integrating:
1.  **Bottom View**: Processed from TPS landmark files.
2.  **Upper View**: Integrated from a separate annotated dataset (Roboflow).


This unified pipeline processes raw data, merges annotations, splits the dataset, and trains a generic YOLOv11 model.

### Data Sources (ICE Cluster)
The project relies on shared datasets available on the ICE cluster.

**Shared Directories:**
*   **Images**: `/storage/ice-shared/cs8903onl/miami_fall_24_jpgs/`
*   **TPS Landmarks**: `/storage/ice-shared/cs8903onl/tps_files/` (Includes `_finger.TPS`, `_toe.TPS`, and `_id.TPS`)
*   **Upper View Dataset**: `/storage/ice-shared/cs8903onl/miami_fall_24_upper_dataset_roboflow/`

**Note:** For local development or training, you can symlink these to your project directory (see "Using shared data on PACE").


### Prerequisites
- Required access/permissions:
  - Active PACE account
  - Allocate computational resources using `salloc`. See [Computational Resources](#computational-resources) for details.
- Software dependencies:
  - conda 
  - Python 3.10+ (tested with 3.13)
  - Ultralytics YOLOv11 (`pip install ultralytics`)
  - Pillow (`pip install pillow`)
- Storage requirements:
  - At least 5–10 GB for dataset preparation and model training
  - Fast local storage (e.g., SSD-backed node storage) is recommended
- Other prerequisites:
  - A labeled dataset (images + YOLO-style .txt annotations)
  - Conda environment setup

### Pipeline Standardization

#### Complete Preprocessing Flow (Reference: `configs/H4.yaml`)
```yaml
preprocessing:
  # Step 1: Raw data sources
  image-dir: /storage/ice-shared/cs8903onl/miami_fall_24_jpgs
  tps-dir: /storage/ice-shared/cs8903onl/tps_files

  # Step 2: Process bottom view with TPS landmarks
  bottom-view-processed-dir: data/processed_bottom
  
  # Step 3: Integrate upper view dataset
  addtional-upper-side-data-dir: /storage/ice-shared/cs8903onl/miami_fall_24_upper_dataset_roboflow/train
  
  # Step 4: Merge bilateral annotations
  merged-processed-dir: data/processed_merged
  
split:
  # Step 5: Train/val split
  images-dir: data/processed_merged/images
  labels-dir: data/processed_merged/labels
  output-dir: data/dataset
  train-ratio: 0.8
```

#### Pipeline Scripts
1. **generate_bottom_view_labels.py** - Converts TPS landmarks to YOLO labels (Bottom View)
2. **merge_upper_bottom_views.py** - Merges upper and bottom view annotations
3. **create_train_val_split.py** - Splits dataset into train/val sets

#### Key Integration: Upper View Dataset

**Class Mapping (6 classes total)**
```python
0: up_finger   # From upper view dataset
1: up_toe      # From upper view dataset
2: bot_finger  # From TPS processing
3: bot_toe     # From TPS processing
4: ruler       # From TPS processing
5: id          # From TPS processing
```

This enables bilateral detection by combining:
- **Bottom view** (TPS-based processing) - classes 2, 3, 4, 5
- **Upper view** (Roboflow annotations) - classes 0, 1

**Note**: Upper view dataset is available on ICE cloud storage at `/storage/ice-shared/cs8903onl/miami_fall_24_upper_dataset_roboflow/`. For local development, the dataset has been copied to `data/upper_dataset_roboflow/train/`.

### Configuration Guidelines

All future configurations should follow the **H4.yaml pattern**:
1. Define all preprocessing paths in `preprocessing` section
2. Specify merged output directory
3. Configure split parameters
4. Include dataset section with 6-class mapping

### Step-by-Step Instructions

### Step-by-Step Instructions

1. Install uv package manager:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"  # Add uv to PATH
   # Optional: Add to ~/.bashrc for permanent PATH update
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   ```

2. Install dependencies (uv will automatically create venv):
   ```bash
   cd ~/Lizard_Toepads
   uv sync  # Install all dependencies from pyproject.toml

   # Install PyTorch with CUDA (rerun after every uv sync)
   uv pip install --python .venv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

   # For CPU-only setups:
   uv pip install --python .venv torch torchvision torchaudio
   ```

   **Note**: PyTorch is not included in pyproject.toml to avoid CPU/GPU version conflicts. After each `uv sync`, rerun `uv pip install --python .venv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` to restore the CUDA build. On newer uv releases, `uv run` re-syncs the environment before executing; that will replace the CUDA wheels with the CPU build. When you need GPU support, either add `--no-sync` to `uv run ...` (or set `UV_NO_SYNC=1`) or call the interpreter directly via `.venv\Scripts\python` so the CUDA wheels stay in place.

3. Run the complete workflow:
   ```bash
   # 1. Generate Bottom View Labels (TPS → YOLO)
   uv run python scripts/preprocessing/generate_bottom_view_labels.py --config configs/H4.yaml

   # 2. Merge Upper and Bottom Views
   uv run python scripts/preprocessing/merge_upper_bottom_views.py --config configs/H4.yaml

   # 3. Create Train/Val Split
   uv run python scripts/preprocessing/create_train_val_split.py --config configs/H4.yaml

   # 4. Standard Allocation Command (H200 GPU)
   salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:H200:1

   # 5. Train YOLO (will auto-download YOLOv11s if needed)
   uv run python scripts/training/train_yolo_aug.py --config configs/H4.yaml

   # 6. Predict / Inference
   uv run python scripts/inference/predict.py --config configs/H4.yaml --quick-test
   ```

   **Note**: Pre-trained models are stored in `models/base_models/`. The training script automatically downloads YOLOv11n on first run if not present.


### Using shared data on PACE (optional)
- If you want to quickly experiment with the data and YOLO model on PACE, first follow this README to set up your environment on PACE, then bring the shared data into your project.
- Shared directories on ICE:
  - `/storage/ice-shared/cs8903onl/miami_fall_24_jpgs/`
  - `/storage/ice-shared/cs8903onl/tps_files/`

Option A — create symlinks (recommended to save quota):
```bash
mkdir -p ~/Lizard_Toepads/data
ln -s /storage/ice-shared/cs8903onl/miami_fall_24_jpgs ~/Lizard_Toepads/data/miami_fall_24_jpgs
ln -s /storage/ice-shared/cs8903onl/tps_files ~/Lizard_Toepads/data/tps_files
```

Option B — copy the data (uses your home/project quota):
```bash
mkdir -p ~/Lizard_Toepads/data
cp -r /storage/ice-shared/cs8903onl/miami_fall_24_jpgs ~/Lizard_Toepads/data/
cp -r /storage/ice-shared/cs8903onl/tps_files ~/Lizard_Toepads/data/
```

Notes:
- The inference example below can then use paths like `data/miami_fall_24_jpgs/1001.jpg`.
- If you organize a YOLO dataset split later, point your `data.yaml` to those `data/dataset/images` and `data/dataset/labels` folders accordingly.

### Validation

Pipeline successfully executed on 843 images:
```
Process:  843 images -> data/processed_bottom
Merge:    843 samples -> data/processed_merged (6 classes)
Split:    674 train / 169 val -> data/dataset
```

**Sample output** (data/processed_merged/labels/1001.txt):
```
0 0.358266 0.322761 0.045961 0.018611  # up_finger
1 0.507359 0.163987 0.032164 0.107369  # up_toe
2 0.359118 0.586415 0.034185 0.029258  # bot_finger
3 0.537083 0.688274 0.022763 0.112405  # bot_toe
4 0.027159 0.855338 0.054319 0.016750  # ruler
5 0.117353 0.104960 0.112912 0.098565  # id
```

### Model Selection Guide

#### Available YOLOv11 Models

| Model | Size | Speed | mAP | GPU Memory | Use Case |
|-------|------|-------|-----|------------|----------|
| yolov11n | 5.4MB | Fastest | Lower | 2-4GB | Quick experiments, small datasets |
| yolov11s | ~18MB | Fast | Good | 4-6GB | **Recommended balance** |
| yolov11m | ~40MB | Medium | Better | 6-10GB | Larger datasets |
| yolov11l | ~50MB | Slow | High | 10GB+ | High accuracy needs |
| yolov11x | ~110MB | Slowest | Highest | 12GB+ | Maximum accuracy |

#### Model Download

##### Download All Models at Once (Recommended)

```bash
# Download all YOLOv11 models (n, s, m, l, x)
uv run python scripts/download_models.py
```

This will download all 5 models (~220MB total) to `models/base_models/` and skip any that already exist.

##### Manual Download for Individual Models

```bash
# Download YOLOv11n (nano - recommended for start)
curl -L -o models/base_models/yolov11n.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

# Download YOLOv11s (small - best balance)
curl -L -o models/base_models/yolov11s.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"

# Download YOLOv11m (medium)
curl -L -o models/base_models/yolov11m.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"

# Download YOLOv11l (large)
curl -L -o models/base_models/yolov11l.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"

# Download YOLOv11x (extra large)
curl -L -o models/base_models/yolov11x.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
```

#### Switching Models
To use a different model, update `configs/H1.yaml`:

```yaml
train:
  model: models/base_models/yolov11s.pt  # Change from yolov11n.pt
```

Or specify via command line:
```bash
uv run python scripts/training/train_yolo.py --model models/base_models/yolov11s.pt
```

### Storage and Resource Considerations
- Disk Space Requirements:
  - Temporary storage: 5-10 GB
  - Permanent storage: Optional, depending on output volume
- Memory Usage:
  - Minimum: 4 GB RAM
  - Recommended: 16+ GB RAM for large batch training
- CPU Requirements:
  - Minimum cores: 2 cores
  - Optimal performance: 8+ cores or GPU-enabled compute node
- Quota Impact:
  - High read/write I/O during training. Avoid running on shared `/home` paths.

### Computational Resources

To run training effectively, allocate a GPU node using `salloc`.

**Standard Allocation Command (H200 GPU):**
```bash
salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:H200:1
```

**Resource Breakdown:**
*   `-N1`: 1 Node
*   `--ntasks-per-node=4`: 4 CPU cores (good for data loading)
*   `-t8:00:00`: 8 hours duration
*   `--gres=gpu:H200:1`: Request 1 NVIDIA H200 GPU


### Additional Resources
- Internal PACE Documentation for accessing additional computational resources:
  - https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096
- External Resources:
  - YOLO: https://www.ultralytics.com/yolo