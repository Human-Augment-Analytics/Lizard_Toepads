# PACE Documentation Contribution Template

## Document Metadata
* **Authors**: Dylan Herbig, Junling Zhuang
* **Date Created**: 2025-07-18
* **Last Updated**: 2025-09-16 By Junling Zhuang
* **Applicable Clusters**: Only utilized on ICE, but also applicable for Phoenix and Hive

## YOLO Model Setup

### Overview
This documentation describes the setup and usage of a YOLOv8-based object detection pipeline used to identify lizard toepads from scanned images. YOLO (You Only Look Once) is a real-time object detection system that is fast, accurate, and well-suited for high-throughput morphometric tasks. This tool is highly relevant to PACE users working in biology, image processing, and machine learning, particularly for research teams digitizing morphological traits at scale. This guide assumes that you have already created a labeled dataset.


### Prerequisites
- Required access/permissions:
  - Active PACE account
  - Allocate computatioanl resources using `salloc` command
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

### Configuration

All scripts use a unified configuration file (e.g., `configs/H4.yaml`). You can customize preprocessing, merging, dataset splitting, and training parameters in one place:

```yaml
preprocessing:
  # Step 1: Raw data sources
  image-dir: /storage/ice-shared/cs8903onl/miami_fall_24_jpgs
  tps-dir: /storage/ice-shared/cs8903onl/tps_files
  
  # Step 2: Process bottom view with TPS landmarks
  bottom-view-processed-dir: data/processed_bottom
  
  # Step 3: Integrate upper view dataset
  addtional-upper-side-data-dir: data/upper_dataset_roboflow/train
  
  # Step 4: Merge bilateral annotations
  merged-processed-dir: data/processed_merged
  target-size: 1280
  point-size: 10
  add-points: true

split:
  # Step 5: Train/val split
  images-dir: data/processed_merged/images
  labels-dir: data/processed_merged/labels
  output-dir: data/dataset
  train-ratio: 0.8
  seed: 42
  copy: false

train:
  model: models/base_models/yolov11s.pt
  epochs: 300
  batch: 32
  imgsz: 1280
  # ... more training parameters
```

### Data Preprocessing

Run the 3-step preprocessing pipeline to generate the bilateral dataset (6 classes):

```bash
# 1. Process TPS landmarks (Bottom View)
uv run python scripts/preprocessing/process_tps_files_with_id.py --config configs/H4.yaml

# 2. Merge with Upper View Dataset
uv run python scripts/preprocessing/merge_bilateral_dataset.py --config configs/H4.yaml

# 3. Create Train/Val Split
uv run python scripts/preprocessing/split_dataset.py --config configs/H4.yaml
```

This will generate:
- `data/processed_bottom/`: Intermediate bottom view data
- `data/processed_merged/`: Combined bilateral dataset (6 classes)
- `data/dataset/`: Final split ready for YOLO training
- `data/processed_merged/visualizations/`: Visualization images with bounding boxes (for verification)

**Class Mapping:**
- 0: up_finger
- 1: up_toe
- 2: bot_finger
- 3: bot_toe
- 4: ruler
- 5: id

### Step-by-Step Instructions

#### Option A: Using uv (Recommended - Fast and Modern)

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
   # Preprocess & Merge Data
   uv run python scripts/preprocessing/process_tps_files_with_id.py --config configs/H4.yaml
   uv run python scripts/preprocessing/merge_bilateral_dataset.py --config configs/H4.yaml

   # Split dataset
   uv run python scripts/preprocessing/split_dataset.py --config configs/H4.yaml

   # Train YOLO (will auto-download YOLOv11s if needed)
   uv run python scripts/training/train_yolo_aug.py --config configs/H4.yaml
   ```

   **Note**: Pre-trained models are stored in `models/base_models/`. The training script automatically downloads YOLOv11n on first run if not present.

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

# Or with standard Python
python scripts/download_models.py
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

#### Option B: Using Conda (Traditional)

1. Enable `conda` to be able to create and manage language-agnostic virtual environments
   ```bash
   # Install latest version of Miniconda via terminal
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh
   ```
   In case you have a different architecture compared to `x86_64` use `uname -m` to figure out what architecture your system is on. Also refer to this guide for additional information: https://www.anaconda.com/docs/getting-started/miniconda/install#linux

2. Create and activate Conda environment:
   ```bash
   conda create -n yolo11-env python=3.13 -y
   conda activate yolo11-env
   ```

3. Install required packages:
   ```bash
   pip install ultralytics pillow numpy tqdm
   ```

4. Run the complete workflow using unified config:
   ```bash
   # Preprocess data
   python scripts/preprocessing/process_tps_files.py

   # Split dataset
   python scripts/preprocessing/split_dataset.py

   # Train model
   python scripts/training/train_yolo.py
   ```
   Parameter Descriptions
   - `task=detect`: Specifies the type of task YOLO should perform. In this case, `detect` means object detection (bounding box prediction). Other options include `segment` (segmentation), `classify` (image classification), etc.
   - `mode=train`: Indicates the operation mode. `train` tells YOLO to train a model from the given data. Other modes include `predict`, `val`, and `export`.
   - `model=yolov11n.pt`: 	Chooses the model architecture and optionally a pretrained weights file to start from. `yolov11n.pt` is the nano version of YOLOv11 — the smallest and fastest, best for quick experiments and resource-constrained environments. Alternatives include `yolov11s.pt` (small), `yolov11m.pt` (medium), etc. More information about current YOLO versions can be found here: https://docs.ultralytics.com/models/#featured-models 
   - `data=custom_data.yaml`: Points to your custom dataset configuration file. This YAML file defines the training/validation folder paths, number of classes (`nc`), and class names.
   - `epochs=50`: The number of times the model will iterate over the entire training dataset. More epochs usually lead to better performance (up to a point), but increase training time.
   - `imgsz=640`: Sets the size (height and width) of input images used during training and validation. All images are resized to this dimension. The default is 640x640. Larger sizes may improve accuracy, but increase GPU memory requirements and training time.

   **Output**: Once successfully trained, your model should be saved under a `runs/detect/model_name` directory

7. After you have successfully trained your model, you can now run inferencing on a new image. You can either create a custom inferencing function from a Python file or from the terminal
   ```bash
   yolo task=detect mode=predict model=runs/detect/tps_yolo_exp4/weights/best.pt source=data/miami_fall_24_jpgs/1001.jpg imgsz=640 save=True
   ```
   Parameter Descriptions:
   - `model`:  Path to your trained model weights 
   - `source`: Path to the image, folder, or video you want to run inference on
   - `save=True`: Save the predicted images with bounding boxes drawn
   
### Configuration Details
1. YOLO `data.yaml` configuration file
   ```yaml
   path: dataset
   train: images/train
   val: images/val
   nc: 3
   names: ["toepad", "toe", "ruler"]
   ```

2. Parameter Descriptions
   - `path`: Base directory of the dataset
   - `train`, `val`: Relative paths to training and validation image folders
   - `nc`: Number of classes (3 for toepads)
   - `names`: List of class names

3. More detailed configuration settings can be found here: https://docs.ultralytics.com/usage/cfg/

### Troubleshooting

#### Common Issue 1
**Error Message:**
```
PIL.Image.DecompressionBombError: Image size (199042800 pixels) exceeds limit of 178956970 pixels
```

**Resolution:**
1. Ensure the input image resolution is within reasonable limits before passing to YOLO.
2. Resize extremely large images or adjust `Image.MAX_IMAGE_PIXELS`:
    ```python
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
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

### Directory Structure
```
project/
├── scripts/
│   ├── preprocessing/     # Data preparation scripts
│   │   └── process_tps_files.py
│   ├── training/          # Model training scripts
│   │   └── train_yolo.py
│   └── inference/         # Prediction scripts
│       └── predict.py
├── models/
│   └── base_models/       # Pre-trained YOLO models (auto-downloaded)
│       └── yolov11n.pt    # Will be downloaded on first use
├── data/
│   ├── processed/
│   │   ├── images/         # YOLO-ready images
│   │   ├── labels/         # YOLO format annotations
│   │   └── visualizations/ # Verification images with bounding boxes
│   ├── dataset/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── splits/             # auto-generated file lists (train.txt/val.txt)
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── runs/
│   └── detect/            # YOLO training outputs
├── pyproject.toml         # Python project dependencies
├── requirements.txt       # Alternative dependency list
└── README.md             # This documentation
```

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

### Create train/val split (two options)

Option A — Standard folders (default symlinks, no extra storage):

```bash
uv run python scripts/preprocessing/split_dataset.py
```

Unified config for split and train:

```bash
# All scripts now use unified config by default
uv run python scripts/preprocessing/split_dataset.py
uv run python scripts/training/train_yolo.py
```

You can now avoid a separate `data.yaml`. Put dataset info in `configs/project.yaml` under `dataset`:

```yaml
dataset:
  path: /home/hice1/jzhuang48/Lizard_Toepads/data/dataset
  train: images/train
  val: images/val
  nc: 3
  names: ["finger", "toe", "ruler"]
```

Option B — File lists (no folders needed):

```yaml
train: /home/hice1/jzhuang48/Lizard_Toepads/data/splits/train.txt
val: /home/hice1/jzhuang48/Lizard_Toepads/data/splits/val.txt
nc: 3
names: ["finger", "toe", "ruler"]
```

### Additional Resources
- Internal PACE Documentation for accessing additional computational resources:
  - https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096
- External Resources:
  - YOLO: https://www.ultralytics.com/yolo

### Project Docs
- Improvement plan for small-object detection (YOLOv11): see `docs/YOLOv11_Improvement_Plan.md`
- Benchmarking strategy and ablation templates: see `docs/Benchmarking_Strategy.md`

### Complete Working Example

#### Using uv (Recommended):
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
cd ~/Lizard_Toepads
uv sync  # or: uv pip install -r requirements.txt

# Allocate computational resources for training
salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:H200:1

# Complete workflow using unified config
uv run python scripts/preprocessing/process_tps_files.py
uv run python scripts/preprocessing/split_dataset.py
uv run python scripts/training/train_yolo.py

# Run inference
uv run yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=data/miami_fall_24_jpgs/1001.jpg
```

#### Using Conda:
```bash
# Allocate computational resources
salloc -N1 --ntasks-per-node=4 -t8:00:00 --gres=gpu:H200:1

# Activate conda environment
conda activate yolo11-env

# Complete workflow using unified config
python scripts/preprocessing/process_tps_files.py
python scripts/preprocessing/split_dataset.py
python scripts/training/train_yolo.py

# Run inference
python scripts/inference/predict.py
```

Expected workflow and output:
```
✅ Dataset prepared at: dataset
   Images: 2100 | Train: 1680 | Val: 420
Predicting with YOLOv11...
Predicted boxes: tensor([[x1, y1, x2, y2], ...])
```

**NOTE**: You may need to fine-tune the imgsz and batch parameters depending on GPU memory availability.

**WARNING**: Make sure your input `.jpg` files and `.txt` labels are correctly matched and formatted in YOLO format (class_id center_x center_y width height, all normalized).

### Version Information
- Software version documented: YOLOv11 (Ultralytics 8.3.168)
- Last tested on PACE: 2025-09-16
- Compatible PACE environments: ICE (with Conda and Python 3.10+)