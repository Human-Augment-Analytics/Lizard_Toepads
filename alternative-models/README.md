# alternative-models

This directory contains experimental deep learning models for lizard toepad landmark detection. Each model is a self-contained Python package with its own preprocessing pipeline, dataset loader, model definition, training script, and configuration.

## Directory Layout

```
alternative-models/
├── README.md                  (this file)
├── common/                    Shared utility functions used by all model packages
│   ├── __init__.py
│   ├── tps_utils.py           TPS file parsing and y-coordinate flipping
│   ├── yolo_utils.py          YOLO crop extraction and coordinate localization
│   └── heatmap_utils.py       Gaussian heatmap generation and overlay visualization
├── stacked-hourglass/         Stacked Hourglass network (heatmap regression)
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── README.md
│   └── configs/default.json
├── vit/                       Vision Transformer (coordinate regression)
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── README.md
│   └── configs/default.json
└── hrnet/                     HRNet with cross-attention head (coordinate regression)
    ├── __init__.py
    ├── model.py
    ├── dataset.py
    ├── preprocessing.py
    ├── train.py
    ├── README.md
    └── configs/default.json
```

## Models

### Stacked Hourglass
A 2-stack hourglass network that performs heatmap regression. Each stack produces a heatmap per landmark; the final prediction is the argmax of the last stack's output. Source: `stacked-hourglass/model/stackedhourglass.py` and `stacked-hourglass/hourglass_preprocessing_tst.ipynb`.

### ViT (Vision Transformer)
A pretrained ViT backbone (via `timm`) with a regression head that directly predicts landmark coordinates. Uses 224×224 ImageNet-normalized inputs. Source: `stacked-hourglass/vit/vit_training_tst.ipynb`.

### HRNet
High-Resolution Network backbone with a cross-attention head for coordinate regression. Uses 512×512 inputs. Source: `stacked-hourglass/hrnet/hrnet_training.ipynb`.

### HRNet-GCN
Graph Convolutional Network variant of HRNet. See `hrnet-gcn/` for details.

## Running Preprocessing and Training

All commands are run from the `alternative-models/` directory.

### Stacked Hourglass

```bash
cd alternative-models/stacked-hourglass
python preprocessing.py --config default
python train.py --config default --data ./data/training_data
```

### ViT

```bash
cd alternative-models/vit
python preprocessing.py --config default
python train.py --config default --data ./data/training_data
```

### HRNet

```bash
cd alternative-models/hrnet
python preprocessing.py --config default
python train.py --config default --data ./data/training_data
```

Edit `configs/default.json` in each model directory to adjust paths and hyperparameters before running.
