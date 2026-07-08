# Running Alternative Models

All commands are run from the `alternative-models/` directory unless noted.

---

## Running All Models Together (Recommended)

To train all models on the same data split and then generate the benchmark report:

```bash
# Step 1: Generate the shared split (only needed once)
cd alternative-models
python benchmarking/generate_split.py

# Step 2: Train all models sequentially on the same split
python benchmarking/run_all.py --split benchmarking/splits/split.json

# Step 3: Generate the benchmark report
python evaluate.py
```

`run_all.py` trains every model (stacked hourglass, ViT, HRNet, HRNet-GCN, ml-morph) using the same split file so results are directly comparable. It prints a summary table of exit codes and runtimes at the end.

> **Note:** ml-morph requires `dlib` to be installed: `uv pip install dlib`

---

## Running Individual Models

### Stacked Hourglass

```bash
# Preprocess (axis-aligned YOLO)
cd alternative-models/stacked-hourglass
python preprocessing.py --config default

# Preprocess (OBB YOLO)
python preprocessing.py --config default --mode obb

# Train
python train.py --config default --data ./data/training_data
```

Checkpoint: `alternative-models/stacked-hourglass/checkpoints/stacked_hourglass_best.pth`
Log: `alternative-models/stacked-hourglass/logs/stacked_hourglass.log`

---

## ViT

```bash
# Preprocess (axis-aligned YOLO)
cd alternative-models/vit
python preprocessing.py --config default

# Preprocess (OBB YOLO)
python preprocessing.py --config default --mode obb

# Train
python train.py --config default --data ../data/training_data
```

Checkpoint: `alternative-models/vit/checkpoints/vit_best.pth`
Log: `alternative-models/vit/logs/vit.log`

---

## HRNet

```bash
# Preprocess (axis-aligned YOLO)
cd alternative-models/hrnet
python preprocessing.py --config default

# Preprocess (OBB YOLO)
python preprocessing.py --config default --mode obb

# Train (cross-attention variant)
python train.py --config default --data /path/to/training_data
```

Checkpoint: `alternative-models/hrnet/checkpoints/hrnet_best.pth`
Log: `alternative-models/hrnet/logs/hrnet.log`

> Paths in `configs/default.json` point to the Linux cluster. Update `training_data_dir`, `imgdir`, and `tps_data_dir` before running locally.

---

## HRNet Heatmap (paper-faithful)

Direct implementation of the Wang et al. CVPR 2019 HRNet landmark detection method.
Uses the highest-resolution HRNet branch + 1×1 conv heatmap head + soft-argmax coordinate extraction.
This is the correct SOTA baseline for sample complexity comparisons.

```bash
cd alternative-models/hrnet

# Train with shared split (recommended for benchmarking)
python train_heatmap.py --split /path/to/split.json

# Train on WFLW (sample complexity study)
python run_wflw.py --split /path/to/wflw/splits/wflw_0.8_seed42.json

# Evaluate on WFLW test set
python evaluate_wflw.py \\
    --checkpoint checkpoints/hrnet_heatmap_wflw_best.pth \\
    --split /path/to/wflw/splits/wflw_0.8_seed42.json \\
    --config configs/wflw-config.json \\
    --output-json results/wflw_eval_hrnet_0.8.json

# Train with data directory (random split)
python train_heatmap.py --data /path/to/training_data
```

Checkpoint: `alternative-models/hrnet/checkpoints/hrnet_heatmap_wflw_best.pth`
Log:        `alternative-models/hrnet/logs/hrnet_heatmap_wflw.log`
Config:     `alternative-models/hrnet/configs/wflw-config.json`

---

## HRNet-GCN

```bash
# Preprocess (OBB YOLO)
cd alternative-models/hrnet-gcn
python preprocessing.py

# Train
python train.py --config default-config.json
```

Checkpoint: `alternative-models/hrnet-gcn/checkpoints/hrnet_gcn_best.pth`
Log: `alternative-models/hrnet-gcn/logs/hrnet_gcn.log`

> Paths in `default-config.json` point to the Linux cluster. Update `training_data_path`, `imgdir`, and `tps_data_dir` before running locally.

---

## Diagnostics

```bash
cd alternative-models/tests
python run_diagnostics.py
```

Runs import checks and forward pass shape assertions for all models. No GPU or real data required.
