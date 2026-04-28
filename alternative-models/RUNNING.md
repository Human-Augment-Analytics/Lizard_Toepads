# Running Alternative Models

All commands are run from the `alternative-models/` directory unless noted.

---

## Stacked Hourglass

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

# Train
python train.py --config default --data /path/to/training_data
```

Checkpoint: `alternative-models/hrnet/checkpoints/hrnet_best.pth`
Log: `alternative-models/hrnet/logs/hrnet.log`

> Paths in `configs/default.json` point to the Linux cluster. Update `training_data_dir`, `imgdir`, and `tps_data_dir` before running locally.

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
