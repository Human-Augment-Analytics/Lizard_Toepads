# Stacked Hourglass

## Architecture

A 2-stack hourglass network for heatmap regression. Each stack produces one heatmap channel per landmark (9 total). The network takes 512×512 RGB crops as input and outputs heatmaps at 128×128 resolution. Loss is computed at each stack using MSE, encouraging intermediate supervision.

Key classes: `Conv`, `Residual`, `Hourglass`, `HeatmapLoss`, `StackedHourGlass`.

## Sources

- Model: `stacked-hourglass/model/stackedhourglass.py`
- Dataset: `stacked-hourglass/lizarddataset.py`
- Preprocessing: `stacked-hourglass/hourglass_preprocessing_tst.ipynb`
- Training: `stacked-hourglass/train.py`

## Data Format

Preprocessing saves `.npz` files to `training_data_dir/heatmaps/`. Each file contains:

- `image`: `np.ndarray` shape `(H, W, 3)` dtype `uint8` — the cropped lizard toepad image
- `heatmap`: `np.ndarray` shape `(H, W, 9)` dtype `float32` — one Gaussian heatmap channel per landmark

## How to Run

```bash
cd alternative-models/stacked-hourglass

# Preprocess raw images into .npz training files
python preprocessing.py --config default

# Train the model
python train.py --config default --data ./data/training_data
```

Edit `configs/default.json` to adjust paths (`yolo_path`, `imgdir`, `tps_data_dir`, `training_data_dir`) and hyperparameters before running.
