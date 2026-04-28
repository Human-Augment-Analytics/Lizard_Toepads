# HRNet Landmark Model

## Architecture

HRNet-W18 backbone (via `timm`, `features_only=True`) with a cross-attention head. Landmark queries attend to image feature tokens via multi-head cross-attention, followed by self-attention between landmarks, then a coordinate regression head that outputs 9 × 2 normalized coordinates (sigmoid-activated, range [0, 1]).

Key classes: `HRNetLandmarkModel`, `LizardDataset`.

## Sources

- Model + training: `stacked-hourglass/hrnet/hrnet_training.ipynb`
- Preprocessing: `stacked-hourglass/hrnet/hrnet_preprocess.ipynb`

## Data Format

Preprocessing saves `.pt` files to `training_data_dir/heatmaps/`. Each file contains:

- `image`: `torch.Tensor` shape `(3, 512, 512)` uint8 — cropped lizard toepad image
- `heatmap`: `torch.Tensor` shape `(9, 512, 512)` float32 — Gaussian heatmap per landmark
- `tps`: `torch.Tensor` shape `(9, 2)` float32 — pixel coordinates

## Configuration

This model uses absolute Linux cluster paths in `configs/default.json`. Update `imgdir`, `tps_data_dir`, `training_data_dir`, and `test_data_dir` before running. The YOLO checkpoint is `yoloobb.pt` (OBB variant), different from the other models.

## How to Run

```bash
cd alternative-models/hrnet

python preprocessing.py --config default

python train.py --config default --data /path/to/training_data
```

Edit `configs/default.json` to set the correct cluster paths before running.
