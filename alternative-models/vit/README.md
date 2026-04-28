# ViT Landmark Model

## Architecture

A pretrained Vision Transformer (`vit_small_patch16_224` via `timm`) with a coordinate regression head. The backbone produces a feature vector which is passed through a 3-layer MLP head that outputs 9 × 2 normalized landmark coordinates (sigmoid-activated, range [0, 1]).

Key classes: `ViTLandmark`, `ViTDataset`.

## Sources

- Model + training: `stacked-hourglass/vit/vit_training_tst.ipynb`
- Preprocessing: `stacked-hourglass/vit/vit_preprocessing_tst.ipynb`

## Data Format

Preprocessing saves `.pt` files to `training_data_dir/vit/`. Each file contains:

- `image`: `torch.Tensor` shape `(3, 224, 224)` float32 — ImageNet-normalized crop
- `keypoints`: `torch.Tensor` shape `(9, 2)` float32 — pixel coordinates

## How to Run

```bash
cd alternative-models/vit

python preprocessing.py --config default

python train.py --config default --data ../data/training_data
```

Edit `configs/default.json` to adjust paths and hyperparameters before running.
