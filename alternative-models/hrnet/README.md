# HRNet Heatmap Model

This directory contains two parallel HRNet heatmap regression implementations
that share the same backbone architecture but target different datasets and
are completely independent in their training pipelines.

---

## 1. Lizard Toepad Model (9-point landmark detection)

Detects 9 landmarks on lizard toepad images using HRNet-W18 with a fused
multi-scale heatmap head.

### Key files

| File | Purpose |
|---|---|
| hrnet_heatmap.py | Shared HRNet-W18 backbone + fused head architecture |
| hrnet_heatmap_lizard.py | Lizard subclass (num_landmarks=9, heatmap_size=128, sigma=4.0) |
| dataset.py | Lizard dataset loader (.pt files, 9-point coords) |
| train_heatmap.py | Lizard training script |
| train.py | Original Lizard training entry point |
| model.py | Earlier Lizard model (reference only) |
| preprocessing.py | YOLO-OBB crop preprocessing for Lizard images |
| configs/heatmap_default.json | Default Lizard training config |

### How to run

```bash
cd alternative-models/hrnet
python preprocessing.py --config heatmap_default
python train_heatmap.py --split /path/to/split.json
```

---

## 2. WFLW Face Alignment Model (98-point landmark detection)

Paper-faithful reproduction of Wang et al. CVPR 2019 HRNet face alignment
on the WFLW 98-point benchmark. Target NME: <= 0.050 (paper: 0.046).

### Key files

| File | Purpose |
|---|---|
| hrnet_heatmap.py | Shared architecture (same as Lizard model above) |
| wflw_pt_dataset.py | WFLW dataset using reference crop() augmentation |
| train_heatmap_wflw_ref.py | Reference pipeline training script |
| configs/wflw-config.json | WFLW config (hyperparameters match paper YAML exactly) |
| evaluate_wflw.py | Standalone evaluator (uses hard_argmax, not decode_preds) |

### How to run

```bash
# From the alternative-datasets/WFLW/ directory:
python run_wflw_heatmap.py --split ./splits/wflw_1.0_seed42.json

# Or directly:
cd alternative-models/hrnet
python train_heatmap_wflw_ref.py \
    --split /path/to/splits/wflw_1.0_seed42.json \
    --config configs/wflw-config.json
```

### Architecture

- Backbone: HRNet-W18 (ImageNet pretrained via timm)
- Head: all 4 resolution branches fused at 64x64, then 1x1 conv to 98 heatmaps
- Loss: MSE between predicted and Gaussian target heatmaps (sigma=1.5)
- Optimizer: Adam, lr=0.0001, WD=0.0
- LR schedule: MultiStepLR decay x0.1 at epochs 30 and 50
- Evaluation: reference decode_preds() + compute_nme() (512px space, IOD normalised)

### Stashed baseline

`_stash/` contains the superseded v1 implementation (train_heatmap_wflw.py,
wflw_heatmap_dataset.py) which achieved Test NME 0.0605. These files must not
be imported. See `_stash/README.md` for details.

---

## Shared components

`hrnet_heatmap.py` is imported by both models and must not be modified without
checking both consumers. It is also used by the GCN hinit variant in
`alternative-models/hrnet-gcn/hrnet_gcn_hinit.py`.
