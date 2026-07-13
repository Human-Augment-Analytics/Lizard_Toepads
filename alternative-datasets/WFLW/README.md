# WFLW Face Landmark Detection

This directory contains the data pipeline and run scripts for WFLW 98-point
facial landmark detection. Two model variants are trained and compared:

| Model | Architecture | Script | Status |
|---|---|---|---|
| HRNet-GCN | HRNet backbone + GCN refinement | run_wflw.py | Baseline |
| HRNet Heatmap | HRNet-W18 pure heatmap regression | run_wflw_heatmap.py | Reference pipeline |

**Important: this project is entirely separate from the Lizard toepad pipeline.**
The frozen Lizard GCN files (hrnet_gcn.py, train.py, utils.py,
lizard_dataset.py, default-config.json) must never be modified for WFLW work.

---

## Results

| Model | Test NME | Notes |
|---|---|---|
| HRNet Heatmap (v1, stashed) | 0.0605 | Augmentation mismatch — see _stash/ |
| HRNet Heatmap (reference pipeline) | target <= 0.050 | Paper reports 0.046 |
| HRNet-GCN (mean init) | in progress | See GCN training logs |

---

## Quick Start

### One-time setup

```bash
python setup_wflw.py --data-dir /home/hice1/axu39/scratch/WFLW_data/
```

This preprocesses JPEG images to .pt crops, generates splits at fractions
[0.10, 0.25, 0.50, 0.75, 1.00], and computes mean shapes.

---

### Train HRNet-GCN (0.8 split)

```bash
python run_wflw.py --split ./splits/wflw_0.8_seed42.json
```

With explicit mean shape:
```bash
python run_wflw.py \
    --split ./splits/wflw_0.8_seed42.json \
    --mean-shape ./mean_shapes/mean_shape_wflw_0.8_seed42.pt
```

### Train HRNet-GCN (1.0 split — full dataset)

```bash
python run_wflw.py \
    --split ./splits/wflw_1.0_seed42.json \
    --mean-shape ./mean_shapes/mean_shape_wflw_1.0_seed42.pt
```

---

### Train HRNet Heatmap — reference pipeline (0.8 split)

```bash
python run_wflw_heatmap.py --split ./splits/wflw_0.8_seed42.json
```

### Train HRNet Heatmap — reference pipeline (1.0 split — paper comparison)

```bash
python run_wflw_heatmap.py --split ./splits/wflw_1.0_seed42.json
```

Checkpoint saved to `alternative-models/hrnet/checkpoints/hrnet_heatmap_wflw_ref_best.pth`.
Final Test NME is logged at the end of the run — compare against the paper's 0.046.

Both models accept the same `--split` argument so they can run on separate
HPC instances using the same split files for direct comparison.

---

### Evaluate HRNet-GCN (0.8 split)

```bash
python -B evaluate_wflw.py \
    --checkpoint ../../alternative-models/hrnet-gcn/checkpoints/hrnet_gcn_wflw_best.pth \
    --split splits/wflw_0.8_seed42.json \
    --mean-shape mean_shapes/mean_shape_wflw_0.8_seed42.pt \
    --config configs/wflw-config.json \
    --output-json results/wflw_eval_0.8.json
```

### Evaluate HRNet-GCN (1.0 split)

```bash
python -B evaluate_wflw.py \
    --checkpoint ../../alternative-models/hrnet-gcn/checkpoints/hrnet_gcn_wflw_best.pth \
    --split splits/wflw_1.0_seed42.json \
    --mean-shape mean_shapes/mean_shape_wflw_1.0_seed42.pt \
    --config configs/wflw-config.json \
    --output-json results/wflw_eval_1.0.json
```

Note: for the HRNet Heatmap model the authoritative Test NME is logged
automatically at the end of the training run using `decode_preds()` +
`compute_nme()`. There is no separate evaluation script needed for that model.

---

## Research Objective

Evaluate whether HRNet-GCN with an anatomically structured graph prior
achieves reasonable landmark localisation under low data availability on
the WFLW 98-point face alignment benchmark. The primary question is whether
the structural prior allows graceful degradation as training data is reduced.

The HRNet heatmap model provides the paper-faithful baseline for comparison.

---

## HRNet-GCN Architecture

### Backbone
HRNet-W18 (timm, features_only=True, ImageNet pretrained).

### GCN Head
Cascade refinement from a pre-computed mean face shape:

```
mean_shape (98, 2) + Gaussian noise (training only)
    for num_iters iterations:
        sample_features(feat_map, coords)
        GCNConv(128, 128) x num_layers
        delta_head: Linear(128, 2)
        coords = coords + delta
```

### Config (configs/wflw-config.json)

| Parameter | Value |
|---|---|
| num_landmarks | 98 |
| gnn_hidden | 128 |
| num_layers | 3 |
| num_iters | 4 |
| batch_size | 32 |
| lr | 1e-4 |
| epochs | 120 |

### Graph topology
98-node facial graph with 93 unique undirected edges:
- Jaw contour (0-32): open chain
- Eyebrows (33-41, 42-50): open chains
- Nose bridge (51-54) + nose base (55-59): open chains
- Left eye (60-67), right eye (68-75): closed loops
- Outer mouth (76-87), inner mouth (88-95): closed loops
- Pupils (96, 97): edges to eye centers (64, 72)

---

## HRNet Heatmap Architecture (reference pipeline)

### Model
HRNet-W18 backbone, all 4 resolution branches fused at 64x64, 1x1 conv head
to 98 heatmaps. Coordinates extracted via decode_preds() (reference argmax +
sub-pixel refinement + inverse affine to 512px space).

### Training (matches paper YAML exactly)
- Loss: MSE on Gaussian heatmaps (sigma=1.5)
- Optimizer: Adam, lr=0.0001, WD=0.0
- Schedule: MultiStepLR x0.1 at epochs 30, 50
- Epochs: 60, batch size: 16, input: 256x256, heatmap: 64x64

### Augmentation (reference pipeline)
Applied on 512px pre-cropped images before resizing to 256x256:
- Horizontal flip (p=0.5) with WFLW landmark remapping
- Scale jitter: Uniform(0.75, 1.25)
- Rotation: Uniform(-30, +30 deg), p=0.6

---

## Data Pipeline

### Preprocessing (preprocess.py)
Raw WFLW JPEG + annotations -> 512x512 affine-cropped .pt files:
- image: (3, 512, 512) uint8
- tps: (98, 2) float32 in [0,1]
- attrs: (6,) uint8 — pose, expression, illumination, makeup, occlusion, blur
- orig_size: (2,) int

### Split generation (generate_split.py)
```bash
python generate_split.py \
    --data-dir . \
    --fraction 1.0 \
    --seed 42 \
    --output splits/wflw_1.0_seed42.json
```

Fractions tested: 0.10, 0.25, 0.50, 0.75, 1.00.
Test set is always the full official WFLW test split (2500 samples).

---

## Evaluation

### HRNet-GCN standalone evaluation

```bash
python -B evaluate_wflw.py \
    --checkpoint ../../alternative-models/hrnet-gcn/checkpoints/hrnet_gcn_wflw_best.pth \
    --split splits/wflw_0.8_seed42.json \
    --mean-shape mean_shapes/mean_shape_wflw_0.8_seed42.pt \
    --config configs/wflw-config.json \
    --output-json results/wflw_eval_0.8.json
```

Output: NME, FR@0.1, AUC@0.1 on full test set and 6 attribute subsets
(pose, expression, illumination, makeup, occlusion, blur).

Note: `evaluate_wflw.py` uses hard_argmax for the GCN model. For the heatmap
model the authoritative test NME is logged at the end of the training run
using the reference decode_preds() + compute_nme() evaluation protocol.

---

## File Structure

```
alternative-datasets/WFLW/
├── README.md                 <- this file
├── WFLW_PROJECT.md           <- superseded by README.md (historical reference)
├── configs/
│   ├── wflw-config.json      <- GCN training config
│   └── hinit-config.json     <- HInit variant config
├── splits/                   <- generated split JSON files (gitignored)
├── mean_shapes/              <- computed mean shapes (gitignored)
├── results/                  <- NME evaluation output (gitignored)
├── preprocess.py             <- raw WFLW -> .pt crops
├── generate_split.py         <- fraction-aware split generator
├── compute_mean_shape.py     <- per-landmark mean from training set
├── setup_wflw.py             <- one-shot setup orchestrator
├── run_wflw.py               <- HRNet-GCN training launcher
├── run_wflw_heatmap.py       <- HRNet heatmap training launcher
│                                (calls train_heatmap_wflw_ref.py)
├── run_hinit_lizard.py       <- Model B (HRNet-init) launcher for Lizard
├── wflw_dataset.py           <- GCN WFLW dataset class
├── graph_topology.py         <- WFLW flip pairs + make_wflw_edge_index
└── evaluate_wflw.py          <- NME evaluator with per-subset breakdown

alternative-models/hrnet/
├── hrnet_heatmap.py          <- shared HRNet architecture
├── wflw_pt_dataset.py        <- heatmap WFLW dataset (reference crop())
├── train_heatmap_wflw_ref.py <- reference pipeline training script
├── _stash/                   <- superseded v1 baseline (NME 0.0605)
└── configs/wflw-config.json  <- heatmap hyperparameters (matches paper)

alternative-models/hrnet-gcn/
├── hrnet_gcn.py              <- FROZEN Lizard GCN model
├── train_wflw.py             <- GCN WFLW training script
└── ...

alternative-datasets/common/
├── split_utils.py            <- shared fraction sampling + JSON writing
└── graph_topologies.py       <- topology registry (chain, wflw)
```
