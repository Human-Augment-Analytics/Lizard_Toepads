# WFLW Face Landmark Detection — Project Summary

## Important: Separation from Lizard HRNet-GCN

**This project is entirely separate from the Lizard toepad landmark detection pipeline.**

The WFLW implementation reuses the `HRNetGNN` model class from `alternative-models/hrnet-gcn/hrnet_gcn.py` as a read-only import, but shares no training infrastructure, data pipeline, configs, or checkpoints with the Lizard project. The five frozen Lizard GCN files must never be modified for WFLW purposes:

- `alternative-models/hrnet-gcn/hrnet_gcn.py` — frozen
- `alternative-models/hrnet-gcn/train.py` — frozen
- `alternative-models/hrnet-gcn/utils.py` — frozen
- `alternative-models/hrnet-gcn/lizard_dataset.py` — frozen
- `alternative-models/hrnet-gcn/default-config.json` — frozen

All WFLW-specific code lives under `alternative-datasets/wflw/` and `alternative-models/hrnet-gcn/train_wflw.py`.

---

## Research Objective

Evaluate whether HRNet-GCN with an anatomically structured graph prior achieves **reasonable landmark localization under low data availability** on the WFLW 98-point face alignment benchmark. The primary question is whether the structural prior allows graceful degradation as training data is reduced — not SOTA performance.

---

## Architecture

### Backbone
HRNet-W18 (`timm/hrnet_w18.ms_aug_in1k`), pretrained on ImageNet, `features_only=True`. Uses the last feature stage (144 channels, ~16×16 spatial resolution at 512px input).

### GCN Head
Cascade refinement starting from a pre-computed mean face shape:

```
mean_shape (98, 2) + Gaussian noise (training only)
         ↓
For each of num_iters iterations:
    sample_features(feat_map, coords)     → (B, 98, 144)
    node_feat_proj: Linear(144, 128)       → (B, 98, 128)
    ReLU
    flatten to (B*98, 128)
    num_layers × GCNConv(128, 128) + ReLU
    delta_head: Linear(128, 2)             → (B, 98, 2)
    coords = coords + delta
```

### Current config (`configs/wflw-config.json`)
| Parameter | Value | Notes |
|---|---|---|
| `num_landmarks` | 98 | WFLW 98-point scheme |
| `gnn_hidden` | 128 | GCN hidden dimension |
| `num_layers` | 3 | GCNConv layers per iteration |
| `num_iters` | 4 | Cascade refinement steps |
| `batch_size` | 32 | Training |
| `val_batch_size` | 64 | Validation (no gradients) |
| `lr` | 1e-4 | Adam optimizer |
| `init_noise_sigma` | 0.05 | ~25px noise on mean shape |
| `epochs` | 150 | |

### Graph topology
98-node facial graph with 93 unique undirected edges (186 directed):
- **Jaw contour** (0–32): open chain
- **Eyebrows** (33–41, 42–50): open chains
- **Nose bridge** (51–54) + **nose base** (55–59): open chains
- **Left eye** (60–67), **right eye** (68–75): closed loops
- **Outer mouth** (76–87), **inner mouth** (88–95): closed loops
- **Pupils** (96, 97): edges to respective eye centers (64, 72)

---

## Data Pipeline

### Input
Raw WFLW JPEG images with per-face bounding box annotations — no object detection required. The annotation file provides ground-truth bounding boxes alongside landmarks.

### Preprocessing (`preprocess.py`)
1. Parse annotation line (207 tokens: 196 landmark coords + 4 bbox + 6 attrs + 1 path)
2. Expand bounding box by 10% padding
3. Crop face region
4. Letterbox resize to 512×512
5. Save `.pt` file: `{"image": (3,512,512) uint8, "tps": (98,2) float32, "attrs": (6,) uint8, "orig_size": (2,) int}`

### Data augmentation (`wflw_dataset.py`)
Applied per-sample during training (not at preprocessing time):

| Transform | Parameters | Probability |
|---|---|---|
| LongestMaxSize + PadIfNeeded | target 512×512 | always |
| Affine (scale + translate + rotate) | scale 0.85–1.15, translate ±5%, rotate ±30° | 80% |
| HorizontalFlip | — | 50% |
| RandomBrightnessContrast | ±20% | 35% (via OneOf) |
| HueSaturationValue | hue ±10, sat ±15, val ±10 | 35% (via OneOf) |
| GaussNoise | default params | 30% |
| ImageNet Normalize | mean/std | always |

**Note:** HorizontalFlip does not currently remap landmark indices. For a research experiment this is acceptable — the model will learn symmetric representations — but for production use the `WFLW_FLIP_PAIRS` from `graph_topology.py` should be applied to remap landmarks after flip.

### Mean shape initialization
A (98, 2) float tensor computed as the per-landmark mean of all training samples in the given split. Gaussian noise (σ=0.05) is added during training to teach the GCN to recover from imperfect initialization — mimicking the larger offsets caused by pose variation at test time.

---

## Training

Script: `alternative-models/hrnet-gcn/train_wflw.py`

- Does NOT use `HRNetGCNTrainingConfig` or any Lizard config class
- Loads config directly from `wflw-config.json` as a plain JSON dict
- Uses `WFLWDataset` (not `LizardDataset`) — supports arbitrary `num_landmarks`
- Gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)`
- LR scheduler: `ReduceLROnPlateau(factor=0.5, patience=15, min_lr=1e-6)`
- Checkpoint: `alternative-models/hrnet-gcn/checkpoints/hrnet_gcn_wflw_best.pth`
- Visualizations: `alternative-models/hrnet-gcn/visualizations/hrnet_gcn_wflw/`
  - Scatter plots every 10 epochs (from frozen `utils.visualize_landmarks`)
  - **Landmark overlays on actual face crops** at epoch 3, then every 10 epochs (`overlay_epoch{N}_sample{i}.jpg`) — green = GT, red = predicted

---

## Quick Start

```bash
# One-time setup (preprocess + split + mean shape + config update)
python alternative-datasets/wflw/setup_wflw.py \
    --data-dir /path/to/raw/wflw/

# Train
python alternative-datasets/wflw/run_wflw.py \
    --split alternative-datasets/wflw/splits/wflw_0.8_seed42.json

# If mean shape path differs from config (e.g. after git pull):
python alternative-datasets/wflw/run_wflw.py \
    --split ./splits/wflw_0.8_seed42.json \
    --mean-shape ./mean_shapes/mean_shape_wflw_0.8_seed42.pt
```

---

## Evaluation

```bash
python alternative-datasets/wflw/evaluate_wflw.py \
    --checkpoint alternative-models/hrnet-gcn/checkpoints/hrnet_gcn_wflw_best.pth \
    --split alternative-datasets/wflw/splits/wflw_0.8_seed42.json \
    --mean-shape alternative-datasets/wflw/mean_shapes/mean_shape_wflw_0.8_seed42.pt \
    --config alternative-datasets/wflw/configs/wflw-config.json \
    --output-json alternative-datasets/wflw/results/nme_wflw.json
```

Output: NME on full test set + 6 subsets (pose, expression, illumination, makeup, occlusion, blur).

---

## File Structure

```
alternative-datasets/wflw/
├── WFLW_PROJECT.md          ← this file
├── configs/
│   └── wflw-config.json     ← training config (WFLW-specific, not shared with Lizard)
├── splits/                  ← generated split JSON files (gitignored)
├── mean_shapes/             ← computed mean shapes (gitignored)
├── results/                 ← NME evaluation output (gitignored)
├── preprocess.py            ← raw WFLW → .pt crops
├── generate_split.py        ← fraction-aware split generator
├── compute_mean_shape.py    ← per-landmark mean from training set
├── setup_wflw.py            ← one-shot setup orchestrator
├── run_wflw.py              ← training run launcher
├── run_hinit_lizard.py      ← Model B (HRNet-init) launcher for Lizard validation
├── wflw_dataset.py          ← WFLW dataset class (supports arbitrary num_landmarks)
├── graph_topology.py        ← WFLW_FLIP_PAIRS + re-export of make_wflw_edge_index
├── evaluate_wflw.py         ← NME evaluator with per-subset breakdown
└── WFLW_MIGRATION.md        ← original migration planning document

alternative-models/hrnet-gcn/
├── hrnet_gcn.py             ← FROZEN — Lizard GCN model (read-only import for WFLW)
├── train.py                 ← FROZEN — Lizard training entry point
├── utils.py                 ← FROZEN — Lizard utilities
├── lizard_dataset.py        ← FROZEN — Lizard dataset (9 landmarks, hardcoded)
├── default-config.json      ← FROZEN — Lizard config
├── hrnet_gcn_hinit.py       ← Model B: HRNet-GCN with image-conditioned init
├── train_wflw.py            ← Model A training script (WFLW, 98 landmarks)
└── train_hinit.py           ← Model B training script

alternative-datasets/common/
├── split_utils.py           ← Shared: sample_fraction, write_split
└── graph_topologies.py      ← Shared: make_chain_edge_index, make_wflw_edge_index, get_edge_index
```
