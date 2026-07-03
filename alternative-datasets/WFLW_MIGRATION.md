# WFLW Migration Plan — HRNet-GCN Low-Data Study

## Objective

Evaluate whether HRNet-GCN with graph-structured landmark refinement achieves **reasonable performance under low data availability** on a standardized face alignment benchmark. The research question is not SOTA performance — it is whether the structural prior encoded by the graph allows the model to learn stable, anatomically plausible predictions from a small fraction of available training data.

Two parallel model variants are developed:

| Model | Init Strategy | Primary Dataset | Status |
|---|---|---|---|
| **Model A** — HRNet-GCN (mean init) | Fixed mean shape from training set | WFLW | New |
| **Model B** — HRNet-GCN (HRNet init) | Image-conditioned initial prediction from HRNet regression head | Lizard (first), then WFLW | New |

Model A establishes baseline GCN performance on WFLW with the existing init strategy.
Model B tests whether replacing the fixed mean shape with an image-conditioned initial prediction improves performance, starting on Lizard where the existing training infrastructure is mature.

---

## Preservation Constraint

**The existing lizard HRNet-GCN implementation is frozen and must not be modified.**

This means the following files are read-only for the purposes of this migration:

```
alternative-models/hrnet-gcn/hrnet_gcn.py       ← DO NOT MODIFY
alternative-models/hrnet-gcn/train.py            ← DO NOT MODIFY
alternative-models/hrnet-gcn/utils.py            ← DO NOT MODIFY
alternative-models/hrnet-gcn/lizard_dataset.py   ← DO NOT MODIFY
alternative-models/hrnet-gcn/default-config.json ← DO NOT MODIFY
```

All new work is additive — new files alongside the existing ones, never edits to existing files. This ensures:
- The lizard benchmark results remain reproducible against a fixed implementation
- The existing trained checkpoint (`hrnet_gcn_best.pth`) remains valid and loadable
- Any regression introduced by new code cannot contaminate the established baseline

New configs (`wflw-config.json`, `hinit-config.json`) are additions, not replacements. New model classes (`hrnet_gcn_hinit.py`) and training scripts (`train_hinit.py`) are new files. The WFLW training loop modifications (noise-augmented init, topology registry lookup) live in the new training script, not in the existing `utils.py`.

---

## Part 1 — Model A: HRNet-GCN with Mean Init on WFLW

This is the existing `HRNetGNN` architecture run on WFLW with minimal changes. The goal is to establish a clean baseline — how does the current model do on a larger, more varied dataset?

### 1.1 Data Ingestion

**New file: `alternative-datasets/wflw/preprocess.py`**

Converts raw WFLW images + annotations to `.pt` crop files in the existing lizard format so all current training scripts, dataset classes, and split tooling work without modification.

Steps:
1. Parse `list_98pt_rect_attr_train_test.txt`
   - Format per line: `x0 y0 ... x97 y97  bb_x1 bb_y1 bb_x2 bb_y2  attr×6  image_path`
2. Load JPEG, crop using bounding box with 10% padding margin
3. Resize to 512×512 via letterbox (LongestMaxSize + PadIfNeeded)
4. Transform landmark coordinates into crop space, normalize to [0,1]
5. Save as `.pt`: `{"image": CHW uint8 tensor, "tps": (98, 2) float tensor}`

Output directory structure mirrors lizard data layout so `generate_split.py` and all dataset classes work unchanged.

### 1.2 Split Generation with Fraction Support

**New file: `alternative-datasets/wflw/generate_split.py`**

```bash
python generate_split.py \
    --data-dir /path/to/wflw_pt \
    --fraction 0.25 \
    --seed 42 \
    --output splits/wflw_0.25_seed42.json
```

Output is identical to the existing lizard `split.json` format. The test set is always the full WFLW test split — only the training fraction varies.

Fractions to generate: `[0.10, 0.25, 0.50, 0.75, 1.00]`

**Shared utility: `alternative-datasets/common/split_utils.py`**

```python
sample_fraction(paths, fraction, seed) -> list
write_split(train, val, test, output_path)
```

Both the lizard and WFLW split generators use this so the sampling logic isn't duplicated.

### 1.3 Mean Shape Computation

**New file: `alternative-datasets/wflw/compute_mean_shape.py`**

```bash
python compute_mean_shape.py \
    --split splits/wflw_1.0_seed42.json \
    --output mean_shape_wflw.pt
```

Compute per fraction — the mean shape for a 10% split should reflect only those 750 training images, not the full 7,500. Using the wrong mean shape for a fraction would be a confound.

### 1.4 GCN Graph Topology for WFLW

The current `make_chain_edge_index` connects all 98 landmarks as a simple chain. For WFLW this is anatomically wrong and will underuse the structural prior. A correct facial topology is required.

**New file: `alternative-datasets/wflw/graph_topology.py`**

```python
def make_wflw_edge_index() -> torch.Tensor
```

Encodes WFLW's 98-point facial structure with bidirectional edges:

| Region | Landmark Indices | Structure |
|---|---|---|
| Jaw contour | 0–32 | Chain |
| Left eyebrow | 33–41 | Chain |
| Right eyebrow | 42–50 | Chain |
| Nose bridge | 51–54 | Chain |
| Nose base | 55–59 | Chain |
| Left eye | 60–67 | Closed loop |
| Right eye | 68–75 | Closed loop |
| Outer mouth | 76–87 | Closed loop |
| Inner mouth | 88–95 | Closed loop |
| Left pupil | 96 | Edge to eye center (LM 64) |
| Right pupil | 97 | Edge to eye center (LM 72) |

**Topology registry: `alternative-datasets/common/graph_topologies.py`**

```python
def make_chain_edge_index(num_landmarks)   # existing lizard topology
def make_wflw_edge_index()                 # facial anatomy
```

Config selects topology via `"graph_topology": "wflw"`. Adding a future dataset requires only a new function here.

### 1.5 Noise-Augmented Initialization

For WFLW's pose variance the GCN must learn to recover from mean shape offsets larger than in the lizard dataset. Add Gaussian noise to initial coordinates during training only:

```python
# training loop only — not at inference
noise = torch.randn_like(mean_shape) * config.get("init_noise_sigma", 0.05)
initial_coords = mean_shape.unsqueeze(0).repeat(B, 1, 1) + noise
```

At inference, use the clean mean shape without noise. `init_noise_sigma=0.05` is the starting value (~25px in 512px space). This is coordinate-space augmentation that teaches the GCN a general recovery policy rather than a policy specific to the mean shape being exactly right.

### 1.6 Horizontal Flip Augmentation

WFLW faces are approximately symmetric. Horizontal flip requires remapping landmark indices.

**Add to `alternative-datasets/wflw/graph_topology.py`:**

```python
WFLW_FLIP_PAIRS = [
    (0, 32), (1, 31), ...,   # jaw
    (33, 42), (34, 43), ..., # brows
    (60, 68), (61, 69), ..., # eyes
    (76, 82), (77, 81), ..., # mouth
    (96, 97),                # pupils
]
```

### 1.7 Evaluation

**New file: `alternative-datasets/wflw/evaluate_wflw.py`**

NME = mean(||pred_i - gt_i||) / inter_ocular_distance, per image, then averaged across test set.

Inter-ocular distance: distance between outer eye corners (landmarks 60 and 72).

Report NME on: full test, pose, expression, illumination, make-up, occlusion, blur subsets.

### 1.8 Config

**New file: `alternative-models/hrnet-gcn/wflw-config.json`**

```json
{
    "training_data_path": "/path/to/wflw_pt/train",
    "num_landmarks": 98,
    "gnn_hidden": 128,
    "num_layers": 2,
    "num_iters": 4,
    "input_size": 512,
    "epochs": 150,
    "batch_size": 16,
    "lr": 1e-4,
    "train_val_split": 0.9,
    "graph_topology": "wflw",
    "mean_shape_path": "/path/to/mean_shape_wflw.pt",
    "init_noise_sigma": 0.05
}
```

No changes to `HRNetGNN` model architecture — `num_landmarks=98` is already a constructor argument.

---

## Part 2 — Model B: HRNet-GCN with HRNet Initial Prediction

### 2.1 Concept

Replace the fixed mean shape initialization with an image-conditioned initial prediction:

```
HRNet backbone → feature maps
      ↓
Global avg pool → FC regression head → initial_coords [B, N, 2]
      ↓                                      ↓
Feature maps ←─────────────────────────────┘
      ↓
GCN refinement (N iters: sample features → GCN → delta → update coords)
      ↓
final_coords [B, N, 2]
```

The backbone runs once. The initial prediction head is a shallow MLP on a pooled global feature. The GCN then refines from this image-conditioned starting point rather than from a fixed prior.

**Hypothesis:** the image-conditioned init gives the GCN a closer starting point, so refinement iterations handle only fine-grained correction rather than coarse positioning. This should improve both accuracy and convergence speed, especially on data with high pose variance.

**Crossover hypothesis:** at very low data fractions, the initial prediction head may itself be poorly trained and variable, making the cascade's starting distribution inconsistent. Mean init has a predictable starting point regardless of data quantity. There may be a fraction threshold below which mean init is more stable. Measuring this crossover on Lizard first (before WFLW) validates the approach in a controlled setting.

### 2.2 Architecture Changes

**New file: `alternative-models/hrnet-gcn/hrnet_gcn_hinit.py`**

Extends `HRNetGNN` with an initial prediction head. The existing `HRNetGNN` class is not modified — this is a parallel implementation.

```python
class HRNetGNNWithInit(nn.Module):
    def __init__(self, hrnet_backbone="hrnet_w18", feat_dim=64,
                 gnn_hidden=128, num_layers=2, num_landmarks=9, num_iters=3):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters

        self.backbone = timm.create_model(
            hrnet_backbone, pretrained=True, features_only=True
        )
        backbone_channels = self.backbone.feature_info[-1]['num_chs']  # 144 for W18

        # Initial prediction head — global avg pool → MLP → (N, 2)
        self.init_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B, C, 1, 1]
            nn.Flatten(),              # [B, C]
            nn.Linear(backbone_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_landmarks * 2),
            nn.Sigmoid()               # normalize to [0, 1]
        )

        # GCN refinement — same as existing HRNetGNN
        self.node_feat_proj = nn.Linear(backbone_channels, gnn_hidden)
        self.gnn_layers = nn.ModuleList([
            GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)
        ])
        self.delta_head = nn.Linear(gnn_hidden, 2)

    def sample_features(self, feat_map, coords):
        # identical to existing HRNetGNN.sample_features
        grid = (coords * 2) - 1
        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True)
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(self, x, edge_index):
        feat_maps = self.backbone(x)
        feat_map = feat_maps[-1]  # [B, C, H, W]

        # Initial image-conditioned prediction
        initial_coords = self.init_head(feat_map).view(-1, self.num_landmarks, 2)

        # GCN refinement
        coords = initial_coords.clone()
        B, N = coords.shape[:2]

        batch_edge_index = torch.cat([edge_index + b * N for b in range(B)], dim=1)

        for _ in range(self.num_iters):
            node_feats = self.sample_features(feat_map, coords)
            node_feats = F.relu(self.node_feat_proj(node_feats))
            h = node_feats.view(B * N, -1)
            for layer in self.gnn_layers:
                h = F.relu(layer(h, batch_edge_index))
            delta = self.delta_head(h).view(B, N, 2)
            coords = coords + delta

        return initial_coords, coords  # return both for dual loss
```

Key differences from `HRNetGNN`:
- `forward` takes only `(x, edge_index)` — no `initial_coords` argument
- Returns `(initial_coords, final_coords)` — both supervised during training
- `init_head` uses `AdaptiveAvgPool2d` to collapse spatial dims before the MLP

### 2.3 Training Loop Changes

**New file: `alternative-models/hrnet-gcn/train_hinit.py`**

Dual loss supervision:

```python
initial_coords, final_coords = model(imgs, edge_index)

loss_init  = criterion(initial_coords, coords_gt)
loss_final = criterion(final_coords, coords_gt)
loss = 0.5 * loss_init + 1.0 * loss_final  # weight final output higher

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

The 0.5 weight on `loss_init` ensures the initial prediction head trains toward useful coordinates while the GCN output remains the primary objective.

Checkpoint saves `final_coords` val loss as the primary metric for model selection.

### 2.4 Evaluation Integration

`evaluate.py` needs a new runner function `run_hrnet_gcn_hinit` that:
- Loads `HRNetGNNWithInit` instead of `HRNetGNN`
- Calls `model(img_tensor, edge_index)` and takes `final_coords` (index 1 of the tuple)
- Otherwise identical to `run_hrnet_gcn`

Checkpoint naming: `hrnet_gcn_hinit_best.pth` — discovered by `discover_checkpoint` via the existing candidate list pattern.

### 2.5 Config

**New file: `alternative-models/hrnet-gcn/hinit-config.json`**

```json
{
    "training_data_path": "/storage/ice-shared/cs8903onl/alternative-models/data",
    "num_landmarks": 9,
    "gnn_hidden": 128,
    "num_layers": 2,
    "num_iters": 4,
    "input_size": 512,
    "epochs": 150,
    "batch_size": 16,
    "lr": 1e-4,
    "train_val_split": 0.9,
    "loss_init_weight": 0.5,
    "loss_final_weight": 1.0
}
```

No `mean_shape_path` — this model computes its own initial coords from the image.

---

## Part 3 — Comparison Protocol

### On Lizard (Model B validation)

Run both models on the same lizard split and compare:

```bash
# Model A — existing mean init (already trained)
python alternative-models/hrnet-gcn/train.py \
    --split alternative-models/benchmarking/splits/split.json

# Model B — HRNet init
python alternative-models/hrnet-gcn/train_hinit.py \
    --split alternative-models/benchmarking/splits/split.json
```

Regenerate the benchmark report:
```bash
python alternative-models/evaluate.py
```

Expected comparison:
- If Model B mean pixel error < Model A on Lizard → proceed to WFLW
- If Model B ≈ Model A → proceed anyway, WFLW's higher variance is the real test
- If Model B > Model A significantly → debug init head before WFLW migration

### On WFLW (after Lizard validation)

Run Model A at all fractions first to establish the baseline degradation curve. Then run Model B at the same fractions. The crossover point (if any) between the two init strategies is the primary experimental finding.

---

## Part 4 — File Structure

```
alternative-datasets/
  WFLW_MIGRATION.md              ← this document
  common/
    split_utils.py               ← NEW: shared fraction sampling + JSON writing
    graph_topologies.py          ← NEW: topology registry (chain, wflw, ...)
  wflw/
    preprocess.py                ← NEW: WFLW JPEGs + annotations → .pt crops
    generate_split.py            ← NEW: fraction-aware split generator
    compute_mean_shape.py        ← NEW: per-landmark mean coords from training set
    graph_topology.py            ← NEW: make_wflw_edge_index() + flip map
    evaluate_wflw.py             ← NEW: NME + per-subset breakdown
    configs/
      gcn_wflw.json              ← NEW: Model A config for WFLW

alternative-models/hrnet-gcn/
  hrnet_gcn.py                   ← FROZEN — do not modify
  hrnet_gcn_hinit.py             ← NEW: Model B architecture
  train.py                       ← FROZEN — do not modify
  train_hinit.py                 ← NEW: Model B training script
  utils.py                       ← FROZEN — do not modify
  lizard_dataset.py              ← FROZEN — do not modify
  default-config.json            ← FROZEN — do not modify
  wflw-config.json               ← NEW: Model A WFLW config
  hinit-config.json              ← NEW: Model B lizard config
```

---

## Part 5 — Work Estimate

### Model A — WFLW migration

| Task | Effort |
|---|---|
| `preprocess.py` | 0.5 day |
| `generate_split.py` + `split_utils.py` | 0.5 day |
| `compute_mean_shape.py` | 1 hour |
| `make_wflw_edge_index()` + flip map | 2 hours |
| `graph_topologies.py` registry | 1 hour |
| Noise-augmented init in training loop | 1 hour |
| `evaluate_wflw.py` | 0.5 day |
| Config + first training run | 1 day |
| **Model A total** | **~3.5 days** |

### Model B — HRNet init

| Task | Effort |
|---|---|
| `hrnet_gcn_hinit.py` | 2 hours |
| `train_hinit.py` | 2 hours |
| `run_hrnet_gcn_hinit` in `evaluate.py` | 1 hour |
| `hinit-config.json` | 30 min |
| Lizard training run + comparison | 0.5 day |
| **Model B total** | **~1.5 days** |

**Total: ~5 days**

---

## Part 6 — Dependencies

No new Python dependencies. `torch_geometric` is already required. WFLW annotation parsing requires only standard library and numpy.
