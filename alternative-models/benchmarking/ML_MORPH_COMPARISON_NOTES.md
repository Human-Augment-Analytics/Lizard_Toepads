# ml-morph Comparison Notes

## What is ml-morph?

ml-morph is the **baseline model** this project is trying to beat. It is a two-stage landmark detection pipeline:

1. **Object detector** (HOG + SVM via dlib) — detects and crops the region of interest from the full image
2. **Shape predictor** (cascade regression trees via dlib) — predicts landmark coordinates within the detected bounding box

The alternative deep learning models (Stacked Hourglass, ViT, HRNet, HRNet-GCN) were all developed to outperform ml-morph on the lizard toepad landmark task.

---

## Output Format

ml-morph predictions are written to a **dlib XML file** (`output.xml` by default) via `prediction.py`.

### XML structure:
```xml
<dataset>
  <images>
    <image file="path/to/image.jpg">
      <box top="..." left="..." width="..." height="...">
        <part name="0" x="123" y="456"/>
        <part name="1" x="234" y="567"/>
        ...
      </box>
    </image>
  </images>
</dataset>
```

Coordinates are in **image-space** (top-left origin, y increases downward). The `dlib_xml_to_pandas` utility in `ml-morph/utils/utils.py` imports predictions into a pandas DataFrame with columns `X0, Y0, X1, Y1, ..., X8, Y8`.

---

## Integration Assessment: Much Easier Than It Looks

After reviewing the full ml-morph codebase, the integration is **significantly simpler** than a naive approach would suggest. The repo already contains almost everything needed.

### What already exists in ml-morph/

| Script | What it does | Relevance |
|---|---|---|
| `scripts/preprocessing/generate_yolo_bbox_xml.py` | Runs YOLO OBB on full images, crops+rotates around each detection, transforms GT landmarks to crop-local coords, saves crop `.jpg` files and dlib XML | **This is the key script** — it produces crops in the same coordinate space as the alternative models |
| `scripts/preprocessing/tps_to_xml.py` | Converts TPS files to dlib XML with y-axis flip | Used upstream of `generate_yolo_bbox_xml.py` |
| `scripts/preprocessing/split_train_val_test.py` | Splits an existing XML into train/val/test subsets | Can be adapted to filter by our benchmarking split |
| `configs/toe_training_yolo_obb.yaml` | Config for OBB-based shape predictor training | Ready to use |
| `shape_trainer.py` | Trains the cascade regression tree predictor | Takes `train.xml`, outputs `.dat` file |
| `shape_tester.py` | Evaluates predictor, reports **average pixel deviation** | Same metric as ViT/HRNet-GCN pixel error |
| `prediction.py` | Runs inference on a directory of images | Produces `output.xml` with predicted landmarks |

### The key insight

`generate_yolo_bbox_xml.py` with `--crops-dir` already:
- Runs YOLO OBB on full lizard images
- Crops and rotates each toepad region using the OBB homography
- Transforms GT landmark coordinates into the crop-local coordinate system
- Saves the crop as a `.jpg` file

This is **exactly the same preprocessing pipeline** as the alternative models. The crops it produces are in the same 512×512-ish space. So ml-morph can be trained and evaluated on the same crop images.

---

## Steps to Integrate ml-morph into the Benchmark

### Prerequisites
1. `dlib` must be installed: `pip install dlib` (requires cmake/boost — usually available on HPC)
2. The OBB YOLO model must be available (same `obb.pt` used by the alternative models)
3. The consolidated TPS file must exist: `ml-morph/consolidated_toe.tps`

### Step 1: Generate the OBB crop XML (one-time setup)

From the `ml-morph/` directory:

```bash
# First generate base XML from TPS
python scripts/preprocessing/tps_to_xml.py \
  -t consolidated_toe.tps \
  -i /storage/ice-shared/cs8903onl/hourglass-data/raw_data/miami_fall_24_jpgs \
  --output-train toe_train_base.xml \
  --output-test toe_test_base.xml \
  --seed 42

# Then replace bounding boxes with OBB crops
python scripts/preprocessing/generate_yolo_bbox_xml.py \
  --input-xml toe_train_base.xml \
  --output-xml toe_train_obb.xml \
  --yolo-model ../yolo/obb.pt \
  --crops-dir obb_crops/ \
  --target-class toe

python scripts/preprocessing/generate_yolo_bbox_xml.py \
  --input-xml toe_test_base.xml \
  --output-xml toe_test_obb.xml \
  --yolo-model ../yolo/obb.pt \
  --crops-dir obb_crops/ \
  --target-class toe
```

This produces `obb_crops/` containing `.jpg` crop files and `toe_train_obb.xml` / `toe_test_obb.xml` with crop-local landmark coordinates.

### Step 2: Filter XML to match the benchmarking split

Write a small adapter script (~50 lines) `benchmarking/ml_morph_adapter.py` that:
- Reads `splits/hrnet_gcn_split.json` (or any model's split — they all use the same image IDs)
- Extracts image IDs from the `.pt` filenames (e.g. `1075_0_b.pt` → ID `1075`)
- Filters `toe_train_obb.xml` to only include those image IDs → `ml_morph_train.xml`
- Filters `toe_test_obb.xml` to only include val image IDs → `ml_morph_val.xml`

This is the **only new code needed** — roughly 50 lines.

### Step 3: Train ml-morph on the filtered split

```bash
cd ml-morph
python shape_trainer.py \
  -d ml_morph_train.xml \
  -t ml_morph_val.xml \
  -o toe_predictor_benchmark \
  -th 8 -dp 4 -c 15 -nu 0.1 -os 20
```

Output: `toe_predictor_benchmark.dat`
Validation metric printed: `Testing error (average pixel deviation): X.XX`

### Step 4: Run inference on val crops for overlays

```bash
python prediction.py \
  -i obb_crops/ \
  -d toe_detector.svm \
  -p toe_predictor_benchmark.dat \
  -o ml_morph_val_predictions.xml
```

Parse `ml_morph_val_predictions.xml` with `dlib_xml_to_pandas` to get `(x, y)` coordinates per image, then draw overlays using the same `draw_overlay` function in `generate_report.py`.

### Step 5: Add to report

In `generate_report.py`:
- Add ml-morph to the summary table with its average pixel deviation
- Add overlay images from the val crops
- ml-morph has no epoch-by-epoch training curves, so it appears only in the summary table and overlay section

---

## Effort Estimate

| Task | Effort |
|---|---|
| Install dlib on cluster | 15–30 min |
| Run `tps_to_xml.py` + `generate_yolo_bbox_xml.py` (one-time) | 1–2 hours (YOLO inference on all images) |
| Write `ml_morph_adapter.py` (split filter) | 1–2 hours |
| Train ml-morph shape predictor | 30–60 min |
| Add to `generate_report.py` | 1–2 hours |
| **Total** | **~half a day** |

---

## How to Run the Full Benchmark Report

### Prerequisites

1. All four models must have been preprocessed (`.pt`/`.npz` files exist in their data directories)
2. Update `benchmark_config.json` with the correct absolute paths for your environment
3. The `lizard` conda environment must be active: `conda activate lizard`

### Step 1: Generate the shared split

```bash
cd alternative-models/benchmarking
python generate_split.py --config benchmark_config.json
```

This writes `splits/stacked_hourglass_split.json`, `splits/vit_split.json`, `splits/hrnet_split.json`, `splits/hrnet_gcn_split.json`.

### Step 2: Train all models

```bash
python run_all.py --config benchmark_config.json
```

This runs all four `train.py` scripts sequentially, each using the shared split. Checkpoints are saved to each model's `checkpoints/` directory. Logs are written to each model's `logs/` directory.

To run a single model manually (e.g. for debugging):

```bash
cd alternative-models/hrnet
python train.py \
  --config default \
  --data /path/to/hrnet/data \
  --split ../benchmarking/splits/hrnet_split.json
```

### Step 3: Generate the report

```bash
cd alternative-models/benchmarking
python generate_report.py --config benchmark_config.json
```

This produces:
- `report/benchmark_report.html` — self-contained HTML with embedded plots and overlay images
- `report/benchmark_summary.md` — Markdown summary table
- `report/overlays/*.png` — 5 overlay images per model

Open `report/benchmark_report.html` in any browser to view the full comparison.

### Running on the cluster (sbatch)

Each model's training can be submitted as a separate sbatch job. Example for HRNet:

```bash
#!/bin/bash
#SBATCH --job-name=hrnet_train
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

conda activate lizard
cd /path/to/alternative-models/hrnet
python train.py \
  --config default \
  --data /home/hice1/axu39/lizard/Lizard_Toepads/stacked-hourglass/data/hrnet \
  --split /path/to/alternative-models/benchmarking/splits/hrnet_split.json
```

After all jobs complete, run `generate_report.py` from the login node.

---

## File Reference

```
alternative-models/benchmarking/
├── benchmark_config.json       Edit paths here before running
├── generate_split.py           Run first — creates split JSON files
├── run_all.py                  Run second — trains all models sequentially
├── generate_report.py          Run third — produces HTML + Markdown report
├── splits/                     Auto-generated by generate_split.py
│   ├── stacked_hourglass_split.json
│   ├── vit_split.json
│   ├── hrnet_split.json
│   └── hrnet_gcn_split.json
└── report/                     Auto-generated by generate_report.py
    ├── benchmark_report.html
    ├── benchmark_summary.md
    └── overlays/
        ├── stacked_hourglass_0.png ... _4.png
        ├── vit_0.png ... _4.png
        ├── hrnet_0.png ... _4.png
        └── hrnet_gcn_0.png ... _4.png
```
