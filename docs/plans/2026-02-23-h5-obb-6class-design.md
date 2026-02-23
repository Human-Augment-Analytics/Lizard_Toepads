# Design: H5_obb_6class — OBB Model with Active Ruler + ID

**Date:** 2026-02-23
**Branch:** leyang/ml-morph

## Overview

Create a new YOLO-OBB model `H5_obb_6class` that extends `H5_obb_noflip` by adding
ground-truth OBB annotations for the `id` class (class 5). The `ruler` class (class 4)
is already annotated in the noflip dataset. Upper-view classes (`up_finger`=0,
`up_toe`=1) remain empty in training and are handled at inference time via the
vertical-flip strategy.

## Class Scheme

| Class | Name       | Training data      | Inference source        |
|-------|------------|--------------------|-------------------------|
| 0     | up_finger  | empty              | flip of bot_finger (2)  |
| 1     | up_toe     | empty              | flip of bot_toe (3)     |
| 2     | bot_finger | OBB from TPS       | standard pass           |
| 3     | bot_toe    | OBB from TPS       | standard pass           |
| 4     | ruler      | OBB from processed_obb | standard pass       |
| 5     | id         | OBB from _id.TPS   | standard pass           |

## Data Sources

- **bot_finger / bot_toe** (classes 2, 3): `data/dataset_obb_6class` labels
- **ruler** (class 4): `data/processed_obb/labels` — class 2 remapped to 4
- **id** (class 5): `/storage/ice-shared/cs8903onl/tps_files/<stem>_id.TPS` — 848
  files, each with 2 diagonal corner points (TPS bottom-left origin coords)

## ID OBB Generation

Each `_id.TPS` file contains 2 landmarks that are opposite diagonal corners of the
ID label rectangle. The conversion:

1. Parse 2 points in TPS coords `(x, y)` with bottom-left origin
2. Convert to image coords: `(x, img_height - y)`
3. Derive 4 axis-aligned corners from the 2 diagonal points
4. Normalize by image dimensions → YOLO OBB format (class 5, 8 normalized coords)

## Files

### New

1. **`scripts/preprocessing/generate_id_obb_from_tps.py`**
   - Args: `--tps-dir`, `--image-dir`, `--output-dir`
   - For each `<stem>_id.TPS`: read 2 corners, convert coords, write class 5 OBB label
   - Output: `data/id_obb_from_tps/<stem>.txt`

2. **`scripts/preprocessing/create_6class_active_obb_dataset.py`**
   - Same logic as `create_noflip_obb_dataset.py` for classes 2, 3, 4
   - Additionally merges id OBB from `data/id_obb_from_tps/`
   - Output: `data/dataset_obb_6class_active`

3. **`configs/H5_obb_6class.yaml`**
   - Clone of `H5_obb_noflip.yaml` with `path` updated to `data/dataset_obb_6class_active`
   - Same 6-class names, same training hyperparameters

### Updated

4. **`scripts/inference/inference_with_flip.py`**
   - Standard pass: keep classes 2 (bot_finger), 3 (bot_toe), 4 (ruler), 5 (id)
   - Flipped pass: class 2 → 0 (up_finger), class 3 → 1 (up_toe); unflip corners
   - Currently uses 2-class indices (0, 1); needs remapping to 6-class indices

## Dataset Output

`data/dataset_obb_6class_active/`
```
images/
  train/   # symlinked from dataset_obb_6class
  val/
labels/
  train/   # classes 2, 3, 4, 5 per image (0 and 1 absent)
  val/
```

## Training

Same hyperparameters as `H5_obb_noflip`. Run via:
```bash
yolo obb train cfg=configs/H5_obb_6class.yaml
```
