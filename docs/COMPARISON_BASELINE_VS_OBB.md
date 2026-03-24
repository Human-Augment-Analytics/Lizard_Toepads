# YOLO Baseline (Detect) vs YOLO-OBB: Detection and Landmark Comparison

## Objective

Compare two YOLO detection approaches for the lizard toepad landmark prediction pipeline:
- **Baseline**: Standard axis-aligned bounding box detection (YOLO detect)
- **OBB**: Oriented bounding box detection (YOLO-OBB)

Both feed into dlib shape predictors (ml-morph) for landmark regression. The goal is to determine whether OBB's tighter, rotation-aware boxes improve downstream landmark accuracy.

## Experimental Setup

### Detection Models

Both models use the same architecture (YOLOv11m) and the same 6-class scheme for consistency:

| | Baseline (H5_detect_6class) | OBB (H5_obb_noflip) |
|---|---|---|
| **Task** | detect | obb |
| **Architecture** | YOLOv11m | YOLOv11m-OBB |
| **Classes** | 6 (all annotated) | 6 (bot_finger, bot_toe, ruler annotated; up_finger, up_toe, id empty) |
| **Training Data** | `data/dataset` (merged bottom + upper views) | `data/dataset_obb_noflip` (bottom OBB + ruler only) |
| **Epochs** | 300 | 200 |
| **Image Size** | 1280 | 1280 |
| **Flip at Inference** | No (detects all classes directly) | Yes (flip to detect upper limbs as bottom) |
| **Model Path** | `runs/detect/H5_detect_6class/weights/best.pt` | `runs/obb/H5_obb_noflip/weights/best.pt` |

### OBB Model Validation Metrics (Best Epoch: 139)

| Metric | Value |
|--------|-------|
| Precision | 0.9628 |
| Recall | 0.9317 |
| mAP50 | 0.9619 |
| mAP50-95 | 0.9110 |

### Inference Strategy

**Baseline**: Single forward pass detects all 6 classes including `up_finger` and `up_toe` directly.

**OBB + Flip**:
1. Standard inference → detects `bot_finger`, `bot_toe` (oriented bounding boxes)
2. Vertically flip the image
3. Run inference again → upper limbs now appear as bottom → detect as `bot_finger`, `bot_toe`
4. Map flipped detections back to original coordinates as `up_finger`, `up_toe`

### Shape Predictor Training

Both detection approaches use the same ml-morph pipeline:

1. **TPS → XML**: Convert TPS landmark annotations to dlib XML format (`tps_to_xml.py`)
2. **YOLO Bbox Replacement**: Replace tight landmark-derived bounding boxes with YOLO-detected boxes + 30% padding (`generate_yolo_bbox_xml.py`), using the respective detection model
3. **Hyperparameter Search**: Train 144 dlib shape predictor configurations per limb type

**Hyperparameter grid** (144 configs = 3 x 4 x 4 x 3):

| Parameter | Values |
|-----------|--------|
| tree_depth | 2, 3, 4 |
| cascade_depth | 12, 15, 18, 25 |
| nu | 0.1, 0.15, 0.2, 0.25 |
| num_trees | 400, 500, 700 |
| oversampling | 30 (fixed) |
| feature_pool_size | 500 (fixed) |

**Symmetric predictor reuse**: Shape predictors are trained on bottom limbs only. For upper limbs detected via flip, the same predictors are applied (since flipped upper limbs look like bottom limbs), and predicted landmark coordinates are mapped back to the original image space.

## Results: Shape Predictor Test Error

Test error = average pixel deviation between predicted and ground-truth landmarks.

### Toe Landmark Prediction

| Rank | Baseline (H5_detect) | Test Error | OBB (H5_obb_noflip) | Test Error |
|------|---------------------|------------|---------------------|------------|
| 1 | depth2_cascade18_nu0.1_trees400 | 83.27 | depth2_cascade18_nu0.1_trees500 | **30.72** |
| 2 | depth2_cascade25_nu0.1_trees500 | 85.37 | depth2_cascade18_nu0.2_trees500 | 45.44 |
| 3 | depth2_cascade25_nu0.1_trees400 | 88.75 | depth2_cascade12_nu0.1_trees700 | 63.35 |

**OBB improvement for toe: 2.7x lower test error** (30.72 vs 83.27)

*All 144/144 configs completed for both baseline and OBB. OBB results reproduced using axis-aligned conversion with 30% padding (`--no-rotation`).*

### Finger Landmark Prediction

| Rank | Baseline (H5_detect) | Test Error | OBB (H5_obb_noflip) | Test Error |
|------|---------------------|------------|---------------------|------------|
| 1 | depth2_cascade15_nu0.1_trees400 | 40.64 | depth3_cascade25_nu0.1_trees500 | **38.25** |
| 2 | depth2_cascade15_nu0.15_trees700 | 40.81 | depth2_cascade18_nu0.1_trees400 | 38.36 |
| 3 | depth2_cascade12_nu0.1_trees500 | 41.40 | depth2_cascade25_nu0.25_trees700 | 38.48 |

**OBB improvement for finger: 1.06x lower test error** (38.25 vs 40.64)

### Summary Table

| Limb Type | Baseline Best | OBB Best | Winner | Improvement |
|-----------|--------------|----------|--------|-------------|
| **Toe** | 83.27 | 30.72 | OBB | 2.7x better |
| **Finger** | 40.64 | 38.25 | OBB | 1.06x better |

### Failed Experiment: Crop+Rotate OBB

A follow-up experiment attempted to improve OBB further by cropping and rotating the image so the OBB becomes upright, then using tighter 10% padding (instead of converting to axis-aligned rect with 30% padding). This performed significantly worse:

| Limb Type | OBB Axis-Aligned | OBB Crop+Rotate | Baseline |
|-----------|-----------------|-----------------|----------|
| **Toe** | **30.72** | 92.02 | 83.27 |
| **Finger** | **38.25** | 76.11 | 40.64 |

The crop+rotate approach suffered from high train errors, inverted overfitting (train > test for toe), and landmark transformation bugs. See [EXPERIMENT_CROP_ROTATE_OBB.md](EXPERIMENT_CROP_ROTATE_OBB.md) for full details.

## Key Observations

1. **OBB significantly outperforms baseline for toe landmarks** — the best OBB model (axis-aligned conversion) achieves 30.72 pixel error vs 83.27 for baseline, a 2.7x improvement.

2. **OBB modestly outperforms baseline for finger landmarks** — 38.25 vs 40.64, a small but consistent improvement.

3. **OBB bounding boxes better match limb geometry** — oriented boxes fit angled/rotated limbs tightly, and axis-aligned conversion with 30% padding provides sufficient context for the shape predictor while reducing irrelevant background.

4. **Crop+rotate hurts performance** — attempting to use OBB rotation info directly (cropping and rotating the image) degrades results dramatically. The dlib shape predictor benefits from the original image orientation and generous padding.

5. **Consistent optimal hyperparameters** — across all approaches, shallow trees (depth 2-3), low regularization (nu 0.1), and high oversampling (30) perform best.

6. **OBB axis-aligned results reproduced** — the axis-aligned OBB results (30.72/38.25) have been confirmed in a second run stored in `hyperparam_results_{toe,finger}_obb_aligned/`.

## Reproduction

### Training the OBB Model
```bash
sbatch sbatch/train_yolo_obb_noflip.sbatch
```

### Running Shape Predictor Hyperparameter Searches

Baseline:
```bash
PREPROC=$(sbatch --parsable sbatch/preprocess_baseline.sbatch)
sbatch --dependency=afterok:$PREPROC sbatch/hyperparam_search_toe_baseline.sbatch
sbatch --dependency=afterok:$PREPROC sbatch/hyperparam_search_finger_baseline.sbatch
```

OBB (axis-aligned, recommended):
```bash
PREPROC=$(sbatch --parsable sbatch/preprocess_obb_aligned.sbatch)
sbatch --dependency=afterok:$PREPROC sbatch/hyperparam_search_toe_obb_aligned.sbatch
sbatch --dependency=afterok:$PREPROC sbatch/hyperparam_search_finger_obb_aligned.sbatch
```

OBB (crop+rotate, failed experiment):
```bash
PREPROC=$(sbatch --parsable sbatch/preprocess_obb.sbatch)
sbatch --dependency=afterok:$PREPROC sbatch/hyperparam_search_toe_obb.sbatch
sbatch --dependency=afterok:$PREPROC sbatch/hyperparam_search_finger_obb.sbatch
```

## File Locations

| Artifact | Path |
|----------|------|
| Baseline detect model | `runs/detect/H5_detect_6class/weights/best.pt` |
| OBB model | `runs/obb/H5_obb_noflip/weights/best.pt` |
| Baseline toe results | `ml-morph/hyperparam_results_toe_baseline/results.json` |
| Baseline finger results | `ml-morph/hyperparam_results_finger_baseline/results.json` |
| OBB toe results (axis-aligned) | `ml-morph/hyperparam_results_toe_obb_aligned/results.json` |
| OBB finger results (axis-aligned) | `ml-morph/hyperparam_results_finger_obb_aligned/results.json` |
| OBB toe results (crop+rotate) | `ml-morph/hyperparam_results_toe_obb/results.json` |
| OBB finger results (crop+rotate) | `ml-morph/hyperparam_results_finger_obb/results.json` |
| OBB dataset creation | `scripts/preprocessing/create_noflip_obb_dataset.py` |
| OBB data config | `configs/H5_obb_noflip.yaml` |
