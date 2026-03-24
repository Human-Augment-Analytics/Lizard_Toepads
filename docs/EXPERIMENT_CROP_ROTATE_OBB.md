# Experiment: Crop+Rotate OBB for Tighter Dlib Bounding Boxes

**Date**: 2026-02-14
**Status**: Failed — baseline outperforms crop+rotate OBB

## Motivation

When converting YOLO-OBB oriented bounding boxes to axis-aligned rectangles for dlib shape predictor training/inference, the previous approach discarded rotation info by taking min/max of OBB corners, then adding 30% padding. For highly rotated limbs, this creates large bounding boxes with excessive background. The worst test images showed 300–1100px landmark error, hypothesized to be caused by the predictor struggling with so much background noise.

**Hypothesis**: By cropping and rotating the image so the OBB becomes upright, we can use tight axis-aligned bounding boxes (10% padding) that match the actual limb geometry, reducing error.

## Approach

### Crop+Rotate Pipeline

Instead of: `OBB corners → axis-aligned rect → 30% padding`

Do: `Crop 2x region around OBB center → Rotate crop by -angle → OBB is now upright → Tight bbox from OBB w,h + 10% padding`

This is applied identically in training preprocessing and inference:

1. **Preprocessing** (`generate_yolo_bbox_xml.py`):
   - Run YOLO-OBB on full image, match detection to GT landmarks by centroid proximity
   - Crop a 2x-OBB-size region around the OBB center
   - Rotate the crop by `-angle` so the OBB becomes axis-aligned
   - Transform GT landmarks through the same affine rotation
   - Save rotated crop to `crops_obb_{toe,finger}/`
   - Compute tight bbox from OBB dimensions + 10% padding
   - Write XML pointing to rotated crops with transformed landmarks and tight bboxes

2. **Inference** (`predict_landmarks_flip.py`):
   - Same crop+rotate, predict landmarks in rotated crop space
   - Inverse-transform predicted landmarks back to original image coordinates

### Key Design Decisions

- **Crop first, then rotate**: Images are 20K x 10K; rotating the full image wastes memory. Crop a ~2000x2000 region first.
- **Bbox from OBB dimensions, not landmarks**: At inference we don't have GT landmarks, so the bbox must come from OBB w,h. Training matches this.
- **10% padding** (vs 30%): Much less padding needed since there's no rotation-induced dead space.

### Files Modified

| File | Changes |
|------|---------|
| `ml-morph/scripts/preprocessing/generate_yolo_bbox_xml.py` | Added `crop_and_rotate()`, `transform_landmarks()`, `--crops-dir`, `--crop-scale`, `--no-rotation` flags. Switched PIL→cv2. |
| `scripts/inference/predict_landmarks_flip.py` | Added `crop_and_rotate_for_inference()`, `get_dlib_rect_from_obb_rotated()` with inverse transform. Added `--no-rotation`, `--padding-ratio`, `--crop-scale` flags. |
| `sbatch/preprocess_obb.sbatch` | Updated padding 0.3→0.1, added `--crops-dir` and `--crop-scale` args |
| `ml-morph/.gitignore` | Added `crops_obb*/` |

## Preprocessing Results

YOLO-OBB model: `runs/obb/H5_obb_noflip/weights/best.pt` (6-class, bottom-only)

| Split | Total | OBB-Rotated | Fallback | Detection Rate |
|-------|-------|-------------|----------|----------------|
| Toe train | 163 | 157 (96%) | 6 (4%) | 96% |
| Toe test | 41 | 39 (95%) | 2 (5%) | 95% |
| Finger train | 163 | 157 (96%) | 6 (4%) | 96% |
| Finger test | 41 | 39 (95%) | 2 (5%) | 95% |

Crop images saved to `ml-morph/crops_obb_toe/` (191 images) and `ml-morph/crops_obb_finger/` (196 images).

Typical rotated crop sizes: ~500–1200px (vs original 12K–20K full images).

## Hyperparameter Search Results

144 configurations searched per limb type (same grid as baseline):

### Best Models Comparison

| Metric | Toe Baseline | Toe Crop+Rotate | Finger Baseline | Finger Crop+Rotate |
|--------|-------------|-----------------|-----------------|-------------------|
| **Best Test Error (px)** | **83.27** | 92.02 | **40.64** | 76.11 |
| **Best Train Error (px)** | 5.90 | 153.36 | 1.96 | 51.64 |
| **Overfitting Gap** | +77.37 | -61.34 | +38.68 | +24.47 |
| **Best Config** | d2_c18_nu0.1_t400 | d2_c25_nu0.1_t700 | d2_c15_nu0.1_t400 | d2_c12_nu0.1_t400 |

**Baseline wins both**: Finger 1.9x better, Toe 1.1x better.

### Earlier OBB Results (Axis-Aligned, Pre-Crop+Rotate)

For reference, a previous OBB run using axis-aligned conversion with 30% padding (same YOLO-OBB model but no crop+rotate) achieved:
- Toe test error: **30.72 px** (much better than both baseline 83.27 and crop+rotate 92.02)
- Finger test error: **38.25 px** (better than both baseline 40.64 and crop+rotate 76.11)

**The crop+rotate approach made OBB results significantly worse, not better.**

## Worst-5 Error Analysis

### Toe (Crop+Rotate OBB)

| Rank | Image | Mean Error (px) | Max Error (px) |
|------|-------|----------------|----------------|
| 1 | 1794 | 567.5 | 774.6 |
| 2 | 1822 | 337.7 | 547.0 |
| 3 | 1452 | 301.4 | 408.6 |
| 4 | 1697 | 288.0 | 306.6 |
| 5 | 1019 | 260.4 | 343.6 |

Overall: Mean=124.8, Median=87.8, Std=113.6

### Finger (Crop+Rotate OBB)

| Rank | Image | Mean Error (px) | Max Error (px) |
|------|-------|----------------|----------------|
| 1 | 1444 | 1154.6 | 1242.8 |
| 2 | 1153 | 152.3 | 284.6 |
| 3 | 1103 | 144.4 | 246.1 |
| 4 | 1093 | 134.3 | 247.1 |
| 5 | 1816 | 108.1 | 158.1 |

Overall: Mean=76.2 (inflated by #1444), Median=38.4

## Root Cause Analysis

### 1. Landmark Transformation Bug (Finger #1444)

The worst finger outlier (1154px error) has GT landmarks at coordinates `(1258, -322)` in a 655x655 rotated crop — completely outside the image. The predicted landmarks are correctly on the hand, but GT was transformed to an impossible location.

**Cause**: When the YOLO OBB detection center is offset from the GT landmark centroid (e.g., detection matches a nearby limb or the OBB center doesn't overlap the GT landmarks well), the crop may not fully contain the landmarks. After rotation, landmarks that were barely inside the crop get transformed outside the rotated canvas.

### 2. Inverted Overfitting (Toe: Train 153 > Test 92)

The training error being higher than test error is highly unusual. Possible explanations:
- The training set has more difficult cases (rotations, occlusions) that the predictor can't fit well
- The crop+rotate transformation introduces artifacts in training that degrade learning
- The 10% padding is too tight for training — the predictor needs more context than the OBB dimensions provide

### 3. High Training Error Generally

Baseline train errors are 1–6px (predictors fit training data almost perfectly). Crop+rotate train errors are 50–150px, meaning the predictor fundamentally struggles to learn from rotated crops. This suggests:
- The rotated crops lose spatial context the predictor relies on
- The rotation interpolation degrades image quality
- The tight bounding boxes constrain the predictor's feature pool too much

### 4. Why Crop+Rotate Hurt vs. Axis-Aligned OBB

The earlier axis-aligned OBB approach (30% padding, no rotation) achieved 30.72px toe error — **3x better** than the crop+rotate attempt. The axis-aligned approach preserves:
- Original image quality (no rotation interpolation)
- More context around the limb (30% padding > 10% padding)
- Consistent coordinate system (no affine transforms that can misalign landmarks)

## Conclusions

1. **Crop+rotate OBB does not improve landmark prediction.** The approach was motivated by wanting tighter boxes, but the dlib shape predictor actually benefits from larger context regions and unmodified images.

2. **The axis-aligned OBB conversion with 30% padding was already the better approach.** It achieved the best results across all conditions tested.

3. **Rotation introduces harmful artifacts.** Image interpolation from rotation, potential landmark misalignment, and loss of spatial context all degrade predictor performance.

4. **There is a landmark transformation bug** affecting cases where the YOLO detection center doesn't closely match the GT landmark centroid. This needs to be fixed if crop+rotate is revisited.

## Recommendations

- **Revert to axis-aligned OBB conversion** with 30% padding (the original approach)
- **Investigate why earlier OBB results (30.72/38.25) can't be reproduced** — the current axis-aligned baseline gets 83.27/40.64, which is much worse for toe
- **Keep the crop+rotate code** behind `--no-rotation` flag (default) for potential future experiments
- Consider **data-specific** approaches for the worst outliers rather than a blanket rotation strategy

## Reproduction

```bash
# Preprocessing (generates rotated crops + XMLs)
sbatch sbatch/preprocess_obb.sbatch

# Hyperparameter search
sbatch sbatch/hyperparam_search_toe_obb.sbatch
sbatch sbatch/hyperparam_search_finger_obb.sbatch

# Worst-5 analysis (from ml-morph/ directory, with lizard conda env)
python scripts/debug/investigate_test_error.py \
    --test-xml toe_test_yolo_obb.xml \
    --model hyperparam_results_toe_obb/depth2_cascade25_nu0.1_trees700_over30_fp500_splits20_tj0.dat \
    --image-dir . --output-dir worst_predictions_toe_obb

python scripts/debug/investigate_test_error.py \
    --test-xml finger_test_yolo_obb.xml \
    --model hyperparam_results_finger_obb/depth2_cascade12_nu0.1_trees400_over30_fp500_splits20_tj0.dat \
    --image-dir . --output-dir worst_predictions_finger_obb
```

## File Locations

| Artifact | Path |
|----------|------|
| Preprocessing script | `ml-morph/scripts/preprocessing/generate_yolo_bbox_xml.py` |
| Inference script | `scripts/inference/predict_landmarks_flip.py` |
| Preprocessing sbatch | `sbatch/preprocess_obb.sbatch` |
| Toe OBB results | `ml-morph/hyperparam_results_toe_obb/` |
| Finger OBB results | `ml-morph/hyperparam_results_finger_obb/` |
| Rotated toe crops | `ml-morph/crops_obb_toe/` |
| Rotated finger crops | `ml-morph/crops_obb_finger/` |
| Toe worst-5 visualizations | `ml-morph/worst_predictions_toe_obb/` |
| Finger worst-5 visualizations | `ml-morph/worst_predictions_finger_obb/` |
