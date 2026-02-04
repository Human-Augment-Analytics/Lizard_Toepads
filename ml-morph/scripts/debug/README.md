# Debug and Experimental Scripts

This directory contains scripts used for debugging, one-off experiments, and exploratory analysis. These are kept for reference but are not actively maintained or part of the core workflow.

## Contents (17 scripts)

### Debug/Analysis Scripts (6)
- `convert_errors_to_mm.py` - Convert pixel errors to millimeters
- `detect_outliers.py` - Detect outliers in predictions
- `detect_tps_outliers.py` - Detect outliers in TPS files
- `investigate_test_error.py` - Investigate test errors
- `test_preprocessing.py` - Test preprocessing functions
- `trace_coordinates.py` - Debug coordinate transformations

### Visualization Scripts (2)
- `show_scale_images.py` - Display scale images for inspection
- `show_single_prediction.py` - Visualize single prediction result

### One-Off Training Experiments (4)
- `train_finger_cropped.py` - Train on cropped finger regions
- `train_scale_cropped.py` - Train on cropped scale regions
- `train_toe_cropped.py` - Train on cropped toe regions
- `train_toe_full.py` - Train on full toe images

**Note**: Use `../../pytorch_keypoint/train_keypoints.py` for new training work.

### One-Off Preprocessing Experiments (5)
- `preprocess_cropped_finger.py` - Specific finger preprocessing
- `preprocess_scale_yolo.py` - YOLO-specific scale preprocessing
- `preprocess_scale_yolo_full.py` - Full scale YOLO preprocessing
- `preprocess_toe_full.py` - Full toe preprocessing
- `inference_preprocess.py` - Inference-specific preprocessing

## Note

These scripts are **not actively maintained** and may have dependencies on outdated code structures. They are kept for historical reference only.

For active development, use:
- **Core scripts**: `../../preprocessing.py`, `../../detector_trainer.py`, etc.
- **Utilities**: `../preprocessing/` (TPS manipulation), `../training/` (hyperparameter search)
- **Modern approach**: `../../pytorch_keypoint/` (recommended)
