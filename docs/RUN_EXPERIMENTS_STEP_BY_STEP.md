# Step-by-Step Instructions: Running ml-morph Experiments

This document provides exact, sequential instructions for reproducing the `ml-morph` landmark prediction experiments (Baseline vs. OBB) on the PACE ICE cluster.

### Prerequisites & Environment Setup

Ensure you are logged into the PACE ICE cluster and set your environment variables, then activate your environment.

```bash
# Set your user environment variable (replace with your actual username, e.g., yloh30)
export USER="your_username"
export PROJECT_ROOT="/home/hice1/$USER/scratch/Lizard_Toepads"

# Navigate to the project directory
cd $PROJECT_ROOT

# Update generic placeholders in YAML files with your actual path 
# (YOLO does not dynamically interpolate bash environment variables)
find configs ml-morph/configs -name "*.yaml" -type f -exec sed -i "s|/home/hice1/YOUR_USERNAME/scratch/Lizard_Toepads|$PROJECT_ROOT|g" {} +

# Load modules and activate the conda environment
module load anaconda3
source activate lizard

# (Optional) Ensure dependencies are up to date 
uv sync
```

### Step 1: Preprocessing Data (TPS → XML → YOLO Bounding Boxes)

First, convert the TPS landmark files to dlib XML format and generate the YOLO bounding box detections for training.

```bash
# 1. Run Baseline preprocessing (Axis-Aligned YOLO detections, 6-class model)
sbatch sbatch/preprocess_baseline.sbatch

# 2. Run OBB preprocessing (YOLO-OBB oriented detections converted to upright boxes)
sbatch sbatch/preprocess_obb.sbatch
```

Wait for these jobs to successfully complete. They will populate the `ml-morph/` directory with training and testing XMLs (e.g., `toe_train_yolo_baseline.xml`, `toe_train_yolo_obb.xml`, etc.).

### Step 2: Training dlib Shape Predictors

Once preprocessing is complete, you can train the models.

```bash
# 1. Train Baseline models (Toe and Finger)
sbatch sbatch/train_mlmorph_toe.sbatch
sbatch sbatch/train_mlmorph_finger.sbatch

# Note: The above scripts can be easily modified to train the OBB configuration
# by changing the input XML files in the script to point to the *_obb.xml files.
```

### Step 3: Hyperparameter Grid Search (Optional)

If you wish to run the 144-configuration hyperparameter search to find the optimal `tree_depth`, `cascade_depth`, `nu`, and `num_trees`:

```bash
# Run Baseline Hyperparameter Search
sbatch sbatch/hyperparam_search_toe_baseline.sbatch
sbatch sbatch/hyperparam_search_finger_baseline.sbatch

# Run OBB Hyperparameter Search
sbatch sbatch/hyperparam_search_toe_obb.sbatch
sbatch sbatch/hyperparam_search_finger_obb.sbatch
```

### Step 4: End-to-End Inference (Predicting Landmarks)

To run the complete inference pipeline on test images using the Flip-Inference strategy (which predicts `up_toe`/`up_finger` by flipping the image and using the `bot_toe`/`bot_finger` models):

```bash
# Run the flipped inference script
uv run python ml-morph/scripts/inference/predict_landmarks_flip.py \
    --input-dir /storage/ice-shared/cs8903onl/miami_fall_24_jpgs \
    --output-dir inference_results \
    --yolo-model runs/obb/H5_obb_noflip/weights/best.pt \
    --toe-predictor ml-morph/toe_predictor_yolo_bbox.dat \
    --finger-predictor ml-morph/finger_predictor_yolo_bbox.dat
```

This will produce the final overlaid visual predictions and coordinates.
