#!/bin/bash
#SBATCH --job-name=mlmorph_toe
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/mlmorph_toe_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@gatech.edu

# Exit on error
set -e

# Change to ml-morph directory
PROJECT_ROOT=/home/hice1/$USER/scratch/Lizard_Toepads
cd $PROJECT_ROOT/ml-morph

# Load necessary modules
module load anaconda3

# Activate environment
source activate lizard

echo "Starting ml-morph TOE training on $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# -------------------------------------------------------------------
# Step 1: Generate train/test XMLs from consolidated_toe.tps
# Uses tps_to_xml.py which creates dlib XML with landmark bboxes
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "STEP 1: Generating XMLs from consolidated_toe.tps"
echo "============================================================"

IMAGE_DIR=/storage/ice-shared/cs8903onl/miami_fall_24_jpgs

python scripts/preprocessing/tps_to_xml.py \
    -t consolidated_toe.tps \
    -i "$IMAGE_DIR" \
    --train-ratio 0.8 \
    --seed 42 \
    --output-train toe_train.xml \
    --output-test toe_test.xml

echo "Created toe_train.xml and toe_test.xml"
echo "Train images: $(grep -c '<image' toe_train.xml)"
echo "Test images:  $(grep -c '<image' toe_test.xml)"

# -------------------------------------------------------------------
# Step 2: Replace bounding boxes with YOLO OBB detections
# This makes training bboxes match what YOLO produces at inference
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "STEP 2: Generating YOLO-derived bounding boxes"
echo "============================================================"

YOLO_MODEL=$PROJECT_ROOT/runs/obb/H1_obb_2class2/weights/best.pt
PADDING=0.3
FALLBACK_PADDING=0.5
CONF=0.25
TARGET_CLASS=bot_toe

echo "YOLO model: $YOLO_MODEL"
echo "Padding ratio: $PADDING"
echo "Target class: $TARGET_CLASS"

# Generate YOLO bbox XML for train set
echo ""
echo "--- Processing toe_train.xml ---"
python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml toe_train.xml \
    --output-xml toe_train_yolo_bbox.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class $TARGET_CLASS

# Generate YOLO bbox XML for test set
echo ""
echo "--- Processing toe_test.xml ---"
python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml toe_test.xml \
    --output-xml toe_test_yolo_bbox.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class $TARGET_CLASS

# -------------------------------------------------------------------
# Step 3: Train dlib shape predictor
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "STEP 3: Training dlib shape predictor (toe)"
echo "============================================================"

python shape_trainer.py \
    -d toe_train_yolo_bbox.xml \
    -t toe_test_yolo_bbox.xml \
    -o toe_predictor_yolo_bbox \
    -th 8 \
    -dp 4 \
    -c 15 \
    -nu 0.1 \
    -os 20 \
    -s 20 \
    -f 500 \
    -n 500

echo ""
echo "============================================================"
echo "Training complete at $(date)"
echo "============================================================"
echo "Output: toe_predictor_yolo_bbox.dat"
