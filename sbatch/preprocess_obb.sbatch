#!/bin/bash
#SBATCH --job-name=preproc_obb
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/preproc_obb_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@gatech.edu

# OBB preprocessing: TPS -> XML -> YOLO-OBB bbox XML with crop+rotate
# Uses H5_obb_noflip model (6-class, no flip - bot_finger, bot_toe + ruler).
# OBB rotated boxes are cropped and rotated upright for tight bounding boxes.

set -e

PROJECT_ROOT=/home/hice1/$USER/scratch/Lizard_Toepads
cd $PROJECT_ROOT/ml-morph

module load anaconda3
source activate lizard

echo "Starting OBB preprocessing on $(hostname)"
echo "Date: $(date)"

IMAGE_DIR=/storage/ice-shared/cs8903onl/miami_fall_24_jpgs
YOLO_MODEL=$PROJECT_ROOT/runs/obb/H5_obb_noflip/weights/best.pt
PADDING=0.1
FALLBACK_PADDING=0.5
CONF=0.25
CROP_SCALE=2.0

# ===================================================================
# TOE
# ===================================================================
echo ""
echo "============================================================"
echo "TOE: Generating XMLs from consolidated_toe.tps"
echo "============================================================"

python scripts/preprocessing/tps_to_xml.py \
    -t consolidated_toe.tps \
    -i "$IMAGE_DIR" \
    --train-ratio 0.8 \
    --seed 42 \
    --output-train toe_train.xml \
    --output-test toe_test.xml

echo "Train images: $(grep -c '<image' toe_train.xml)"
echo "Test images:  $(grep -c '<image' toe_test.xml)"

echo ""
echo "TOE: Generating YOLO-OBB bounding boxes with crop+rotate (bot_toe only)"
echo "------------------------------------------------------------"

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml toe_train.xml \
    --output-xml toe_train_yolo_obb.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class toe \
    --crops-dir crops_obb_toe \
    --crop-scale $CROP_SCALE

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml toe_test.xml \
    --output-xml toe_test_yolo_obb.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class toe \
    --crops-dir crops_obb_toe \
    --crop-scale $CROP_SCALE

# ===================================================================
# FINGER
# ===================================================================
echo ""
echo "============================================================"
echo "FINGER: Generating XMLs from consolidated_finger.tps"
echo "============================================================"

python scripts/preprocessing/tps_to_xml.py \
    -t consolidated_finger.tps \
    -i "$IMAGE_DIR" \
    --train-ratio 0.8 \
    --seed 42 \
    --output-train finger_train.xml \
    --output-test finger_test.xml

echo "Train images: $(grep -c '<image' finger_train.xml)"
echo "Test images:  $(grep -c '<image' finger_test.xml)"

echo ""
echo "FINGER: Generating YOLO-OBB bounding boxes with crop+rotate (bot_finger only)"
echo "------------------------------------------------------------"

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml finger_train.xml \
    --output-xml finger_train_yolo_obb.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class finger \
    --crops-dir crops_obb_finger \
    --crop-scale $CROP_SCALE

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml finger_test.xml \
    --output-xml finger_test_yolo_obb.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class finger \
    --crops-dir crops_obb_finger \
    --crop-scale $CROP_SCALE

echo ""
echo "============================================================"
echo "OBB preprocessing complete at $(date)"
echo "============================================================"
echo "Generated:"
echo "  - toe_train_yolo_obb.xml"
echo "  - toe_test_yolo_obb.xml"
echo "  - finger_train_yolo_obb.xml"
echo "  - finger_test_yolo_obb.xml"
echo "  - crops_obb_toe/ (rotated crop images)"
echo "  - crops_obb_finger/ (rotated crop images)"
