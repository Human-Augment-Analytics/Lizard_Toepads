#!/bin/bash
#SBATCH --job-name=preproc_obb_al
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/preproc_obb_aligned_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@gatech.edu

# OBB preprocessing: axis-aligned conversion with 30% padding (original approach).
# No crop+rotate. OBB corners -> min/max -> axis-aligned rect -> 30% padding.
# Uses H5_obb_noflip model (6-class, bottom-only).

set -e

PROJECT_ROOT=/home/hice1/$USER/scratch/Lizard_Toepads
cd $PROJECT_ROOT/ml-morph

module load anaconda3
source activate lizard

echo "Starting OBB axis-aligned preprocessing on $(hostname)"
echo "Date: $(date)"

IMAGE_DIR=/storage/ice-shared/cs8903onl/miami_fall_24_jpgs
YOLO_MODEL=$PROJECT_ROOT/runs/obb/H5_obb_noflip/weights/best.pt
PADDING=0.3
FALLBACK_PADDING=0.5
CONF=0.25

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
echo "TOE: Generating YOLO-OBB axis-aligned bounding boxes (bot_toe only)"
echo "------------------------------------------------------------"

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml toe_train.xml \
    --output-xml toe_train_yolo_obb_aligned.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class toe \
    --no-rotation

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml toe_test.xml \
    --output-xml toe_test_yolo_obb_aligned.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class toe \
    --no-rotation

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
echo "FINGER: Generating YOLO-OBB axis-aligned bounding boxes (bot_finger only)"
echo "------------------------------------------------------------"

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml finger_train.xml \
    --output-xml finger_train_yolo_obb_aligned.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class finger \
    --no-rotation

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml finger_test.xml \
    --output-xml finger_test_yolo_obb_aligned.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class finger \
    --no-rotation

echo ""
echo "============================================================"
echo "OBB axis-aligned preprocessing complete at $(date)"
echo "============================================================"
echo "Generated:"
echo "  - toe_train_yolo_obb_aligned.xml"
echo "  - toe_test_yolo_obb_aligned.xml"
echo "  - finger_train_yolo_obb_aligned.xml"
echo "  - finger_test_yolo_obb_aligned.xml"
