#!/bin/bash
#SBATCH --job-name=preproc_bl
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/preproc_baseline_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yloh30@gatech.edu

# Baseline preprocessing: TPS -> XML -> YOLO baseline bbox XML
# Generates train/test XMLs for both toe and finger using the
# non-OBB YOLO detect model (H5_detect_6class).
# No flip strategy needed - the 6-class model detects all views directly.

set -e

PROJECT_ROOT=/home/hice1/yloh30/scratch/Lizard_Toepads
cd $PROJECT_ROOT/ml-morph

module load anaconda3
source activate lizard

echo "Starting BASELINE preprocessing on $(hostname)"
echo "Date: $(date)"

IMAGE_DIR=/storage/ice-shared/cs8903onl/miami_fall_24_jpgs
YOLO_MODEL=$PROJECT_ROOT/runs/detect/H5_detect_6class/weights/best.pt
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
echo "TOE: Generating YOLO baseline bounding boxes"
echo "------------------------------------------------------------"

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml toe_train.xml \
    --output-xml toe_train_yolo_baseline.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class toe

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml toe_test.xml \
    --output-xml toe_test_yolo_baseline.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class toe

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
echo "FINGER: Generating YOLO baseline bounding boxes"
echo "------------------------------------------------------------"

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml finger_train.xml \
    --output-xml finger_train_yolo_baseline.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class finger

python scripts/preprocessing/generate_yolo_bbox_xml.py \
    --input-xml finger_test.xml \
    --output-xml finger_test_yolo_baseline.xml \
    --yolo-model "$YOLO_MODEL" \
    --padding-ratio $PADDING \
    --fallback-padding-ratio $FALLBACK_PADDING \
    --conf-threshold $CONF \
    --target-class finger

echo ""
echo "============================================================"
echo "Baseline preprocessing complete at $(date)"
echo "============================================================"
echo "Generated:"
echo "  - toe_train_yolo_baseline.xml"
echo "  - toe_test_yolo_baseline.xml"
echo "  - finger_train_yolo_baseline.xml"
echo "  - finger_test_yolo_baseline.xml"
