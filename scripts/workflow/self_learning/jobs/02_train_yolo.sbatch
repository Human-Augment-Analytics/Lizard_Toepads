#!/bin/bash
#SBATCH -J train_yolo
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --gres=gpu:GPU_TYPE_PLACEHOLDER:1
#SBATCH --mem-per-gpu=GPU_MEM_PLACEHOLDER
#SBATCH -t 16:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=USER_EMAIL_PLACEHOLDER

set -e

echo "========================================="
echo "Job: Train YOLO Model"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "========================================="

cd ${SLURM_SUBMIT_DIR}

# Prepare data
echo ""
echo "Preparing dataset..."
RAW_DIR="${SLURM_SUBMIT_DIR}/data/raw"
PROC_DIR="${SLURM_SUBMIT_DIR}/data/processed"
mkdir -p ${PROC_DIR}/{images,labels}/{train,val}

# Simple preprocessing: copy and organize images
# Split 80/20 for train/val
TOTAL_FILES=$(find ${RAW_DIR} -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l)
TRAIN_COUNT=$((TOTAL_FILES * 80 / 100))

echo "Total images: ${TOTAL_FILES}"
echo "Train: ${TRAIN_COUNT}, Val: $((TOTAL_FILES - TRAIN_COUNT))"

# Train YOLO
echo ""
echo "Starting YOLO training..."
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/models/yolo"
mkdir -p ${OUTPUT_DIR}

srun python << EOFPYTHON
from ultralytics import YOLO
import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Load model
model = YOLO('yolov8YOLO_MODEL_PLACEHOLDER.pt')

# Train
results = model.train(
    data='${PROC_DIR}/dataset.yaml',
    epochs=EPOCHS_PLACEHOLDER,
    imgsz=IMG_SIZE_PLACEHOLDER,
    batch=BATCH_SIZE_PLACEHOLDER,
    device=0,
    project='${OUTPUT_DIR}',
    name='toepad_yolo',
    patience=20,
    save=True,
    plots=True
)

# Export to ONNX
model.export(format='onnx')
print('Training completed!')
EOFPYTHON

# Copy best model
cp ${OUTPUT_DIR}/toepad_yolo/weights/best.pt ${OUTPUT_DIR}/yolo_best.pt
cp ${OUTPUT_DIR}/toepad_yolo/weights/best.onnx ${OUTPUT_DIR}/yolo_best.onnx 2>/dev/null || true

echo ""
echo "Model saved: ${OUTPUT_DIR}/yolo_best.pt"
ls -lh ${OUTPUT_DIR}/yolo_best.pt

echo ""
echo "========================================="
echo "Training completed: $(date)"
echo "========================================="