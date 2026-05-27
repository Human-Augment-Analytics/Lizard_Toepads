#!/bin/bash
#SBATCH --job-name=H5_obb_6class
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:A40:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/H5_obb_6class_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@gatech.edu

# Train YOLO-OBB on 6-class dataset (bot + ruler + id OBBs).
# Step 1: Create the dataset (merge bot_finger/bot_toe + ruler + id OBBs)
# Step 2: Train YOLO-OBB

set -e

PROJECT_ROOT=/home/hice1/$USER/scratch/Lizard_Toepads
cd $PROJECT_ROOT

module load anaconda3
source activate lizard

echo "Starting YOLO-OBB 6-class training on $(hostname)"
echo "Date: $(date)"

# Step 1: Create dataset
echo ""
echo "============================================================"
echo "STEP 1: Creating 6-class active OBB dataset"
echo "============================================================"

python scripts/preprocessing/create_6class_active_obb_dataset.py

# Step 2: Train YOLO-OBB
echo ""
echo "============================================================"
echo "STEP 2: Training YOLO-OBB (6-class, with ruler and id)"
echo "============================================================"

yolo obb train \
    data=$PROJECT_ROOT/configs/H5_obb_6class.yaml \
    model=yolo11m-obb.pt \
    epochs=200 \
    patience=100 \
    imgsz=1280 \
    batch=16 \
    device=0 \
    project=runs/obb \
    name=H5_obb_6class

echo ""
echo "============================================================"
echo "Training complete at $(date)"
echo "============================================================"
