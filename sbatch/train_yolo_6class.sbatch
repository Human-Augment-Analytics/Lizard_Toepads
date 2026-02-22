#!/bin/bash
#SBATCH --job-name=yolo_train_obb_6class
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:V100:1
#SBATCH --constraint=V100-32GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/yolo_train_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@gatech.edu

# Exit on error
set -e

PROJECT_ROOT=/home/hice1/$USER/scratch/Lizard_Toepads

# Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Load necessary modules
module load anaconda3

# Activate your environment
source activate lizard

echo "Starting YOLO-OBB 6-class training on $(hostname)"
echo "Date: $(date)"
echo "Environment: lizard"

# Run YOLO training
# Using yolo11m-obb.pt (Medium) with 6-class merged dataset
yolo obb train \
    data=$PROJECT_ROOT/configs/H1_obb_6class.yaml \
    model=yolo11m-obb.pt \
    epochs=200 \
    patience=100 \
    imgsz=1280 \
    batch=16 \
    device=0 \
    project=runs/obb \
    name=H6_obb_6class

echo "Training finished at $(date)"
