#!/bin/bash
#SBATCH --job-name=yolo_train_obb_2class
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:A40:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/yolo_train_2class_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yloh30@gatech.edu

PROJECT_ROOT=/home/hice1/yloh30/scratch/Lizard_Toepads
cd $PROJECT_ROOT

# Load environment
module load anaconda3
source activate lizard

# Training Command:
yolo obb train \
    data=$PROJECT_ROOT/configs/H1_obb_2class.yaml \
    model=yolo11m-obb.pt \
    epochs=200 \
    patience=100 \
    imgsz=1280 \
    batch=16 \
    device=0 \
    project=runs/obb \
    name=H1_obb_2class
