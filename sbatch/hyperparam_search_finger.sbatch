#!/bin/bash
#SBATCH --job-name=hp_finger
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hp_finger_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yloh30@gatech.edu

set -e

cd /home/hice1/yloh30/scratch/Lizard_Toepads/ml-morph

module load anaconda3
source activate lizard

echo "Starting hyperparameter search for FINGER on $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python scripts/training/hyperparameter_search.py \
    --train finger_train_yolo_bbox.xml \
    --test finger_test_yolo_bbox.xml \
    --category finger \
    --threads 8 \
    --quick

echo "Hyperparameter search complete at $(date)"
