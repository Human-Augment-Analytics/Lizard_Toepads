#!/bin/bash
#SBATCH --job-name=hp_toe
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hp_toe_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yloh30@gatech.edu

set -e

PROJECT_ROOT=/home/hice1/yloh30/scratch/Lizard_Toepads
cd $PROJECT_ROOT/ml-morph

module load anaconda3
source activate lizard

echo "Starting hyperparameter search for TOE on $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python scripts/training/hyperparameter_search.py \
    --train toe_train_yolo_bbox.xml \
    --test toe_test_yolo_bbox.xml \
    --category toe \
    --threads 8 \
    --quick

echo "Hyperparameter search complete at $(date)"
