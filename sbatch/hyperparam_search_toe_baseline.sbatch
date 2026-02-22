#!/bin/bash
#SBATCH --job-name=hp_toe_bl
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=16:00:00
#SBATCH --output=logs/hp_toe_baseline_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@gatech.edu

# 144-config hyperparameter search for TOE with YOLO baseline bboxes.
# Requires preprocess_baseline.sbatch to have completed first.

set -e

PROJECT_ROOT=/home/hice1/$USER/scratch/Lizard_Toepads
cd $PROJECT_ROOT/ml-morph

module load anaconda3
source activate lizard

echo "Starting BASELINE hyperparameter search for TOE on $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Verify preprocessing outputs exist
if [ ! -f toe_train_yolo_baseline.xml ] || [ ! -f toe_test_yolo_baseline.xml ]; then
    echo "ERROR: Baseline XMLs not found. Run preprocess_baseline.sbatch first."
    exit 1
fi

python scripts/training/hyperparameter_search.py \
    --train toe_train_yolo_baseline.xml \
    --test toe_test_yolo_baseline.xml \
    --category toe \
    --output-dir hyperparam_results_toe_baseline \
    --threads 8 \
    --baseline

echo ""
echo "============================================================"
echo "Hyperparameter search complete at $(date)"
echo "============================================================"
