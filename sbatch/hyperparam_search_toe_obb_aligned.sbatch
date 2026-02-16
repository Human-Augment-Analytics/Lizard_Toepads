#!/bin/bash
#SBATCH --job-name=hp_toe_obb_al
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=16:00:00
#SBATCH --output=logs/hp_toe_obb_aligned_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yloh30@gatech.edu

# 144-config hyperparameter search for TOE with YOLO-OBB axis-aligned bboxes.
# Requires preprocess_obb_aligned.sbatch to have completed first.

set -e

PROJECT_ROOT=/home/hice1/yloh30/scratch/Lizard_Toepads
cd $PROJECT_ROOT/ml-morph

module load anaconda3
source activate lizard

echo "Starting OBB axis-aligned hyperparameter search for TOE on $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Verify preprocessing outputs exist
if [ ! -f toe_train_yolo_obb_aligned.xml ] || [ ! -f toe_test_yolo_obb_aligned.xml ]; then
    echo "ERROR: OBB aligned XMLs not found. Run preprocess_obb_aligned.sbatch first."
    exit 1
fi

python scripts/training/hyperparameter_search.py \
    --train toe_train_yolo_obb_aligned.xml \
    --test toe_test_yolo_obb_aligned.xml \
    --category toe \
    --output-dir hyperparam_results_toe_obb_aligned \
    --threads 8 \
    --baseline

echo ""
echo "============================================================"
echo "Hyperparameter search complete at $(date)"
echo "============================================================"
