#!/bin/bash
#SBATCH --job-name=hp_fing_obb
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=16:00:00
#SBATCH --output=logs/hp_finger_obb_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@gatech.edu

# 144-config hyperparameter search for FINGER with YOLO-OBB bboxes.
# Requires preprocess_obb.sbatch to have completed first.

set -e

PROJECT_ROOT=/home/hice1/$USER/scratch/Lizard_Toepads
cd $PROJECT_ROOT/ml-morph

module load anaconda3
source activate lizard

echo "Starting OBB hyperparameter search for FINGER on $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Verify preprocessing outputs exist
if [ ! -f finger_train_yolo_obb.xml ] || [ ! -f finger_test_yolo_obb.xml ]; then
    echo "ERROR: OBB XMLs not found. Run preprocess_obb.sbatch first."
    exit 1
fi

python scripts/training/hyperparameter_search.py \
    --train finger_train_yolo_obb.xml \
    --test finger_test_yolo_obb.xml \
    --category finger \
    --output-dir hyperparam_results_finger_obb \
    --threads 8 \
    --baseline

echo ""
echo "============================================================"
echo "Hyperparameter search complete at $(date)"
echo "============================================================"
