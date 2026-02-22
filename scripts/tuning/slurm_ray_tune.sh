#!/bin/bash
#SBATCH --job-name=ray-tune-yolo
#SBATCH --partition=ice-gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/ray_tune_%j.out
#SBATCH --error=logs/ray_tune_%j.err

# ============================================================
# Ray Tune Multi-Node Hyperparameter Tuning on PACE/ICE
# ============================================================
#
# Usage:
#   sbatch scripts/tuning/slurm_ray_tune.sh
#
# Resources:
#   --nodes=4          : 4 H200 nodes (4 concurrent trials)
#   --gres=gpu:h200:1  : 1 H200 GPU per node
#   --time=4:00:00     : 4 hours
#
# QOS Limits (coc-ice):
#   Max GPU·minutes per job: 960 (= 16 GPU·hours)
#   Max CPU·minutes per job: 30720 (= 512 CPU·hours)
#   => nodes × gpus × hours <= 16
#   => 4 nodes × 1 GPU × 4h = 16 GPU·h (max allowed)
#
# Partition limits (ice-gpu):
#   Max walltime: 16:00:00
#   Available GPUs: a100, h100, h200, v100
#
# ============================================================

set -e

mkdir -p logs

# Project paths (shared filesystem, accessible from all nodes)
PROJECT_DIR=~/Lizard_Toepads
VENV_ACTIVATE="source $PROJECT_DIR/.venv/bin/activate"
cd $PROJECT_DIR
source .venv/bin/activate

# Suppress noisy warnings and ANSI codes
export NO_COLOR=1
export ANSI_COLORS_DISABLED=1
export PY_COLORS=0
export TERM=dumb
export RAY_COLOR_PREFIX=0
export RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

# Put heavy Ray/YOLO artifacts on ICE scratch.
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${TUNE_BASE_DIR:-$HOME/scratch}"
SCRATCH_ROOT="$BASE_DIR/lizard_toepads/ray_tune/$JOB_TAG"
mkdir -p "$SCRATCH_ROOT"/{ray_results,tune}
export RAY_STORAGE_PATH="$SCRATCH_ROOT/ray_results"
export YOLO_PROJECT="$SCRATCH_ROOT/tune"

# Convenient symlink inside project
ln -sfn "$SCRATCH_ROOT" "$PROJECT_DIR/runs_scratch"

# Get node information
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
PORT=6379

echo "============================================================"
echo "Ray Tune Multi-Node Setup"
echo "============================================================"
echo "Head Node: $HEAD_NODE ($HEAD_NODE_IP:$PORT)"
echo "Total Nodes: $SLURM_NNODES"
echo "GPUs per Node: 1"
echo "Python: $(python --version)"
echo "Scratch Root: $SCRATCH_ROOT"
echo "Ray Storage: $RAY_STORAGE_PATH"
echo "YOLO Project: $YOLO_PROJECT"
echo "============================================================"

# ---- Start Ray cluster ----
# We avoid --block because it monitors all subprocesses and kills the entire
# cluster if any one (e.g. ray_client_server) exits unexpectedly.
# Instead, ray start returns immediately (daemons stay alive) and we use
# "sleep infinity" to keep the srun task alive for SLURM.

echo "Starting Ray head node on $HEAD_NODE..."
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    bash -c "$VENV_ACTIVATE && ray start --head --port=$PORT --dashboard-host=0.0.0.0 --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=1 --disable-usage-stats && sleep infinity" &

sleep 10

WORKER_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2)
for NODE in $WORKER_NODES; do
    echo "Starting Ray worker on $NODE..."
    srun --nodes=1 --ntasks=1 -w "$NODE" \
        bash -c "$VENV_ACTIVATE && ray start --address='$HEAD_NODE_IP:$PORT' --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=1 && sleep infinity" &
done

sleep 15

# ---- Run tuning ----
echo "Starting hyperparameter tuning..."
export RAY_ADDRESS="$HEAD_NODE_IP:$PORT"

python scripts/tuning/tune_hyperparams.py \
    --config configs/H6.yaml \
    --num-samples 20 \
    --gpus-per-trial 1.0 \
    --cpus-per-trial 4 \
    --metric 'metrics/mAP50(B)' \
    --mode max

echo "============================================================"
echo "Tuning complete! Check $SCRATCH_ROOT for results."
echo "============================================================"
