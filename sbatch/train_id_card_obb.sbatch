#!/bin/bash
#SBATCH --job-name=id_card_obb
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A40:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/id_card_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/id_cards/pipeline.py "${@:-train}"
