#!/bin/bash
#SBATCH -J download_data
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH -t 2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=USER_EMAIL_PLACEHOLDER

set -e

echo "========================================="
echo "Job: Download Training Data from Dropbox"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "========================================="

# Change to repository root
cd ${SLURM_SUBMIT_DIR}

# Load Python environment
module load anaconda3

# Check if rclone is available
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    pip install rclone --user
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "rclone version: $(rclone version | head -1)"

# Install PyYAML if needed
pip install pyyaml --quiet

echo ""
echo "========================================="
echo "Downloading Images"
echo "========================================="

# Download images
srun uv python scripts/workflow/download_data.py \
    --config configs/pace_pipeline.yaml \
    --data-type images

echo ""
echo "========================================="
echo "Downloading Labels"
echo "========================================="

# Download labels
srun uv python scripts/workflow/download_data.py \
    --config configs/pace_pipeline.yaml \
    --data-type labels

# Summary
echo ""
echo "========================================="
echo "Download Summary"
echo "========================================="
TOTAL_IMAGES=$(find data -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
TOTAL_LABELS=$(find data -type f -name "*.txt" | wc -l)
TOTAL_SIZE=$(du -sh data | cut -f1)

echo "Total images: ${TOTAL_IMAGES}"
echo "Total labels: ${TOTAL_LABELS}"
echo "Total size: ${TOTAL_SIZE}"
echo "Location: $(pwd)/data"

echo ""
echo "========================================="
echo "Download completed: $(date)"
echo "========================================="