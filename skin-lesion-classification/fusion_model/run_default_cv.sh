#!/bin/bash
# 5-fold cross-validation of the fusion model with the default configuration.
# Produced report Table 5 (mean minimum age-group AUROC 0.856 +/- 0.066).
#
# Submit from the repo root (projectB_final/):  sbatch fusion_model/run_default_cv.sh
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-2:00:00
#SBATCH -o fusion_cv_%j.out
#SBATCH -e fusion_cv_%j.err

echo "Running on $(hostname)"
echo "Start time: $(date)"

# Environment modules from the Tufts HPC course setup; replace with your own
# Python environment (see requirements.txt) if running elsewhere.
module load class/default
module load cs137/2026spring

# The Python scripts read and write files in the current directory,
# so run them from data/ and call them by relative path.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR/data" || exit 1

# Sampler power is passed explicitly: this run used 0.15.
python ../fusion_model/cross_validate.py \
  --head-lr 1e-3 \
  --head-weight-decay 1e-4 \
  --dropout 0.3 \
  --batch-size 16 \
  --sampler-power 0.15 \
  --run-name default_run

echo "End time: $(date)"
