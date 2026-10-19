#!/bin/bash
# Problem 1, step 1: single-split RF exploration and per-class threshold
# analysis (train_rf.py), with rare-class augmentation on the training split.
#
# Submit from the repo root (projectB_final/):  sbatch RF_model/train_rf.sh
#
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00
#SBATCH --mem=16G
#SBATCH -p batch
#SBATCH -o train_rf_%j.out
#SBATCH -e train_rf_%j.err

# Environment modules from the Tufts HPC course setup; replace with your own
# Python environment (see requirements.txt) if running elsewhere.
module load class/default
module load cs137/2026spring

# The Python scripts read and write files in the current directory,
# so run them from data/ and call them by relative path.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR/data" || exit 1

python ../RF_model/train_rf.py \
  --x x_train.csv \
  --y y_train.csv \
  --img-features x_train_img_features.csv \
  --train-img-npz x_train_img.npz \
  --augment-rare
