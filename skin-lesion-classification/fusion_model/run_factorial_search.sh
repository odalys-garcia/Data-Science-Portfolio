#!/bin/bash
# 8-run fractional-factorial hyperparameter search for the fusion model.
# Writes hyperparam_search_summary.csv.
#
# Design: 2^(4-1) fractional factorial, resolution IV, over four factors
# at two levels each (batch size fixed at 16):
#
#   dropout        0.2   / 0.4
#   head lr        5e-4  / 1e-3
#   weight decay   1e-4  / 5e-4
#   sampler power  0.1   / 0.2
#
# Generator: power = dropout x lr x weight decay (in -/+ coding). Main
# effects are not aliased with any two-factor interactions, so each
# factor's effect can be estimated by averaging over the other three.
#
# Submit from the repo root (projectB_final/):  sbatch fusion_model/run_factorial_search.sh
#
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-2:00
#SBATCH -o hp8_%j.out
#SBATCH -e hp8_%j.err

echo "Start time: $(date)"
echo "Running on host: $(hostname)"

# Environment modules from the Tufts HPC course setup; replace with your own
# Python environment (see requirements.txt) if running elsewhere.
module load class/default
module load cs137/2026spring

# The Python scripts read and write files in the current directory,
# so run them from data/ and call them by relative path.
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR/data" || exit 1

# These runs were executed on CPU (no GPU requested above).
export CUDA_VISIBLE_DEVICES=""

# Columns: dropout  head_lr  weight_decay  batch_size  sampler_power
declare -a configs=(
  "0.2 5e-4 1e-4 16 0.10"
  "0.2 5e-4 5e-4 16 0.20"
  "0.2 1e-3 1e-4 16 0.20"
  "0.2 1e-3 5e-4 16 0.10"
  "0.4 5e-4 1e-4 16 0.20"
  "0.4 5e-4 5e-4 16 0.10"
  "0.4 1e-3 1e-4 16 0.10"
  "0.4 1e-3 5e-4 16 0.20"
)

for cfg in "${configs[@]}"; do
  read -r dropout head_lr head_weight_decay batch_size sampler_power <<< "$cfg"

  run_name="lr${head_lr}_wd${head_weight_decay}_do${dropout}_bs${batch_size}_sp${sampler_power}"
  run_name=${run_name//./p}

  echo "=============================================================="
  echo "Running $run_name"
  echo "  dropout=$dropout head_lr=$head_lr weight_decay=$head_weight_decay"
  echo "  batch_size=$batch_size sampler_power=$sampler_power"
  echo "=============================================================="

  python ../fusion_model/cross_validate.py \
    --dropout "$dropout" \
    --head-lr "$head_lr" \
    --head-weight-decay "$head_weight_decay" \
    --batch-size "$batch_size" \
    --sampler-power "$sampler_power" \
    --run-name "$run_name"

  echo "Finished run: $run_name at $(date)"
done

echo "End time: $(date)"
