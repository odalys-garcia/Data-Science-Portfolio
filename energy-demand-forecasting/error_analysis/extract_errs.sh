#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o extract_errs_%j.out
#SBATCH -e extract_errs_%j.err

echo "Running on $(hostname)"
echo "Start time: $(date)"

# Load modules
module load class/default
module load cs137/2026spring

# Go to your project directory
cd /cluster/home/ogarci02/project_energy_new



# Optional: check torch / cuda
python -c "import torch; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
python extract_energy_errors.py \
    --checkpoint model_1_v2_ema_accum_best.pth \
    --model-file  model_adjusted_v2.py \
    --manifest-dir /cluster/home/ogarci02/project_energy_new/manifests \
    --years 2023 \
    --seq-len-hist 24 \
    --output-dir  ./extreme_analysis \
    --batch-size  8 \
    --device cuda

echo "End time: $(date)"