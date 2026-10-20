#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-1:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o analysis_errs_%j.out
#SBATCH -e analysis_errs_%j.err

echo "Running on $(hostname)"
echo "Start time: $(date)"

# Load modules
module load class/default
module load cs137/2026spring

# Go to your project directory
cd /cluster/home/ogarci02/project_energy_new



# Optional: check torch / cuda
python -c "import torch; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
python analyze_extreme_errors.py \
    --output-dir ./extreme_analysis \
    --extreme-pct 5 \
    --min-channels-hit 2

echo "End time: $(date)"