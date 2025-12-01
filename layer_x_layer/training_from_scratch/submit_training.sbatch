#!/bin/bash
#SBATCH --job-name=layerx_training
#SBATCH --output=logs/training_%A_%a.out
#SBATCH --error=logs/training_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxh100
#SBATCH --account=undergrad_research

# Create log directory
mkdir -p logs

# Initialize conda using miniforge3
source /usr/local/miniforge3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/layerxlayer

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python
python --version

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "Starting training on node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the script
python3 patch_causal_train_v4.py --batch_size=20 --skip_cache_scan