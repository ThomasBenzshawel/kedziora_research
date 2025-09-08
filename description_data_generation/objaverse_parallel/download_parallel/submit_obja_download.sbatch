#!/bin/bash
#SBATCH --job-name=objaverse_download
#SBATCH --output=logs/download.out
#SBATCH --error=logs/download.err
#SBATCH --ntasks=1               # One task per array job
#SBATCH --cpus-per-task=36        # CPUs per task
#SBATCH --mem=200G               # Memory per task
#SBATCH --time=48:00:00          # Maximum time per job
#SBATCH --partition=teaching     # GPU partition
#SBATCH --account=undergrad_research

# Set up conda environment
# Initialize conda (required for batch jobs)
source ~/.bashrc

# Initialize conda properly
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate /home/benzshawelt/.conda/envs/objaverse_download

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python
python --version

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run the script
python3 batch_obja_download.py