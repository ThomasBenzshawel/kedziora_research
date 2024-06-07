#!/bin/bash
#SBATCH --job-name=point2mesh
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=32
#SBATCH --time=1-0:0

# set up the environment
source ~/anaconda3/bin/activate point2mesh

# Pulled from the scripts/examples/bull.sh
python main.py --input-pc ./data/bull.ply --initial-mesh ./data/bull_initmesh.obj --save-path ./checkpoints/bull --pools 0.1 0.0 0.0 0.0 --iterations 6000

# deactivate env
conda deactivate
