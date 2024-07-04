#!/bin/bash

#SBATCH --job-name=train_PointMLP
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=15
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source ~/anaconda3/bin/activate pointmlp

# train pointMLP
python main.py --model pointMLP

# train pointMLP-elite
python main.py --model pointMLPElite

# deactivate env
conda deactivate
