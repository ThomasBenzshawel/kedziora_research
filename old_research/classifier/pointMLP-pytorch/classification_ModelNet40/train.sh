#!/bin/bash

#SBATCH --job-name=train_PointMLP
#SBATCH --output=slurm/train-%j.out
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=15
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source ~/anaconda3/bin/activate pointmlp

# train pointMLP
python main.py --model pointMLP --num_points 2048 --seed 42 --msg 422

# train pointMLP-elite
python main.py --model pointMLPElite --num_points 2048 --seed 42 --msg 422

# deactivate env
conda deactivate
