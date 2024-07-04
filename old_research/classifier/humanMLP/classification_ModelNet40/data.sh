#!/bin/bash

#SBATCH --job-name=data_processing
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=15
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source ~/anaconda3/bin/activate pointmlp

# data!
python data.py

# deactivate env
conda deactivate
