#!/bin/bash

#SBATCH --job-name=data_processing
#SBATCH --output=model-%j.out
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=15
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source ~/anaconda3/bin/activate pointmlp

# data!
python pointmlp.py

# deactivate env
conda deactivate
