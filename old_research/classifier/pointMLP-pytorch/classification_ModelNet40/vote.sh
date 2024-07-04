#!/bin/bash

#SBATCH --job-name=vote_PointMLP
#SBATCH --output=slurm/vote-%j.out
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=15
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source ~/anaconda3/bin/activate pointmlp

# vote!
python voting.py --model pointMLP --msg 20240203185342-9145 --partition test
python voting.py --model pointMLP --msg 20240203185342-9145 --partition thingiverse

# deactivate env
conda deactivate
