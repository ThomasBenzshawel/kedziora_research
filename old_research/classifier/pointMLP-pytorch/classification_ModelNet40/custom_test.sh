#!/bin/bash

#SBATCH --job-name=test_PointMLP
#SBATCH --output=slurm/custom_test-%j.out
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=15
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source ~/anaconda3/bin/activate pointmlp

# vote!
python custom_test.py --model pointMLPElite --msg 422-42 --partition generated --num_points 2048 --wipe
python custom_test.py --model pointMLPElite --msg 422-42 --partition ModelNet12_test --num_points 2048

#python custom_test.py --model pointMLP --msg 422-42 --partition generated --num_points 2048

# deactivate env
conda deactivate
