#!/bin/bash

#SBATCH --job-name=stl_to_npy
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schmockerc@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=7-0:0
#SBATCH --account=practicum

# set up the environment
source /usr/local/anaconda3/bin/activate image_conversion

# data!
python3 image_conversion.py

# deactivate env
conda deactivate
