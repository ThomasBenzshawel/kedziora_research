#!/bin/bash

#SBATCH --job-name="Fixed Print Test"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schneideral@msoe.edu
#SBATCH --partition=highmem
#SBATCH --nodes=1

## SCRIPT START

eval "$(conda shell.bash hook)"
conda activate dsp
python3 fixed_thingiverse_check.py

## SCRIPT END