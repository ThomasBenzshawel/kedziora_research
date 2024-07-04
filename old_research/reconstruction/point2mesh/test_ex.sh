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
# source ~/anaconda3/bin/activate point2mesh
source /usr/local/anaconda3/bin/activate /data/csc4801/KedzioraLab/envs/point2mesh

# Pulled from the scripts/examples/bull.sh
python main.py --input-pc ./data/test2.ply --initial-mesh ./data/test2_initmesh.obj --save-path ./checkpoints/test2_pool --iterations 10000 --unoriented --export-interval 1000 --pools 0.1 0.0 0.0 0.0

# deactivate env
conda deactivate
