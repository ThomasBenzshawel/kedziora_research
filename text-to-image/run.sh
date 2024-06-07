#!/bin/bash

#SBATCH --job-name="SD Test"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=susslandt@msoe.edu
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=2

## SCRIPT START

##pip install -r requirements.txt
nvidia-smi
##pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

##pip install git+https://github.com/huggingface/diffusers
##accelerate launch train_dreambooth.py --instance_data_dir="/data/csc4801/KedzioraLab/TrainingData/Images/airplane/airplane/" --instance_prompt="airplane" --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --output_dir="sd-airplane-model" --resolution=256

accelerate launch test.py

## SCRIPT END