#!/bin/bash
#SBATCH --job-name=download_objaverse
#SBATCH --output=logs/download_%A_%a.out
#SBATCH --error=logs/download_%A_%a.err
#SBATCH --array=1-4                # 4 parallel jobs
#SBATCH --ntasks=1                   # One task per array job
#SBATCH --cpus-per-task=36           # CPUs per task
#SBATCH --mem=200G                   # Memory per task
#SBATCH --partition=teaching         # GPU partition
#SBATCH --account=undergrad_research

# Configuration - modify these as needed
TOTAL_OBJECTS=5000000
NUM_JOBS=10

# Data directories
DATA_DIR="/data/ur/kedziora/layer_x_layer/objaverse_xl"
TEMP_BASE_DIR="/data/ur/kedziora/layer_x_layer/objaverse_temp"

# Calculate objects per job and start offset
OBJECTS_PER_JOB=$((TOTAL_OBJECTS / NUM_JOBS))
START_OFFSET=$(((SLURM_ARRAY_TASK_ID - 1) * OBJECTS_PER_JOB))

# Adjust last job to handle remainder
if [ $SLURM_ARRAY_TASK_ID -eq $NUM_JOBS ]; then
    OBJECTS_PER_JOB=$((TOTAL_OBJECTS - START_OFFSET))
fi

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Processing objects $START_OFFSET to $((START_OFFSET + OBJECTS_PER_JOB - 1))"
echo "Total objects for this job: $OBJECTS_PER_JOB"
echo "Download directory: $DATA_DIR"
echo "Temp directory: $TEMP_BASE_DIR/job_$SLURM_ARRAY_TASK_ID"

# Set up conda environment
# Initialize conda (required for batch jobs)
source ~/.bashrc

# Initialize conda properly
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/objaverse_download

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python
python --version

# Create log directory
mkdir -p logs

# Create job-specific output directory
mkdir -p output/job_$SLURM_ARRAY_TASK_ID

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run the script with job-specific parameters
python3 batch_obja_download.py \
    --xl \
    --num-objects $OBJECTS_PER_JOB \
    --start-offset $START_OFFSET \
    --job-id $SLURM_ARRAY_TASK_ID \
    --download-dir $DATA_DIR \
    --temp-dir $TEMP_BASE_DIR/job_$SLURM_ARRAY_TASK_ID \
    --batch-size 1000 \
    --save-repos

echo "Job $SLURM_ARRAY_TASK_ID completed with exit code $?"