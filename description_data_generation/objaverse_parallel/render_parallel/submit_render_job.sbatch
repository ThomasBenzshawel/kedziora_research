#!/bin/bash
#SBATCH --job-name=render_objaverse
#SBATCH --output=logs/render_%A_%a.out
#SBATCH --error=logs/render_%A_%a.err
#SBATCH --array=0-12             # Array indices (0 to N-1 where N is number of GPUs you want)
#SBATCH --ntasks=1               # One task per array job
#SBATCH --cpus-per-task=4        # CPUs per task
#SBATCH --mem=128G                # Memory per task
#SBATCH --gres=gpu:1             # One GPU per task
#SBATCH --partition=teaching     # GPU partition (adjust for your cluster)
#SBATCH --account=undergrad_research

# Set up conda environment
# Initialize conda (required for batch jobs)
source ~/.bashrc

# Initialize conda properly
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/camera

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python
python --version

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export PYOPENGL_PLATFORM=egl

# Input paths
JSON_PATH="/home/ad.msoe.edu/benzshawelt/.objaverse/hf-objaverse-v1/object-paths.json"
SCAN_DIR="/home/ad.msoe.edu/benzshawelt/.objaverse"
SCAN_DIR_2="/home/ad.msoe.edu/benzshawelt/objaverse_temp"
OUTPUT_DIR="../objaverse_images"
CHECK_DIR="../check_objaverse_images"

# Job parameters
TOTAL_CHUNKS=$SLURM_ARRAY_TASK_COUNT
CHUNK_ID=$SLURM_ARRAY_TASK_ID

# Create output directory
mkdir -p $OUTPUT_DIR

# Log file for this worker
LOG_FILE="logs/worker_${CHUNK_ID}.log"

echo "Starting worker $CHUNK_ID of $TOTAL_CHUNKS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "JSON Path: $JSON_PATH"
echo "Scan Directory: $SCAN_DIR"

# Run the worker script
python3 render_worker.py \
    --json_path "$JSON_PATH" \
    --scan_dir "$SCAN_DIR" \
    --scan_dir_2 "$SCAN_DIR_2" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_id $CHUNK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --use_folder_name_as_uid \
    --check_dir "$CHECK_DIR" \
    --log_file "$LOG_FILE"

echo "Worker $CHUNK_ID completed"