#!/bin/bash
#SBATCH --job-name=render_objaverse
#SBATCH --output=logs/render_%A_%a.out
#SBATCH --error=logs/render_%A_%a.err
#SBATCH --array=0-2             
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=6        # Increased from 4 to 6 for multiprocessing
#SBATCH --mem=128G                
#SBATCH --gres=gpu:1             
#SBATCH --partition=teaching     
#SBATCH --account=undergrad_research

# Set up conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/camera

echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python
python --version


# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export PYOPENGL_PLATFORM=egl

# IMPORTANT: Limit OpenMP threads to prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Input paths
JSON_PATH="/home/ad.msoe.edu/benzshawelt/.objaverse/hf-objaverse-v1/object-paths.json"
SCAN_DIR_2="/home/ad.msoe.edu/benzshawelt/.objaverse"
SCAN_DIR="/data/ur/kedziora/layer_x_layer/objaverse_xl"
OUTPUT_DIR="/data/ur/kedziora/layer_x_layer/objaverse_images"
CHECK_DIR="/data/ur/kedziora/layer_x_layer/objaverse_images_check"

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECK_DIR"

# Job parameters
TOTAL_CHUNKS=$SLURM_ARRAY_TASK_COUNT
CHUNK_ID=$SLURM_ARRAY_TASK_ID

# Log file for this worker
LOG_FILE="logs/worker_${CHUNK_ID}.log"

echo "Starting worker $CHUNK_ID of $TOTAL_CHUNKS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 128G"
echo "JSON Path: $JSON_PATH"
echo "Scan Directory: $SCAN_DIR"

# Run the worker script
python3 isolated_render_worker.py \
    --json_path "$JSON_PATH" \
    --scan_dir "$SCAN_DIR" \
    --scan_dir_2 "$SCAN_DIR_2" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_id $CHUNK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --use_folder_name_as_uid \
    --check_dir "$CHECK_DIR" \
    --log_file "$LOG_FILE"

EXIT_CODE=$?

echo "Worker $CHUNK_ID completed with exit code: $EXIT_CODE"
exit $EXIT_CODE