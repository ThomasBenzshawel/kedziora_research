#!/bin/bash
# SLURM array job script - launched by submit_voxelize_jobs.sh

# Get parameters passed from wrapper
FILE_LIST=$1
OUTPUT_DIR=$2
VOXEL_RESOLUTION=$3

# Set up conda environment - use . instead of source for POSIX compatibility
. ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/voxelize

# Verify environment
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python
python --version

# Set environment variables for better CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Job parameters
TOTAL_CHUNKS=$SLURM_ARRAY_TASK_COUNT
CHUNK_ID=$SLURM_ARRAY_TASK_ID

# Log file for this worker
LOG_FILE="logs/voxel_worker_${CHUNK_ID}.log"

echo "==================== JOB INFO ===================="
echo "Starting worker $CHUNK_ID of $TOTAL_CHUNKS"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "File List: $FILE_LIST"
echo "Output Directory: $OUTPUT_DIR"
echo "Voxel Resolution: $VOXEL_RESOLUTION"
echo "=================================================="

# Run the worker script with cached file list
python3 voxelize_worker_cached.py \
    --file_list "$FILE_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_id $CHUNK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --check_dir "$OUTPUT_DIR" \
    --log_file "$LOG_FILE" \
    --voxel_resolution $VOXEL_RESOLUTION

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Worker $CHUNK_ID completed successfully"
else
    echo "Worker $CHUNK_ID failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE