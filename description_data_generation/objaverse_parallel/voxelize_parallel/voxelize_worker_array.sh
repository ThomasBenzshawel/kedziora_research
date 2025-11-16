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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Job parameters
TOTAL_CHUNKS=$SLURM_ARRAY_TASK_COUNT
CHUNK_ID=$SLURM_ARRAY_TASK_ID

echo "==================== DEBUG INFO ===================="
echo "About to run Python script"
echo "Script exists: $(ls -la voxelize_worker_cached.py)"
echo "File list exists: $(ls -la $FILE_LIST)"
echo "Output dir exists: $(ls -ld $OUTPUT_DIR)"
echo "===================================================="

export PYTHONUNBUFFERED=1


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
python3 -u  voxelize_worker_cached.py \
    --file_list "$FILE_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_id $CHUNK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --check_dir "$OUTPUT_DIR" \
    --voxel_resolution $VOXEL_RESOLUTION

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Worker $CHUNK_ID completed successfully"
else
    echo "Worker $CHUNK_ID failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE