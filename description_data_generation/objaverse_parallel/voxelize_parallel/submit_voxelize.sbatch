#!/bin/bash
#SBATCH --job-name=voxelize_glb
#SBATCH --output=logs/voxel_%A_%a.out
#SBATCH --error=logs/voxel_%A_%a.err
#SBATCH --array=0-99            # Array indices (adjust based on your needs)
#SBATCH --ntasks=1              # One task per array job
#SBATCH --cpus-per-task=4       # CPUs per task (trimesh can use multiple threads)
#SBATCH --mem=32G               # Memory per task (adjust based on model complexity)
#SBATCH --time=24:00:00         # Maximum time per job
#SBATCH --partition=compute     # CPU partition (adjust for your cluster)
# Remove GPU allocation since voxelization is CPU-based
# Add back if you implement GPU acceleration later:
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

# Set up conda environment
# Initialize conda (required for batch jobs)
source ~/.bashrc
# Initialize conda properly
eval "$(conda shell.bash hook)"

# Activate your conda environment
# Update this path to your environment
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/voxel_env

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python
python --version

# Create log directory
mkdir -p logs

# Set environment variables for better CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Input paths - UPDATE THESE FOR YOUR SETUP
JSON_PATH="/home/benzshawelt/.objaverse/hf-objaverse-v1/object-paths.json"
SCAN_DIR="/home/benzshawelt/.objaverse"
OUTPUT_DIR="../objaverse_voxels"
CHECK_DIR="../objaverse_voxels"  # Use output dir for checking by default

# Job parameters
TOTAL_CHUNKS=$SLURM_ARRAY_TASK_COUNT
CHUNK_ID=$SLURM_ARRAY_TASK_ID

# Create output directory
mkdir -p $OUTPUT_DIR

# Log file for this worker
LOG_FILE="logs/voxel_worker_${CHUNK_ID}.log"

echo "==================== JOB INFO ===================="
echo "Starting worker $CHUNK_ID of $TOTAL_CHUNKS"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "JSON Path: $JSON_PATH"
echo "Scan Directory: $SCAN_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "=================================================="

# Run the worker script
python3 voxel_worker.py \
    --json_path "$JSON_PATH" \
    --scan_dir "$SCAN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_id $CHUNK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --check_dir "$CHECK_DIR" \
    --log_file "$LOG_FILE"

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Worker $CHUNK_ID completed successfully"
else
    echo "Worker $CHUNK_ID failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE