#!/bin/bash

#SBATCH --job-name=llava_distributed
#SBATCH --output=logs/llava_dist_%A_%a.out
#SBATCH --error=logs/llava_dist_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxh100
#SBATCH --array=0-6%7  # Run 7 workers in parallel (adjust as needed)
#SBATCH --account=undergrad_research


# ============================================
# Configuration
# ============================================
TOTAL_WORKERS=7
FILE_LIST="../voxelize_parallel/logs/glb_file_list.json"
IMAGES_BASE_DIR="/data/ur/kedziora/layer_x_layer/objaverse_images"
OUTPUT_DIR="/data/ur/kedziora/layer_x_layer/objaverse_descriptions"
PYTHON_SCRIPT="llava_parallel_worker.py"
SAVE_INTERVAL=100

# Create necessary directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"
mkdir -p checkpoints

# Set worker ID from SLURM array task ID
WORKER_ID=$SLURM_ARRAY_TASK_ID

# ============================================
# Job Information
# ============================================
echo "=========================================="
echo "SLURM Job Info:"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Worker ID: $WORKER_ID"
echo "Total Workers: $TOTAL_WORKERS"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo "Configuration:"
echo "File List: $FILE_LIST"
echo "Images Dir: $IMAGES_BASE_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Python Script: $PYTHON_SCRIPT"
echo "=========================================="

# ============================================
# Environment Setup
# ============================================
# Initialize conda
source ~/.bashrc
source /usr/local/miniforge3/etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"

# Activate conda environment  
conda activate llava

echo "Worker $WORKER_ID: Activated conda environment 'llava'"
echo "Python path: $(which python3)"
echo "Python version: $(python3 --version)"

# Set CUDA device based on SLURM allocation
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# ============================================
# Checkpoint Management
# ============================================
CHECKPOINT_FILE="checkpoints/worker_${WORKER_ID}_checkpoint.txt"

# Function to handle worker completion
cleanup_worker() {
    echo "Worker $WORKER_ID: Cleaning up..."
    echo "$(date): Worker $WORKER_ID completed" >> "$CHECKPOINT_FILE"
    
    # Merge worker metadata cache into main cache
    python3 << EOF
import json
import os
import glob
import fcntl
import time

def merge_worker_caches():
    """Merge all worker caches into main cache"""
    main_cache_file = "./objaverse_metadata_cache.json"
    worker_cache_pattern = "./objaverse_metadata_cache_worker_*.json"
    
    # Load main cache
    main_cache = {}
    main_search_times = {}
    
    if os.path.exists(main_cache_file):
        with open(main_cache_file, 'r') as f:
            try:
                data = json.load(f)
                main_cache = data.get('cache', {})
                main_search_times = data.get('search_times', {})
            except:
                pass
    
    # Merge worker caches
    merged_count = 0
    for worker_cache_file in glob.glob(worker_cache_pattern):
        try:
            with open(worker_cache_file, 'r') as f:
                data = json.load(f)
                worker_cache = data.get('cache', {})
                worker_search_times = data.get('search_times', {})
                
                # Merge into main cache
                for uid, name in worker_cache.items():
                    if uid not in main_cache:
                        main_cache[uid] = name
                        main_search_times[uid] = worker_search_times.get(uid, 0)
                        merged_count += 1
        except Exception as e:
            print(f"Error processing {worker_cache_file}: {e}")
    
    # Save merged cache
    if merged_count > 0:
        try:
            # Use file locking for thread safety
            with open(main_cache_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                merged_data = {
                    'cache': main_cache,
                    'search_times': main_search_times,
                    'last_updated': time.time(),
                    'merged_workers': merged_count
                }
                json.dump(merged_data, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            print(f"Worker $WORKER_ID: Merged {merged_count} new metadata entries")
        except Exception as e:
            print(f"Worker $WORKER_ID: Error merging caches: {e}")

merge_worker_caches()
EOF
}

# Set trap to run cleanup on exit
trap cleanup_worker EXIT

# Check if this worker was already completed
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Worker $WORKER_ID: Found checkpoint, checking if already completed..."
    if grep -q "completed successfully" "$CHECKPOINT_FILE"; then
        echo "Worker $WORKER_ID: Already completed, exiting."
        exit 0
    fi
fi

# Record start time
echo "$(date): Worker $WORKER_ID starting" > "$CHECKPOINT_FILE"

# ============================================
# Pre-flight Checks
# ============================================
echo "Worker $WORKER_ID: Running pre-flight checks..."

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

# Check if file list exists
if [ ! -f "$FILE_LIST" ]; then
    echo "ERROR: File list $FILE_LIST not found!"
    echo "Please generate the file list first using the voxelization script's file list generator."
    exit 1
fi

# Check if images directory exists
if [ ! -d "$IMAGES_BASE_DIR" ]; then
    echo "ERROR: Images directory $IMAGES_BASE_DIR not found!"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi not available, GPU access may be limited"
fi

echo "Worker $WORKER_ID: GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits

# Display file list info
echo "Worker $WORKER_ID: File list information:"
python3 << EOF
import json
with open("$FILE_LIST", 'r') as f:
    data = json.load(f)
    print(f"  Total files in list: {data['total_files']}")
    print(f"  Generated: {data['scan_timestamp']}")
    print(f"  Directories: {data['scan_directories']}")
EOF

# ============================================
# Resource Monitoring
# ============================================
# Monitor system resources in background
monitor_resources() {
    while true; do
        echo "$(date): Worker $WORKER_ID GPU Memory:" >> logs/worker_${WORKER_ID}_resources.log
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >> logs/worker_${WORKER_ID}_resources.log
        echo "$(date): Worker $WORKER_ID CPU/RAM:" >> logs/worker_${WORKER_ID}_resources.log
        ps -o pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -10 >> logs/worker_${WORKER_ID}_resources.log
        echo "---" >> logs/worker_${WORKER_ID}_resources.log
        sleep 300  # Monitor every 5 minutes
    done
}

# Start resource monitoring in background
monitor_resources &
MONITOR_PID=$!

# Function to kill background processes on exit
cleanup_background() {
    kill $MONITOR_PID 2>/dev/null || true
}
trap cleanup_background EXIT

# ============================================
# Run LLaVA Worker
# ============================================
echo "Worker $WORKER_ID: Starting LLaVA processing..."
echo "Worker $WORKER_ID: Processing items from file list..."

# Run the Python worker with updated parameters
python3 "$PYTHON_SCRIPT" \
    --worker_id "$WORKER_ID" \
    --total_workers "$TOTAL_WORKERS" \
    --file_list "$FILE_LIST" \
    --images_base_dir "$IMAGES_BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --save_interval "$SAVE_INTERVAL"

PYTHON_EXIT_CODE=$?

# ============================================
# Completion
# ============================================
# Record completion status
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "$(date): Worker $WORKER_ID completed successfully" >> "$CHECKPOINT_FILE"
    echo "Worker $WORKER_ID: Processing completed successfully!"
    
    # Display statistics
    echo "Worker $WORKER_ID: Final statistics:"
    NUM_OUTPUTS=$(find "$OUTPUT_DIR" -name "*.txt" -type f | wc -l)
    echo "  Total descriptions created: $NUM_OUTPUTS"
else
    echo "$(date): Worker $WORKER_ID failed with exit code $PYTHON_EXIT_CODE" >> "$CHECKPOINT_FILE"
    echo "Worker $WORKER_ID: Processing failed with exit code $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi

echo "Worker $WORKER_ID: Job finished."
echo "=========================================="