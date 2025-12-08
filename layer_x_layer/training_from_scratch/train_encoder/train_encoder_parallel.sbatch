#!/bin/bash
#SBATCH --job-name=encoder_parallel
#SBATCH --output=logs/encoder_parallel_%j.out
#SBATCH --error=logs/encoder_parallel_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=dgxh100
#SBATCH --account=undergrad_research

# train_encoder_parallel.sbatch
# This script trains encoders for ALL resolution pairs IN PARALLEL on different GPUs

# Create logs directory
mkdir -p logs

echo "================================================================"
echo "Parallel Encoder Training (One Encoder Per GPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "================================================================"
echo ""

# Initialize conda using miniforge3
source /usr/local/miniforge3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/layerxlayer

# Verify environment
echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"
which python3
python3 --version
echo ""

export PYTHONUNBUFFERED=1

# Get number of GPUs
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
echo "Available GPUs: $NUM_GPUS"
echo ""

# Base directory
BASE_DATA_DIR="/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels"

# Array of resolution pairs to train
# Format: "low_res:high_res"
RESOLUTION_PAIRS=(
    "16:32"
    "32:64"
)

# Training parameters
EPOCHS=30
BATCH_SIZE=16
LEARNING_RATE=1e-4
ENCODER_MODE="spatial_aware"
BASE_CHANNELS=64

echo "================================================================"
echo "Training Configuration"
echo "================================================================"
echo "Resolution pairs: ${#RESOLUTION_PAIRS[@]}"
for pair in "${RESOLUTION_PAIRS[@]}"; do
    echo "  - $pair"
done
echo "Epochs per pair:  ${EPOCHS}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Learning rate:    ${LEARNING_RATE}"
echo "Encoder mode:     ${ENCODER_MODE}"
echo "Base channels:    ${BASE_CHANNELS}"
echo "GPUs available:   ${NUM_GPUS}"
echo "================================================================"
echo ""

# Function to train a single encoder on a specific GPU
train_encoder() {
    local GPU_ID=$1
    local LOW_RES=$2
    local HIGH_RES=$3
    local LOG_FILE=$4
    
    LOW_RES_PATH="${BASE_DATA_DIR}/${LOW_RES}"
    HIGH_RES_PATH="${BASE_DATA_DIR}/${HIGH_RES}"
    OUTPUT_DIR="./encoder_checkpoints_stage${LOW_RES}_to_stage${HIGH_RES}"
    
    echo "[GPU ${GPU_ID}] Training: ${LOW_RES}³ → ${HIGH_RES}³" >> "${LOG_FILE}" 2>&1
    echo "[GPU ${GPU_ID}] Output: ${OUTPUT_DIR}" >> "${LOG_FILE}" 2>&1
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Run single-GPU training (no torchrun, no distributed)
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_low_res_encoder.py \
        --low_res_path "${LOW_RES_PATH}" \
        --high_res_path "${HIGH_RES_PATH}" \
        --low_res_gran ${LOW_RES} \
        --high_res_gran ${HIGH_RES} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --val_split 0.1 \
        --encoder_mode ${ENCODER_MODE} \
        --base_channels ${BASE_CHANNELS} \
        --output_dir "${OUTPUT_DIR}" \
        --checkpoint_interval 5 \
        >> "${LOG_FILE}" 2>&1
    
    return $?
}

# Verify all paths exist before starting
echo "Verifying data paths..."
ALL_PATHS_VALID=true
for pair in "${RESOLUTION_PAIRS[@]}"; do
    LOW_RES=$(echo $pair | cut -d':' -f1)
    HIGH_RES=$(echo $pair | cut -d':' -f2)
    
    LOW_RES_PATH="${BASE_DATA_DIR}/${LOW_RES}"
    HIGH_RES_PATH="${BASE_DATA_DIR}/${HIGH_RES}"
    
    if [ ! -d "${LOW_RES_PATH}" ]; then
        echo "  ✗ Missing: ${LOW_RES_PATH}"
        ALL_PATHS_VALID=false
    else
        echo "  ✓ Found: ${LOW_RES_PATH}"
    fi
    
    if [ ! -d "${HIGH_RES_PATH}" ]; then
        echo "  ✗ Missing: ${HIGH_RES_PATH}"
        ALL_PATHS_VALID=false
    else
        echo "  ✓ Found: ${HIGH_RES_PATH}"
    fi
done

if [ "$ALL_PATHS_VALID" = false ]; then
    echo ""
    echo "ERROR: Some data paths are missing. Exiting."
    exit 1
fi

echo ""
echo "================================================================"
echo "Launching Parallel Training Jobs"
echo "================================================================"
echo ""

# Array to store PIDs
declare -a PIDS
declare -a PAIRS_INFO

# Launch each encoder training on a separate GPU
GPU_ID=0
for pair in "${RESOLUTION_PAIRS[@]}"; do
    LOW_RES=$(echo $pair | cut -d':' -f1)
    HIGH_RES=$(echo $pair | cut -d':' -f2)
    
    LOG_FILE="logs/encoder_${LOW_RES}_to_${HIGH_RES}_${SLURM_JOB_ID}.log"
    
    echo "Launching on GPU ${GPU_ID}: ${LOW_RES}³ → ${HIGH_RES}³"
    echo "  Log file: ${LOG_FILE}"
    
    # Launch in background
    train_encoder ${GPU_ID} ${LOW_RES} ${HIGH_RES} ${LOG_FILE} &
    PIDS+=($!)
    PAIRS_INFO+=("${pair}")
    
    # Move to next GPU (wrap around if more pairs than GPUs)
    GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
done

echo ""
echo "================================================================"
echo "All jobs launched. Waiting for completion..."
echo "================================================================"
echo ""
echo "Monitor progress with:"
for i in "${!RESOLUTION_PAIRS[@]}"; do
    pair="${RESOLUTION_PAIRS[$i]}"
    LOW_RES=$(echo $pair | cut -d':' -f1)
    HIGH_RES=$(echo $pair | cut -d':' -f2)
    echo "  tail -f logs/encoder_${LOW_RES}_to_${HIGH_RES}_${SLURM_JOB_ID}.log"
done
echo ""

# Wait for all jobs and collect exit statuses
declare -a EXIT_STATUSES
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    EXIT_STATUSES+=($?)
done

echo ""
echo "================================================================"
echo "All Training Jobs Complete"
echo "End Time: $(date)"
echo "================================================================"
echo ""

# Summary
ALL_SUCCESS=true
echo "Results:"
echo "----------------------------------------------------------------"
for i in "${!RESOLUTION_PAIRS[@]}"; do
    pair="${RESOLUTION_PAIRS[$i]}"
    LOW_RES=$(echo $pair | cut -d':' -f1)
    HIGH_RES=$(echo $pair | cut -d':' -f2)
    OUTPUT_DIR="./encoder_checkpoints_stage${LOW_RES}_to_stage${HIGH_RES}"
    EXIT_STATUS=${EXIT_STATUSES[$i]}
    
    if [ ${EXIT_STATUS} -eq 0 ] && [ -f "${OUTPUT_DIR}/encoder_best.pth" ]; then
        SIZE=$(du -h "${OUTPUT_DIR}/encoder_best.pth" | cut -f1)
        echo "  ✓ ${LOW_RES}³ → ${HIGH_RES}³ (${SIZE})"
        echo "    Checkpoint: ${OUTPUT_DIR}/encoder_best.pth"
    else
        echo "  ✗ ${LOW_RES}³ → ${HIGH_RES}³ FAILED (exit code: ${EXIT_STATUS})"
        echo "    Check log: logs/encoder_${LOW_RES}_to_${HIGH_RES}_${SLURM_JOB_ID}.log"
        ALL_SUCCESS=false
    fi
done
echo "----------------------------------------------------------------"

if [ "$ALL_SUCCESS" = true ]; then
    echo ""
    echo "✓ All encoders trained successfully in parallel!"
    echo ""
    echo "Checkpoints ready for use in main training script:"
    for pair in "${RESOLUTION_PAIRS[@]}"; do
        LOW_RES=$(echo $pair | cut -d':' -f1)
        HIGH_RES=$(echo $pair | cut -d':' -f2)
        echo "  encoder_checkpoints_stage${LOW_RES}_to_stage${HIGH_RES}/encoder_best.pth"
    done
    exit 0
else
    echo ""
    echo "⚠ Some encoder trainings failed. Check individual logs for details."
    exit 1
fi