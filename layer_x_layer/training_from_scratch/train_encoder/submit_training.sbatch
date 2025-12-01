#!/bin/bash
#SBATCH --job-name=encoder_multi
#SBATCH --output=logs/encoder_multi_%j.out
#SBATCH --error=logs/encoder_multi_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=dgxh100
#SBATCH --account=undergrad_research
#SBATCH --time=96:00:00

# train_encoder_progressive.sbatch
# This script trains encoders for ALL resolution pairs in your progressive pipeline

# Create logs directory
mkdir -p logs

echo "================================================================"
echo "Progressive Encoder Training"
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

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand for H100
export PYTHONUNBUFFERED=1

echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Base directory
BASE_DATA_DIR="/home/ad.msoe.edu/benzshawelt/Kedziora/kedziora_research/description_data_generation/objaverse_parallel/objaverse_voxels"

# Get number of GPUs
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

# Array of resolution pairs to train
# Format: "low_res:high_res"
RESOLUTION_PAIRS=(
    "16:32"
    "32:64"
)

# Training parameters
EPOCHS=50
BATCH_SIZE=8
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
echo "GPUs:             ${NUM_GPUS}"
echo "================================================================"
echo ""

# Track overall success
ALL_SUCCESS=true

# Train each resolution pair sequentially
for pair in "${RESOLUTION_PAIRS[@]}"; do
    # Split the pair
    LOW_RES=$(echo $pair | cut -d':' -f1)
    HIGH_RES=$(echo $pair | cut -d':' -f2)
    
    LOW_RES_PATH="${BASE_DATA_DIR}/${LOW_RES}"
    HIGH_RES_PATH="${BASE_DATA_DIR}/${HIGH_RES}"
    OUTPUT_DIR="./encoder_checkpoints_${LOW_RES}_to_${HIGH_RES}"
    
    echo ""
    echo "================================================================"
    echo "Training Encoder: ${LOW_RES}³ → ${HIGH_RES}³"
    echo "================================================================"
    echo "Low-res path:  ${LOW_RES_PATH}"
    echo "High-res path: ${HIGH_RES_PATH}"
    echo "Output dir:    ${OUTPUT_DIR}"
    echo "================================================================"
    echo ""
    
    # Verify paths
    if [ ! -d "${LOW_RES_PATH}" ]; then
        echo "ERROR: Low-res path not found: ${LOW_RES_PATH}"
        echo "Skipping this pair..."
        ALL_SUCCESS=false
        continue
    fi
    
    if [ ! -d "${HIGH_RES_PATH}" ]; then
        echo "ERROR: High-res path not found: ${HIGH_RES_PATH}"
        echo "Skipping this pair..."
        ALL_SUCCESS=false
        continue
    fi
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Run training
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        train_low_res_encoder.py \
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
            --use_fsdp \
            --mixed_precision \
            --output_dir "${OUTPUT_DIR}" \
            --checkpoint_interval 5
    
    EXIT_STATUS=$?
    
    if [ ${EXIT_STATUS} -eq 0 ]; then
        echo ""
        echo "✓ Successfully trained encoder: ${LOW_RES}³ → ${HIGH_RES}³"
        echo "  Output: ${OUTPUT_DIR}"
        echo ""
    else
        echo ""
        echo "✗ Failed to train encoder: ${LOW_RES}³ → ${HIGH_RES}³"
        echo "  Exit status: ${EXIT_STATUS}"
        echo ""
        ALL_SUCCESS=false
    fi
    
    # Brief pause between trainings to ensure cleanup
    sleep 10
done

echo ""
echo "================================================================"
echo "All Encoder Training Complete"
echo "End Time: $(date)"
echo "================================================================"
echo ""
echo "Summary of trained encoders:"
echo "================================================================"

for pair in "${RESOLUTION_PAIRS[@]}"; do
    LOW_RES=$(echo $pair | cut -d':' -f1)
    HIGH_RES=$(echo $pair | cut -d':' -f2)
    OUTPUT_DIR="./encoder_checkpoints_${LOW_RES}_to_${HIGH_RES}"
    
    if [ -f "${OUTPUT_DIR}/encoder_best.pth" ]; then
        SIZE=$(du -h "${OUTPUT_DIR}/encoder_best.pth" | cut -f1)
        echo "  ✓ ${LOW_RES}³ → ${HIGH_RES}³"
        echo "    File: ${OUTPUT_DIR}/encoder_best.pth (${SIZE})"
    else
        echo "  ✗ ${LOW_RES}³ → ${HIGH_RES}³: FAILED or incomplete"
    fi
    echo ""
done

echo "================================================================"

if [ "$ALL_SUCCESS" = true ]; then
    echo ""
    echo "✓ All encoders trained successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Verify checkpoints in encoder_checkpoints_*/ directories"
    echo "  2. Load them in your main training script with:"
    echo "     load_pretrained_encoder(model, 'encoder_checkpoints_16_to_32/encoder_best.pth')"
    exit 0
else
    echo ""
    echo "⚠ Some encoder trainings failed. Check logs for details."
    exit 1
fi