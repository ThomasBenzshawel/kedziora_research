#!/bin/bash
#SBATCH --job-name=3d_diffusion_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks-per-node=1          # One task (torchrun will handle multi-GPU)
#SBATCH --cpus-per-task=32           # CPUs for data loading (adjust based on GPU count)
#SBATCH --mem=256G                   # Memory per node
#SBATCH --gres=gpu:4                 # Request 4 GPUs (change to 2, 4, 8 as needed)
#SBATCH --partition=dgxh100          # GPU partition (adjust for your cluster)
#SBATCH --account=undergrad_research

# =============================================================================
# SLURM Distributed Training Script for 3D Diffusion Model
# =============================================================================
# This script runs distributed training on a single node with multiple GPUs
# using PyTorch FSDP (Fully Sharded Data Parallel)
# =============================================================================

echo "=========================================="
echo "Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Initialize conda using miniforge3
source /usr/local/miniforge3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/layerxlayer

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python
python --version
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p training_samples
mkdir -p t5_cache

# =============================================================================
# Distributed Training Environment Variables
# =============================================================================

# Set master address and port for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Additional NCCL tuning (optional, adjust based on your network)
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_TIMEOUT=7200            # 2 hours timeout for long operations

# Set CUDA options
export CUDA_LAUNCH_BLOCKING=0       # Asynchronous CUDA operations
export TORCH_DISTRIBUTED_DEBUG=INFO # Set to DETAIL for more verbose output

# OMP settings for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "Environment Variables"
echo "=========================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "=========================================="

# =============================================================================
# GPU Information
# =============================================================================

echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo "=========================================="

# =============================================================================
# Training Configuration
# =============================================================================

# Training script path
TRAIN_SCRIPT="train_v3.py"

# Number of GPUs to use (automatically detected from SLURM)
NUM_GPUS=$SLURM_GPUS_ON_NODE

# Training arguments
BATCH_SIZE=4                    # Batch size per GPU
NUM_EPOCHS=500
VALIDATION_INTERVAL=10
CHECKPOINT_INTERVAL=50

# FSDP configuration
SHARDING_STRATEGY="FULL_SHARD"  # Options: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
USE_CPU_OFFLOAD="false"         # Set to "true" if running out of memory

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Training script: $TRAIN_SCRIPT"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Number of epochs: $NUM_EPOCHS"
echo "Sharding strategy: $SHARDING_STRATEGY"
echo "CPU offload: $USE_CPU_OFFLOAD"
echo "=========================================="

# =============================================================================
# Run Training
# =============================================================================

echo "Starting training at $(date)"
echo "=========================================="

# Build the training command
CMD="torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --shard_model \
    --sharding_strategy $SHARDING_STRATEGY \
    --mixed_precision \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --validation_interval $VALIDATION_INTERVAL \
    --checkpoint_interval $CHECKPOINT_INTERVAL"

# Add CPU offload flag if enabled
if [ "$USE_CPU_OFFLOAD" = "true" ]; then
    CMD="$CMD --cpu_offload"
fi

# Log the full command
echo "Executing command:"
echo "$CMD"
echo "=========================================="

# Run the training
$CMD

# Capture exit status
EXIT_STATUS=$?

# =============================================================================
# Post-Training
# =============================================================================

echo "=========================================="
echo "Training completed at $(date)"
echo "Exit status: $EXIT_STATUS"
echo "=========================================="

# Log GPU stats at end
echo "Final GPU state:"
nvidia-smi

# Check if training completed successfully
if [ $EXIT_STATUS -eq 0 ]; then
    echo "✓ Training completed successfully!"
    
    # Optional: Copy important files to a backup location
    # cp best_model.pth /path/to/backup/
    # cp training_metrics.json /path/to/backup/
else
    echo "✗ Training failed with exit status $EXIT_STATUS"
    echo "Check logs for details: logs/train_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

exit $EXIT_STATUS
