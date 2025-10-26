#!/bin/bash
#SBATCH --job-name=3d_diffusion_multinode
#SBATCH --output=logs/train_multinode_%j.out
#SBATCH --error=logs/train_multinode_%j.err
#SBATCH --nodes=2                    # Number of nodes (2, 4, 8, etc.)
#SBATCH --ntasks-per-node=1          # One task per node (torchrun handles GPUs)
#SBATCH --cpus-per-task=32           # CPUs per task
#SBATCH --mem=256G                   # Memory per node
#SBATCH --gres=gpu:4                 # GPUs per node (must be same on all nodes)
#SBATCH --partition=teaching         # GPU partition
#SBATCH --account=undergrad_research
#SBATCH --time=72:00:00              # Max runtime (72 hours)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@msoe.edu

# =============================================================================
# SLURM Multi-Node Distributed Training Script for 3D Diffusion Model
# =============================================================================
# This script runs distributed training across MULTIPLE nodes with multiple
# GPUs per node using PyTorch FSDP
# 
# Total GPUs = nodes × gpus_per_node = 2 × 4 = 8 GPUs in this example
# =============================================================================

echo "=========================================="
echo "Multi-Node Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Master node: $SLURMD_NODENAME"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"

# Activate environment
conda activate layerxlayer

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Create directories
mkdir -p logs checkpoints training_samples t5_cache

# =============================================================================
# Distributed Training Setup for Multi-Node
# =============================================================================

# Get the master node hostname (first node in the allocation)
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

# Number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# Node rank will be set by srun
# SLURM_PROCID will be used as node_rank

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                # Enable InfiniBand
export NCCL_SOCKET_IFNAME=ib0           # Network interface
export NCCL_IB_GID_INDEX=3              # InfiniBand GID (may need adjustment)
export NCCL_IB_HCA=mlx5_0               # InfiniBand adapter (may need adjustment)
export NCCL_NET_GDR_LEVEL=5             # GPU Direct RDMA level
export NCCL_TIMEOUT=7200                # 2 hour timeout

# Additional performance tuning
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_NCHANNELS=4

# CUDA settings
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=INFO

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "Distributed Training Configuration"
echo "=========================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "=========================================="

# =============================================================================
# Test Network Connectivity (Optional but Recommended)
# =============================================================================

echo "=========================================="
echo "Testing network connectivity..."
echo "=========================================="

# Test if master node is reachable from all nodes
srun --nodes=$NNODES --ntasks=$NNODES bash -c "
    echo \"Node \$(hostname) can reach master: \$(ping -c 1 -W 1 $MASTER_ADDR > /dev/null 2>&1 && echo 'YES' || echo 'NO')\"
"

echo "=========================================="

# =============================================================================
# GPU Information on All Nodes
# =============================================================================

echo "=========================================="
echo "GPU Information on All Nodes"
echo "=========================================="

srun --nodes=$NNODES --ntasks=$NNODES bash -c "
    echo \"=== Node \$(hostname) ===\"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo \"\"
"

echo "=========================================="

# =============================================================================
# Training Configuration
# =============================================================================

TRAIN_SCRIPT="train.py"
BATCH_SIZE=4                    # Batch size per GPU
NUM_EPOCHS=500
VALIDATION_INTERVAL=10
CHECKPOINT_INTERVAL=50
SHARDING_STRATEGY="FULL_SHARD"
USE_CPU_OFFLOAD="false"

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * WORLD_SIZE))

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Training script: $TRAIN_SCRIPT"
echo "Total GPUs: $WORLD_SIZE"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Sharding strategy: $SHARDING_STRATEGY"
echo "=========================================="

# =============================================================================
# Run Multi-Node Training with srun
# =============================================================================

echo "Starting multi-node training at $(date)"
echo "=========================================="

# Build the training command
# Note: We use srun to launch torchrun on each node
LAUNCH_CMD="torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=\$SLURM_PROCID \
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

if [ "$USE_CPU_OFFLOAD" = "true" ]; then
    LAUNCH_CMD="$LAUNCH_CMD --cpu_offload"
fi

echo "Launching training on all nodes with command:"
echo "$LAUNCH_CMD"
echo "=========================================="

# Use srun to launch the training on all nodes
# Each node will run torchrun with its own node_rank
srun bash -c "$LAUNCH_CMD"

EXIT_STATUS=$?

# =============================================================================
# Post-Training
# =============================================================================

echo "=========================================="
echo "Multi-node training completed at $(date)"
echo "Exit status: $EXIT_STATUS"
echo "=========================================="

# Show final GPU state on master node
echo "Final GPU state on master node:"
nvidia-smi

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✓ Multi-node training completed successfully!"
    echo "Total GPUs used: $WORLD_SIZE"
    echo "Training time: See timestamps above"
else
    echo "✗ Training failed with exit status $EXIT_STATUS"
    echo "Check error logs: logs/train_multinode_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

exit $EXIT_STATUS
