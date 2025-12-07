#!/bin/bash

# =============================================================================
# Quick Start Scripts for Running Training with FSDP
# =============================================================================
# This script contains various configurations for running your training script
# with different FSDP settings. Uncomment the configuration you want to use.
#
# Prerequisites:
# 1. PyTorch 2.0+ installed
# 2. NCCL properly configured for multi-GPU communication
# 3. Training script modified with FSDP code from practical_fsdp_implementation.py
# =============================================================================

# =============================================================================
# BASIC CONFIGURATIONS
# =============================================================================

# 1. Single GPU (No Sharding) - Baseline
# Use this for testing or when you only have 1 GPU
# python train.py

# 2. Multi-GPU with Automatic Sharding (Recommended for most cases)
# Automatically uses all available GPUs
# torchrun --nproc_per_node=auto train.py --shard_model

# 3. Multi-GPU with Specific Number of GPUs
# Explicitly specify 4 GPUs
# torchrun --nproc_per_node=4 train.py --shard_model


# =============================================================================
# MEMORY-OPTIMIZED CONFIGURATIONS
# =============================================================================

# 4. Maximum Memory Savings (for very large models)
# Uses full sharding + CPU offload
# Best when: Model doesn't fit in GPU memory even with sharding
# torchrun --nproc_per_node=auto train.py \
#     --shard_model \
#     --sharding_strategy FULL_SHARD \
#     --cpu_offload \
#     --mixed_precision


# 5. Balanced Memory and Speed
# Shards only gradients and optimizer states, keeps model replicated
# Best when: Model fits in GPU but optimizer states are large
# torchrun --nproc_per_node=auto train.py \
#     --shard_model \
#     --sharding_strategy SHARD_GRAD_OP \
#     --mixed_precision


# 6. Full Model Sharding without CPU Offload (Recommended)
# Good balance of memory savings and performance
# Best when: Model is large but can fit with sharding alone
torchrun --nproc_per_node=4 train_v4.py \
    --shard_model \
    --sharding_strategy FULL_SHARD \
    --mixed_precision


# =============================================================================
# PERFORMANCE TUNING CONFIGURATIONS
# =============================================================================

# 7. DDP-style (No Sharding) for Performance Comparison
# Model is replicated on all GPUs like traditional DDP
# Use this to compare performance with sharded version
# torchrun --nproc_per_node=4 train.py \
#     --shard_model \
#     --sharding_strategy NO_SHARD


# 8. Large Batch Size Training
# Increase batch size to utilize multiple GPUs better
# torchrun --nproc_per_node=4 train.py \
#     --shard_model \
#     --batch_size 8 \
#     --mixed_precision


# 9. Extended Training Run
# torchrun --nproc_per_node=auto train.py \
#     --shard_model \
#     --num_epochs 1000 \
#     --validation_interval 20 \
#     --checkpoint_interval 100


# =============================================================================
# DEBUGGING CONFIGURATIONS
# =============================================================================

# 10. Debug Mode with NCCL Information
# Shows detailed communication logs for debugging
# NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py --shard_model


# 11. Test Run with 2 GPUs (easier to debug than 4+)
# torchrun --nproc_per_node=2 train.py \
#     --shard_model \
#     --num_epochs 5 \
#     --validation_interval 1


# 12. Deterministic Run (for reproducibility)
# CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --nproc_per_node=2 train.py \
#     --shard_model \
#     --num_epochs 10


# =============================================================================
# MULTI-NODE CONFIGURATIONS (for clusters)
# =============================================================================

# 13. Multi-Node Training - Node 0 (Master)
# Run this on the first machine
# Replace MASTER_IP with your master node's IP address
# torchrun \
#     --nproc_per_node=4 \
#     --nnodes=2 \
#     --node_rank=0 \
#     --master_addr="MASTER_IP" \
#     --master_port=29500 \
#     train.py --shard_model


# 14. Multi-Node Training - Node 1 (Worker)
# Run this on the second machine
# torchrun \
#     --nproc_per_node=4 \
#     --nnodes=2 \
#     --node_rank=1 \
#     --master_addr="MASTER_IP" \
#     --master_port=29500 \
#     train.py --shard_model


# =============================================================================
# PRODUCTION CONFIGURATIONS
# =============================================================================

# 15. Full Production Run with All Features
# Maximum performance and memory efficiency
# torchrun --nproc_per_node=8 train.py \
#     --shard_model \
#     --sharding_strategy FULL_SHARD \
#     --mixed_precision \
#     --num_epochs 500 \
#     --batch_size 4 \
#     --validation_interval 10 \
#     --checkpoint_interval 50


# 16. Long Training Run with Monitoring
# Pipe output to log file for long runs
# torchrun --nproc_per_node=auto train.py \
#     --shard_model \
#     --mixed_precision \
#     --num_epochs 1000 \
#     2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt


# 17. Resume from Checkpoint
# (Assuming you've added checkpoint loading logic)
# torchrun --nproc_per_node=auto train.py \
#     --shard_model \
#     --resume_from checkpoint_epoch_250.pth


# =============================================================================
# HELPER SCRIPTS
# =============================================================================

# Check GPU availability and CUDA version
check_gpus() {
    echo "=== GPU Information ==="
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
    echo "CUDA Version:"
    nvcc --version | grep release
    echo ""
    echo "PyTorch CUDA Available:"
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
}

# Monitor GPU usage during training
monitor_gpus() {
    echo "Starting GPU monitor (Ctrl+C to stop)..."
    watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
}

# Test distributed setup
test_distributed() {
    echo "Testing distributed setup with $1 GPUs..."
    torchrun --nproc_per_node=$1 python -c "
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.cuda.current_device()
print(f'[Rank {rank}/{world_size}] Device: cuda:{device} | Test tensor: {torch.ones(1).cuda()}')
dist.barrier()
if rank == 0:
    print('✓ All processes initialized successfully!')
dist.destroy_process_group()
"
}

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

show_help() {
    cat << EOF

=============================================================================
                    FSDP Training Script Helper
=============================================================================

This script provides quick commands for running your training with various
FSDP configurations.

QUICK START:
-----------
1. Single GPU:          python train.py
2. Multi-GPU (Auto):    torchrun --nproc_per_node=auto train.py --shard_model
3. With CPU Offload:    torchrun --nproc_per_node=auto train.py --shard_model --cpu_offload

HELPER FUNCTIONS:
----------------
Run these commands directly from your terminal:

  source $(basename $0)              # Source this file to use functions
  check_gpus                         # Show GPU information
  monitor_gpus                       # Monitor GPU usage (Ctrl+C to stop)
  test_distributed 4                 # Test distributed with 4 GPUs

COMMON CONFIGURATIONS:
---------------------
Uncomment the desired configuration in this script and run:
  bash $(basename $0)

Configuration #4  - Maximum memory savings (CPU offload)
Configuration #6  - Recommended: Full sharding without CPU offload
Configuration #8  - Large batch size training
Configuration #15 - Full production run

TROUBLESHOOTING:
---------------
1. Check GPUs are visible:         nvidia-smi
2. Test distributed initialization: test_distributed 2
3. Enable debug logging:           NCCL_DEBUG=INFO torchrun ...
4. Check CUDA/PyTorch:             python -c "import torch; print(torch.cuda.is_available())"

For more details, see:
- model_sharding_implementation_guide.md
- practical_fsdp_implementation.py

=============================================================================
EOF
}

# Uncomment this to show help by default
# show_help

# =============================================================================
# DEFAULT RUN
# =============================================================================
# Uncomment ONE of these to set your default training configuration

# Default: Basic multi-GPU training with sharding
# torchrun --nproc_per_node=auto train.py --shard_model --mixed_precision

# For debugging/testing:
# test_distributed 2
# check_gpus
