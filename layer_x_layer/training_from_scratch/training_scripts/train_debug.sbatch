#!/bin/bash
#SBATCH --job-name=3d_diffusion_debug
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err
#SBATCH --nodes=1                    # Single node for debugging
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # Fewer CPUs for debug
#SBATCH --mem=128G                   # Less memory for debug
#SBATCH --gres=gpu:2                 # Just 2 GPUs for quick testing
#SBATCH --partition=teaching
#SBATCH --account=undergrad_research
#SBATCH --time=02:00:00              # Short runtime: 2 hours
#SBATCH --mail-type=FAIL             # Only email on failure

# =============================================================================
# DEBUG/TEST Script for 3D Diffusion Model Training
# =============================================================================
# This is a quick test script to verify:
# 1. Distributed training setup works
# 2. Data loading works correctly
# 3. Model can train without errors
# 4. FSDP is functioning properly
#
# Use this before launching full training jobs!
# =============================================================================

echo "=========================================="
echo "DEBUG JOB - Quick Test Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

# =============================================================================
# Environment Setup
# =============================================================================

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate layerxlayer

echo "Environment check:"
which python
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

mkdir -p logs checkpoints

# =============================================================================
# Distributed Setup
# =============================================================================

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NUM_GPUS=$SLURM_GPUS_ON_NODE

echo "=========================================="
echo "Test Configuration"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "=========================================="

# =============================================================================
# Step 1: Test GPU Availability
# =============================================================================

echo ""
echo "=========================================="
echo "STEP 1: Testing GPU Availability"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=========================================="

# =============================================================================
# Step 2: Test Distributed Initialization
# =============================================================================

echo ""
echo "=========================================="
echo "STEP 2: Testing Distributed Initialization"
echo "=========================================="

# Create a temporary test script
cat > /tmp/test_dist_${SLURM_JOB_ID}.py << 'PYEOF'
import os
import torch
import torch.distributed as dist

# CRITICAL: Set the device based on LOCAL_RANK
local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)

# Initialize process group
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.cuda.current_device()

# Create a test tensor on the correct device
test_tensor = torch.ones(1, device=device) * rank

# Test all_reduce
dist.all_reduce(test_tensor)

print(f'[Rank {rank}/{world_size}] Local Rank: {local_rank}, Device: cuda:{device}, Test sum: {test_tensor.item()}')

dist.barrier()
if rank == 0:
    print('✓ Distributed initialization successful!')

dist.destroy_process_group()
PYEOF

# Run the test
torchrun --nproc_per_node=$NUM_GPUS /tmp/test_dist_${SLURM_JOB_ID}.py 2>&1 | tee logs/dist_test_${SLURM_JOB_ID}.log

DIST_TEST_STATUS=${PIPESTATUS[0]}

# Cleanup
rm -f /tmp/test_dist_${SLURM_JOB_ID}.py

if [ $DIST_TEST_STATUS -ne 0 ]; then
    echo "✗ Distributed initialization FAILED!"
    echo "Fix this before running full training."
    exit 1
else
    echo "✓ Distributed initialization PASSED!"
fi

echo "=========================================="

# =============================================================================
# Step 3: Test Training Script (Short Run)
# =============================================================================

echo ""
echo "=========================================="
echo "STEP 3: Running Short Training Test"
echo "=========================================="
echo "Config: 2 epochs, validation every epoch"
echo "=========================================="

TRAIN_CMD="torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --shard_model \
    --sharding_strategy FULL_SHARD \
    --mixed_precision \
    --batch_size 2 \
    --num_epochs 2 \
    --validation_interval 1 \
    --checkpoint_interval 1"

echo "Command: $TRAIN_CMD"
echo ""

$TRAIN_CMD 2>&1 | tee logs/train_test_${SLURM_JOB_ID}.log

TRAIN_STATUS=$?

echo "=========================================="

# =============================================================================
# Results Summary
# =============================================================================

echo ""
echo "=========================================="
echo "DEBUG TEST RESULTS SUMMARY"
echo "=========================================="

if [ $TRAIN_STATUS -eq 0 ]; then
    echo "✓ Training test PASSED!"
    echo ""
    echo "Your setup is working correctly!"
    echo "You can now run the full training with:"
    echo "  sbatch train_single_node.sbatch"
    echo ""
    echo "Checkpoints created:"
    ls -lh checkpoint_*.pth 2>/dev/null || echo "  (none - this is a test run)"
else
    echo "✗ Training test FAILED!"
    echo ""
    echo "Please check the logs for errors:"
    echo "  Error log: logs/debug_${SLURM_JOB_ID}.err"
    echo "  Train log: logs/train_test_${SLURM_JOB_ID}.log"
    echo ""
    echo "Common issues:"
    echo "  1. Data not found - check dataset paths"
    echo "  2. Out of memory - reduce batch size"
    echo "  3. NCCL errors - check GPU compatibility"
    echo "  4. Import errors - check conda environment"
fi

echo "=========================================="
echo "GPU state at end:"
nvidia-smi
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

exit $TRAIN_STATUS
