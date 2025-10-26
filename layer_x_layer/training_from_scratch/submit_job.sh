#!/bin/bash

# =============================================================================
# Helper Script for Submitting SLURM Training Jobs
# =============================================================================
# This script provides easy commands for submitting different training jobs
# 
# Usage:
#   bash submit_job.sh [option]
#
# Or source it and use functions directly:
#   source submit_job.sh
#   submit_debug
# =============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# Check Prerequisites
# =============================================================================

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if SLURM is available
    if ! command -v sbatch &> /dev/null; then
        print_error "sbatch command not found. Are you on a SLURM cluster?"
        return 1
    fi
    print_success "SLURM is available"
    
    # Check if training script exists
    if [ ! -f "train.py" ]; then
        print_warning "train.py not found in current directory"
        print_info "Make sure you're in the correct directory"
    else
        print_success "train.py found"
    fi
    
    # Check if logs directory exists
    if [ ! -d "logs" ]; then
        print_info "Creating logs directory..."
        mkdir -p logs
        print_success "logs directory created"
    else
        print_success "logs directory exists"
    fi
    
    # Check conda environment
    if command -v conda &> /dev/null; then
        print_success "Conda is available"
        if conda env list | grep -q "layerxlayer"; then
            print_success "layerxlayer environment found"
        else
            print_warning "layerxlayer environment not found"
            print_info "Make sure to create it or update the SBATCH scripts"
        fi
    else
        print_warning "Conda not found in PATH"
    fi
    
    echo ""
}

# =============================================================================
# Job Status Functions
# =============================================================================

show_queue() {
    print_header "Your SLURM Jobs"
    squeue -u $USER --format="%.18i %.9P %.20j %.8T %.10M %.6D %.20R"
    echo ""
}

show_job_details() {
    if [ -z "$1" ]; then
        print_error "Usage: show_job_details <job_id>"
        return 1
    fi
    print_header "Job Details: $1"
    scontrol show job $1
    echo ""
}

cancel_job() {
    if [ -z "$1" ]; then
        print_error "Usage: cancel_job <job_id>"
        return 1
    fi
    print_info "Cancelling job $1..."
    scancel $1
    if [ $? -eq 0 ]; then
        print_success "Job $1 cancelled"
    else
        print_error "Failed to cancel job $1"
    fi
    echo ""
}

tail_job_output() {
    if [ -z "$1" ]; then
        print_error "Usage: tail_job_output <job_id>"
        return 1
    fi
    
    OUTPUT_FILE="logs/train_${1}.out"
    ERROR_FILE="logs/train_${1}.err"
    
    if [ -f "$OUTPUT_FILE" ]; then
        print_header "Tailing Output: $OUTPUT_FILE"
        tail -f "$OUTPUT_FILE"
    elif [ -f "$ERROR_FILE" ]; then
        print_header "Tailing Errors: $ERROR_FILE"
        tail -f "$ERROR_FILE"
    else
        print_error "Log files not found for job $1"
        print_info "Looking for: $OUTPUT_FILE or $ERROR_FILE"
    fi
}

# =============================================================================
# Job Submission Functions
# =============================================================================

submit_debug() {
    print_header "Submitting Debug Job (2 GPUs, 2 epochs)"
    print_info "This is a quick test to verify everything works"
    print_info "Runtime: ~10-30 minutes"
    echo ""
    
    JOB_ID=$(sbatch train_debug.sbatch | grep -oP '\d+')
    
    if [ -z "$JOB_ID" ]; then
        print_error "Failed to submit job"
        return 1
    fi
    
    print_success "Job submitted: $JOB_ID"
    print_info "Monitor with: tail_job_output $JOB_ID"
    print_info "Check status: show_queue"
    echo ""
}

submit_single_node() {
    local GPUS=${1:-4}
    
    print_header "Submitting Single-Node Training ($GPUS GPUs)"
    print_info "This will train on a single node with $GPUS GPUs"
    print_info "Runtime: hours to days depending on configuration"
    echo ""
    
    # Modify the SBATCH script temporarily to use the specified GPU count
    sed "s/#SBATCH --gres=gpu:.*/#SBATCH --gres=gpu:$GPUS/" train_single_node.sbatch > train_single_node_temp.sbatch
    
    JOB_ID=$(sbatch train_single_node_temp.sbatch | grep -oP '\d+')
    rm train_single_node_temp.sbatch
    
    if [ -z "$JOB_ID" ]; then
        print_error "Failed to submit job"
        return 1
    fi
    
    print_success "Job submitted: $JOB_ID"
    print_info "Using $GPUS GPUs on single node"
    print_info "Monitor with: tail_job_output $JOB_ID"
    echo ""
}

submit_multinode() {
    local NODES=${1:-2}
    local GPUS=${2:-4}
    
    print_header "Submitting Multi-Node Training ($NODES nodes × $GPUS GPUs)"
    print_info "Total GPUs: $((NODES * GPUS))"
    print_warning "Multi-node jobs may take longer to start"
    echo ""
    
    # Modify the SBATCH script temporarily
    sed -e "s/#SBATCH --nodes=.*/#SBATCH --nodes=$NODES/" \
        -e "s/#SBATCH --gres=gpu:.*/#SBATCH --gres=gpu:$GPUS/" \
        train_multinode.sbatch > train_multinode_temp.sbatch
    
    JOB_ID=$(sbatch train_multinode_temp.sbatch | grep -oP '\d+')
    rm train_multinode_temp.sbatch
    
    if [ -z "$JOB_ID" ]; then
        print_error "Failed to submit job"
        return 1
    fi
    
    print_success "Job submitted: $JOB_ID"
    print_info "Using $NODES nodes with $GPUS GPUs each"
    print_info "Monitor with: tail_job_output $JOB_ID"
    echo ""
}

submit_with_cpu_offload() {
    print_header "Submitting Training with CPU Offload"
    print_info "This will use CPU offload for maximum memory savings"
    print_warning "Training will be slower but can handle larger models"
    echo ""
    
    # Create a modified version with CPU offload enabled
    sed 's/USE_CPU_OFFLOAD="false"/USE_CPU_OFFLOAD="true"/' train_single_node.sbatch > train_cpu_offload_temp.sbatch
    
    JOB_ID=$(sbatch train_cpu_offload_temp.sbatch | grep -oP '\d+')
    rm train_cpu_offload_temp.sbatch
    
    if [ -z "$JOB_ID" ]; then
        print_error "Failed to submit job"
        return 1
    fi
    
    print_success "Job submitted: $JOB_ID"
    print_info "Using CPU offload for memory efficiency"
    echo ""
}

submit_interactive() {
    local GPUS=${1:-2}
    local TIME=${2:-02:00:00}
    
    print_header "Requesting Interactive Session"
    print_info "GPUs: $GPUS"
    print_info "Time: $TIME"
    echo ""
    
    print_info "Starting interactive session..."
    srun --nodes=1 \
         --ntasks-per-node=1 \
         --cpus-per-task=16 \
         --mem=128G \
         --gres=gpu:$GPUS \
         --partition=teaching \
         --account=undergrad_research \
         --time=$TIME \
         --pty bash
}

# =============================================================================
# Resource Check Functions
# =============================================================================

check_available_resources() {
    print_header "Available Cluster Resources"
    
    print_info "Partition Information:"
    sinfo --format="%.10P %.5a %.10l %.6D %.6c %.8m %.10G %.20C"
    echo ""
    
    print_info "Node Details:"
    sinfo -N --format="%.10N %.6D %.6c %.8m %.10G %.10T"
    echo ""
}

estimate_queue_time() {
    local GPUS=${1:-4}
    
    print_header "Queue Estimation"
    print_info "Requesting $GPUS GPUs"
    
    # Show current queue for the partition
    print_info "Current jobs in teaching partition:"
    squeue -p teaching --format="%.10i %.9P %.8u %.2t %.10M %.5D %.6C %.20R" | head -20
    
    echo ""
    print_warning "Actual queue time depends on cluster load and job priorities"
}

# =============================================================================
# Monitoring Functions
# =============================================================================

watch_gpu_usage() {
    if [ -z "$1" ]; then
        print_error "Usage: watch_gpu_usage <job_id>"
        return 1
    fi
    
    JOB_ID=$1
    
    # Get the node(s) running the job
    NODES=$(squeue -j $JOB_ID -h -o "%N")
    
    if [ -z "$NODES" ]; then
        print_error "Job $JOB_ID not found or not running"
        return 1
    fi
    
    print_header "GPU Usage for Job $JOB_ID on $NODES"
    print_info "Press Ctrl+C to stop monitoring"
    echo ""
    
    # Watch GPU usage on the node
    watch -n 2 "ssh $NODES 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits'"
}

show_job_efficiency() {
    if [ -z "$1" ]; then
        print_error "Usage: show_job_efficiency <job_id>"
        return 1
    fi
    
    print_header "Job Efficiency Report"
    seff $1
    echo ""
}

# =============================================================================
# Cleanup Functions
# =============================================================================

cleanup_old_logs() {
    local DAYS=${1:-7}
    
    print_header "Cleaning Up Old Logs"
    print_info "Removing log files older than $DAYS days"
    
    find logs/ -name "*.out" -type f -mtime +$DAYS -delete
    find logs/ -name "*.err" -type f -mtime +$DAYS -delete
    
    print_success "Cleanup complete"
    echo ""
}

cleanup_old_checkpoints() {
    print_header "Checkpoint Cleanup"
    print_warning "This will list checkpoints - review before deleting!"
    
    echo "Checkpoint files:"
    ls -lh checkpoint_*.pth 2>/dev/null || echo "No checkpoints found"
    
    echo ""
    print_info "To delete old checkpoints:"
    print_info "  find . -name 'checkpoint_epoch_*.pth' -type f | sort | head -n -5 | xargs rm"
}

# =============================================================================
# Help and Menu
# =============================================================================

show_help() {
    cat << EOF

${BLUE}========================================${NC}
  SLURM Training Job Submission Helper
${BLUE}========================================${NC}

${GREEN}Job Submission:${NC}
  submit_debug                     - Quick test run (2 GPUs, 2 epochs)
  submit_single_node [gpus]        - Single node training (default: 4 GPUs)
  submit_multinode [nodes] [gpus]  - Multi-node training (default: 2×4)
  submit_with_cpu_offload          - Training with CPU offload enabled
  submit_interactive [gpus] [time] - Interactive session

${GREEN}Job Management:${NC}
  show_queue                       - Show your jobs in queue
  show_job_details <job_id>        - Show detailed job information
  cancel_job <job_id>              - Cancel a job
  tail_job_output <job_id>         - Follow job output in real-time

${GREEN}Monitoring:${NC}
  watch_gpu_usage <job_id>         - Watch GPU usage for a job
  show_job_efficiency <job_id>     - Show job efficiency report

${GREEN}Resources:${NC}
  check_available_resources        - Check cluster availability
  estimate_queue_time [gpus]       - Estimate wait time

${GREEN}Maintenance:${NC}
  cleanup_old_logs [days]          - Remove old log files (default: 7 days)
  cleanup_old_checkpoints          - List and help remove old checkpoints
  check_prerequisites              - Verify setup

${GREEN}Examples:${NC}
  ${YELLOW}# Run a quick test${NC}
  submit_debug

  ${YELLOW}# Train on 4 GPUs (single node)${NC}
  submit_single_node 4

  ${YELLOW}# Train on 8 GPUs (2 nodes × 4 GPUs)${NC}
  submit_multinode 2 4

  ${YELLOW}# Monitor a running job${NC}
  show_queue
  tail_job_output 12345
  watch_gpu_usage 12345

  ${YELLOW}# Interactive session for debugging${NC}
  submit_interactive 2 01:00:00

${BLUE}========================================${NC}

EOF
}

show_menu() {
    print_header "Interactive Job Submission Menu"
    
    echo "1) Quick Test (Debug)"
    echo "2) Single Node Training (4 GPUs)"
    echo "3) Single Node Training (8 GPUs)"
    echo "4) Multi-Node Training (2×4 GPUs)"
    echo "5) Training with CPU Offload"
    echo "6) Interactive Session"
    echo "7) Show Queue"
    echo "8) Check Available Resources"
    echo "9) Help"
    echo "0) Exit"
    echo ""
    
    read -p "Select option: " choice
    
    case $choice in
        1) submit_debug ;;
        2) submit_single_node 4 ;;
        3) submit_single_node 8 ;;
        4) submit_multinode 2 4 ;;
        5) submit_with_cpu_offload ;;
        6) 
            read -p "Number of GPUs (default: 2): " gpus
            read -p "Time limit (default: 02:00:00): " time
            submit_interactive ${gpus:-2} ${time:-02:00:00}
            ;;
        7) show_queue ;;
        8) check_available_resources ;;
        9) show_help ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) print_error "Invalid option" ;;
    esac
}

# =============================================================================
# Main Execution
# =============================================================================

# If script is run directly (not sourced), show menu or execute command
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    if [ -z "$1" ]; then
        # No arguments, show menu
        check_prerequisites
        show_menu
    else
        # Execute the specified function
        case $1 in
            debug)
                check_prerequisites
                submit_debug
                ;;
            single)
                check_prerequisites
                submit_single_node ${2:-4}
                ;;
            multi)
                check_prerequisites
                submit_multinode ${2:-2} ${3:-4}
                ;;
            interactive)
                submit_interactive ${2:-2} ${3:-02:00:00}
                ;;
            queue)
                show_queue
                ;;
            help)
                show_help
                ;;
            *)
                print_error "Unknown command: $1"
                echo ""
                show_help
                exit 1
                ;;
        esac
    fi
else
    # Script is being sourced, just make functions available
    print_success "Helper functions loaded!"
    print_info "Type 'show_help' for usage information"
    echo ""
fi
