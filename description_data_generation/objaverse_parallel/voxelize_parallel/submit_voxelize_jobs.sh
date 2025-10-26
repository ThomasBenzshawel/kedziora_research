#!/bin/bash
# Submit preprocessing and workers as separate dependent jobs
# Now with smart cache checking!

# Configuration
SCAN_DIR="/home/ad.msoe.edu/benzshawelt/.objaverse"
SCAN_DIR_2="/home/ad.msoe.edu/benzshawelt/objaverse_temp"
OUTPUT_DIR="../objaverse_voxels"
FILE_LIST="logs/glb_file_list.json"
VOXEL_RESOLUTION=16
NUM_WORKERS=17

# Create logs directory
mkdir -p logs

# Check if we should force re-scan
FORCE_RESCAN=false
if [ "$1" == "--force" ] || [ "$1" == "-f" ]; then
    FORCE_RESCAN=true
    echo "Force re-scan enabled"
fi

echo "==================== CHECKING CACHE ===================="

# Check if file list already exists
if [ -f "$FILE_LIST" ] && [ "$FORCE_RESCAN" = false ]; then
    echo "✓ Found existing file list: $FILE_LIST"
    
    # Extract info from existing cache
    TOTAL_FILES=$(python3 -c "import json; data=json.load(open('$FILE_LIST')); print(data['total_files'])" 2>/dev/null || echo "unknown")
    SCAN_TIME=$(python3 -c "import json; data=json.load(open('$FILE_LIST')); print(data['scan_timestamp'])" 2>/dev/null || echo "unknown")
    
    echo "  Total files: $TOTAL_FILES"
    echo "  Scanned at: $SCAN_TIME"
    echo ""
    echo "Skipping preprocessing - using cached file list"
    echo "To force re-scan, run: bash $0 --force"
    echo ""
    
    # Skip preprocessing, go straight to workers
    SKIP_PREPROCESSING=true
else
    if [ "$FORCE_RESCAN" = true ]; then
        echo "Force re-scan requested - will regenerate file list"
    else
        echo "No existing file list found - will run preprocessing"
    fi
    SKIP_PREPROCESSING=false
fi

echo "========================================================"
echo ""

if [ "$SKIP_PREPROCESSING" = false ]; then
    echo "==================== SUBMITTING JOBS ===================="
    echo "This will submit two jobs:"
    echo "  1. Preprocessing job (scans directories)"
    echo "  2. Array job with workers (waits for preprocessing)"
    echo "=========================================================="

    # Submit preprocessing job
    PREPROCESS_JOB=$(sbatch --parsable \
        --job-name=preprocess_files \
        --output=logs/preprocess_%j.out \
        --error=logs/preprocess_%j.err \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=8G \
        --partition=teaching \
        --account=undergrad_research \
        <<'EOFPREPROCESS'
#!/bin/bash
. ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /home/ad.msoe.edu/benzshawelt/.conda/envs/voxelize
python3 preprocess_file_list.py \
    --scan_dir '/home/ad.msoe.edu/benzshawelt/.objaverse' \
    --scan_dir_2 '/home/ad.msoe.edu/benzshawelt/objaverse_temp' \
    --output_file 'logs/glb_file_list.json'
EOFPREPROCESS
    )

    echo "Submitted preprocessing job: $PREPROCESS_JOB"

    # Submit worker array job with dependency on preprocessing
    WORKER_JOB=$(sbatch --parsable \
        --dependency=afterok:$PREPROCESS_JOB \
        --array=0-$((NUM_WORKERS-1)) \
        --job-name=voxelize_workers \
        --output=logs/voxel_%A_%a.out \
        --error=logs/voxel_%A_%a.err \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --partition=teaching \
        --account=undergrad_research \
        voxelize_worker_array.sh "$FILE_LIST" "$OUTPUT_DIR" $VOXEL_RESOLUTION)

    echo "Submitted worker array job: $WORKER_JOB"
    echo ""
    echo "=========================================================="
    echo "Job submission complete!"
    echo ""
    echo "Job $PREPROCESS_JOB will scan directories first"
    echo "Job $WORKER_JOB will start after preprocessing completes"
    echo ""
    echo "Monitor with: squeue -u $USER"
    echo "Cancel with: scancel $PREPROCESS_JOB $WORKER_JOB"
    echo "=========================================================="
    
else
    # Cache exists, submit workers directly without preprocessing
    echo "==================== SUBMITTING WORKERS ===================="
    echo "Using existing file list: $FILE_LIST"
    echo "Submitting $NUM_WORKERS workers..."
    echo "============================================================"

    WORKER_JOB=$(sbatch --parsable \
        --array=0-$((NUM_WORKERS-1)) \
        --job-name=voxelize_workers \
        --output=logs/voxel_%A_%a.out \
        --error=logs/voxel_%A_%a.err \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --partition=teaching \
        --account=undergrad_research \
        voxelize_worker_array.sh "$FILE_LIST" "$OUTPUT_DIR" $VOXEL_RESOLUTION)

    echo "Submitted worker array job: $WORKER_JOB"
    echo ""
    echo "============================================================"
    echo "Job submission complete!"
    echo ""
    echo "Workers will start immediately using cached file list"
    echo ""
    echo "Monitor with: squeue -u $USER"
    echo "Cancel with: scancel $WORKER_JOB"
    echo "============================================================"
fi