#!/bin/bash
# Parallel ObjectNav to StreamVLN converter
# Usage: ./run_parallel_conversion.sh <gpu_id> <annot_dir> <data_dir>

set -e

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <gpu_id> <annot_dir> <data_dir>"
    echo ""
    echo "Example:"
    echo "  $0 0 ./data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/suzhou-room-zhangbo-metacam-2025-07-09_22-27-19 ./data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
    echo ""
    echo "This will launch 4 parallel jobs for annotation_0.json to annotation_3.json"
    exit 1
fi

GPU_ID=$1
ANNOT_DIR=$2
DATA_DIR=$3

# Remove trailing slash if present
ANNOT_DIR=${ANNOT_DIR%/}
DATA_DIR=${DATA_DIR%/}

echo "========================================="
echo "Parallel Converter Launcher"
echo "========================================="
echo "GPU Device: $GPU_ID"
echo "Annotation Directory: $ANNOT_DIR"
echo "Data Directory: $DATA_DIR"
echo ""

# Extract scene name from path for display
SCENE_NAME=$(basename "$ANNOT_DIR")
echo "Scene: $SCENE_NAME"
echo ""

# Check if annotation split files exist
ANNOT_BASE="${ANNOT_DIR}/annotations.json"
if [ ! -f "$ANNOT_BASE" ]; then
    echo "Error: $ANNOT_BASE not found!"
    echo "Please run split_annotations.py first to create annotation_0.json to annotation_3.json"
    exit 1
fi

# Check if split files exist
SPLIT_FILES=()
for i in {0..3}; do
    FILE="${ANNOT_DIR}/annotation_${i}.json"
    if [ ! -f "$FILE" ]; then
        echo "Error: $FILE not found!"
        echo "Please run: python scripts/objnav_converters/split_annotations.py $ANNOT_BASE --num-parts 4"
        exit 1
    fi
    SPLIT_FILES+=("$FILE")
done

echo "Found 4 annotation split files:"
for i in "${!SPLIT_FILES[@]}"; do
    FILE="${SPLIT_FILES[$i]}"
    ITEMS=$(python -c "import json; print(len(json.load(open('$FILE'))))")
    echo "  $(basename $FILE): $ITEMS episodes"
done
echo ""

# Extract scene name from annotation directory
SCENE_NAME=$(basename "$ANNOT_DIR")

# Find matching data file
DATA_FILE=$(find "$DATA_DIR" -name "${SCENE_NAME}.json.gz" 2>/dev/null | head -1)
if [ -z "$DATA_FILE" ]; then
    echo "Error: Cannot find data file matching ${SCENE_NAME}.json.gz in $DATA_DIR"
    exit 1
fi

echo "Using data file: $DATA_FILE"
echo ""
echo "Launching 4 parallel jobs..."
echo "========================================="
echo ""

# Function to run a single job
run_job() {
    local job_id=$1
    local annot_file=$2
    local gpu_id=$3
    local data_file=$4

    echo "[Job $job_id] Starting..."
    echo "[Job $job_id] Annotation: $annot_file"
    echo "[Job $job_id] Data: $data_file"
    echo "[Job $job_id] GPU: $gpu_id"
    echo ""

    CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python ./scripts/objnav_converters/objnav2streamvln.py \
        --annot-path "$annot_file" \
        habitat.dataset.data_path="$data_file"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[Job $job_id] ✓ Completed successfully"
    else
        echo "[Job $job_id] ✗ Failed with exit code $exit_code"
    fi
    echo ""
    return $exit_code
}

# Export function for subshell
export -f run_job

# Launch 4 jobs in parallel
pids=()

for i in {0..3}; do
    annot_file="${ANNOT_DIR}/annotation_${i}.json"

    # Run in background
    run_job $i "$annot_file" "$GPU_ID" "$DATA_FILE" &
    pids+=($!)
done

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
echo ""

failed_jobs=0
for i in {0..3}; do
    pid=${pids[$i]}
    wait $pid
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        failed_jobs=$((failed_jobs + 1))
    fi
done

echo "========================================="
echo "All jobs completed!"
echo "========================================="
echo "Failed jobs: $failed_jobs / 4"
echo ""

if [ $failed_jobs -eq 0 ]; then
    echo "✓ All jobs completed successfully!"
    exit 0
else
    echo "✗ Some jobs failed. Please check the logs above."
    exit 1
fi
