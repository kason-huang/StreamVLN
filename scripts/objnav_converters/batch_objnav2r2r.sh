#!/bin/bash
# Batch convert objectnav episodes to R2R annotation format
# Usage: bash scripts/objnav_converters/batch_objnav2r2r.sh

INPUT_DIR="data/trajectory_data/objectnav/hm3d_v2/train/merged"
OUTPUT_DIR="data/trajectory_data/objectnav/hm3d_v2_annotation"

mkdir -p "$OUTPUT_DIR"

# Count stats
total=0
skipped=0
processed=0
failed=0

# Get total count
total=$(find "$INPUT_DIR" -name "*.json.gz" | wc -l)
echo "========================================"
echo "Batch Conversion: objectnav -> R2R"
echo "========================================"
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Total files:      $total"
echo "========================================"
echo ""

# Process each .json.gz file
current=0
for input_file in "$INPUT_DIR"/*.json.gz; do
    current=$((current + 1))
    scene_name=$(basename "$input_file" .json.gz)
    annotation_dir="$OUTPUT_DIR/$scene_name"
    annotation_file="$annotation_dir/annotations.json"

    # Check if already processed
    if [ -d "$annotation_dir" ] && [ -f "$annotation_file" ] && [ -s "$annotation_file" ]; then
        echo "[$current/$total] ⊘ Skipping $scene_name (already processed)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[$current/$total] → Processing $scene_name"

    # Run conversion
    if python scripts/objnav_converters/objnav2r2r.py \
        --input "$input_file" \
        --output "$annotation_file"; then
        echo "  ✓ Success"
        processed=$((processed + 1))
    else
        echo "  ✗ Failed"
        failed=$((failed + 1))
    fi
    echo ""
done

# Summary
echo "========================================"
echo "Batch Conversion Summary"
echo "========================================"
echo "Total files:       $total"
echo "Skipped:           $skipped"
echo "Processed:         $processed"
echo "Failed:            $failed"
echo "========================================"

if [ $failed -gt 0 ]; then
    echo "⚠ Warning: $failed scene(s) failed to convert"
    exit 1
else
    echo "✓ Batch conversion completed!"
fi
