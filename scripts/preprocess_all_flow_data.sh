#!/bin/bash

# Script to preprocess all flow data files in the background
# This creates much smaller files for faster training

set -e

INPUT_DIR="/home/sh/CB/RE550_test"
OUTPUT_DIR="/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"
FIELD="u"
RESOLUTION_SCALE="2 3 1"
LOG_FILE="/home/sh/CB/icon-thewell-dev/logs/preprocessing.log"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting flow data preprocessing..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Field: $FIELD"
echo "Resolution scale: $RESOLUTION_SCALE"
echo "Log file: $LOG_FILE"

# Run preprocessing in background with logging
nohup python scripts/preprocess_flow_data.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --field "$FIELD" \
    --resolution_scale $RESOLUTION_SCALE \
    --create_dataset \
    --overwrite \
    > "$LOG_FILE" 2>&1 &

PREPROCESS_PID=$!

echo "Preprocessing started in background with PID: $PREPROCESS_PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Check if still running: ps -p $PREPROCESS_PID"
echo "Stop preprocessing: kill $PREPROCESS_PID"

# Save PID for later reference
echo $PREPROCESS_PID > /tmp/preprocessing_pid.txt

echo ""
echo "Expected benefits after preprocessing:"
echo "- File size reduction: ~7000x smaller (2.3GB -> 0.3MB per file)"
echo "- Faster data loading: No need to downsample during training"
echo "- Reduced disk I/O bottleneck"
echo "- Better GPU utilization"
echo ""
echo "You can continue training with existing data while preprocessing runs."
