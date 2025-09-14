#!/bin/bash

# Script to preprocess all flow data files in the background
# This creates much smaller files for faster training

set -e

INPUT_DIR="/home/sh/CB/RE550_test"
OUTPUT_DIR="/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"
FIELDS="u v w"
RESOLUTION_SCALE="2 3 1"
Y_SLICE="5"
START_FILE="t00001.h5"
LOG_FILE="/home/sh/CB/icon-thewell-dev/logs/preprocessing.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --y_slice)
            Y_SLICE="$2"
            shift 2
            ;;
        --start_file)
            START_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default pattern or use start_file pattern
if [[ -n "$START_FILE" ]]; then
    PATTERN="t[0-9][0-9][0-9][0-9][0-9].h5"
    # We'll filter files starting from START_FILE in Python script
else
    PATTERN="*.h5"
fi

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting flow data preprocessing..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Fields: $FIELDS"
echo "Resolution scale: $RESOLUTION_SCALE"
if [[ -n "$Y_SLICE" ]]; then
    echo "Y-slice: $Y_SLICE"
fi
if [[ -n "$START_FILE" ]]; then
    echo "Starting from file: $START_FILE"
fi
echo "Log file: $LOG_FILE"

# Build the command
PYTHON_CMD="nohup python scripts/preprocess_flow_data.py \
    --input_dir \"$INPUT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --fields $FIELDS \
    --resolution_scale $RESOLUTION_SCALE \
    --pattern \"$PATTERN\" \
    --create_dataset \
    --overwrite"

# Add optional parameters
if [[ -n "$Y_SLICE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --y_slice $Y_SLICE"
fi

if [[ -n "$START_FILE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --start_file $START_FILE"
fi

PYTHON_CMD="$PYTHON_CMD > \"$LOG_FILE\" 2>&1 &"

# Run preprocessing in background with logging
eval $PYTHON_CMD

PREPROCESS_PID=$!

echo "Preprocessing started in background with PID: $PREPROCESS_PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Check if still running: ps -p $PREPROCESS_PID"
echo "Stop preprocessing: kill $PREPROCESS_PID"

# Save PID for later reference
echo $PREPROCESS_PID > /tmp/preprocessing_pid.txt

echo ""
echo "Expected benefits after preprocessing:"
echo "- Multi-channel data: u, v, w fields combined into single files"
echo "- File size reduction: ~7000x smaller (2.3GB -> 0.3MB per file)"
echo "- Faster data loading: No need to downsample during training"
echo "- Reduced disk I/O bottleneck"
echo "- Better GPU utilization"
echo ""
echo "You can continue training with existing data while preprocessing runs."
