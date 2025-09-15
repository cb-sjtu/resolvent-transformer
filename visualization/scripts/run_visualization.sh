#!/bin/bash

# Flow Data Visualization Runner Script
#
# This script runs the flow data visualization with commonly used parameters.
# Modify the parameters below as needed for your specific use case.

echo "Starting Flow Data Visualization..."

# Default parameters (modify as needed)
DATA_DIR="../../data/preprocessed_flow"
OUTPUT_DIR="../videos"
MAX_FRAMES=100
FPS=10
FIELDS="u v w"
PERCENTILES="1 99"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please check the path or modify DATA_DIR in this script."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Parameters:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Max frames: $MAX_FRAMES"
echo "  FPS: $FPS"
echo "  Fields: $FIELDS"
echo "  Colorbar percentiles: $PERCENTILES"
echo ""

# Run the visualization script
python create_flow_videos.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_frames $MAX_FRAMES \
    --fps $FPS \
    --fields $FIELDS \
    --percentiles $PERCENTILES

if [ $? -eq 0 ]; then
    echo ""
    echo "Visualization completed successfully!"
    echo "Videos saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated videos:"
    ls -la "$OUTPUT_DIR"/*.mp4 2>/dev/null || ls -la "$OUTPUT_DIR"/*.gif 2>/dev/null
else
    echo ""
    echo "Visualization failed. Please check the error messages above."
    exit 1
fi
