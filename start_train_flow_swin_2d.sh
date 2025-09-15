#!/bin/bash

# Training script for flow_swin_2d model
# This script starts the training of the Swin Transformer 2D model for flow field prediction

echo "Starting training for flow_swin_2d model..."

# Sync dependencies with GPU support (CUDA 12.6 recommended)
echo "Syncing dependencies..."
uv sync --extra cu126

# Run the training with the flow_swin_2d configuration
echo "Starting training..."
uv run python src/train.py --config-name=train_flow_swin_2d

echo "Training completed!"
