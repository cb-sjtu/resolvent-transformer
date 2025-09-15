#!/bin/bash

echo "Installing ffmpeg for video encoding..."

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then
    echo "Running as root, installing ffmpeg..."
    apt update && apt install -y ffmpeg
elif command -v sudo &> /dev/null; then
    echo "Using sudo to install ffmpeg..."
    sudo apt update && sudo apt install -y ffmpeg
else
    echo "Cannot install ffmpeg: no sudo access"
    echo "Please run: sudo apt install ffmpeg"
    exit 1
fi

# Verify installation
if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg installed successfully!"
    ffmpeg -version | head -1
else
    echo "❌ ffmpeg installation failed"
    exit 1
fi
