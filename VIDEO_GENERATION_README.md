# Video Generation for MR-PC Cross-Scale Evaluation

## Overview

The `evaluation_1plane_cross_scale.py` script now supports generating videos to visualize the temporal evolution of MR-PC fusion predictions compared to small-scale baseline predictions.

## Video Outputs

When enabled, the script generates **3 video files**:

1. **`mrpc_prediction_sample_X.mp4`**: MR-PC fusion predictions showing all 3 channels (u, v, w)
   - Highlights fusion points with [FUSION] label in title
   - Shows when large-scale corrections are applied

2. **`small_prediction_sample_X.mp4`**: Pure small-scale baseline predictions
   - Same visualization format for easy comparison

3. **`comparison_sample_X.mp4`**: Side-by-side comparison (3x3 grid)
   - Row 1: Ground truth (u, v, w)
   - Row 2: MR-PC predictions (u, v, w) with fusion indicators
   - Row 3: Small-scale baseline (u, v, w)

## Usage

### Basic Usage

Run evaluation with video generation:

```bash
python evaluation_1plane_cross_scale.py --generate_videos
```

### Advanced Options

```bash
python evaluation_1plane_cross_scale.py \
    --generate_videos \
    --sample_idx 50 \
    --num_predictions 100 \
    --fusion_weight 0.8 \
    --video_fps 15
```

### Command-line Arguments

- `--generate_videos`: Enable video generation (flag, no value needed)
- `--video_fps`: Frames per second for output videos (default: 10)
  - Lower fps = slower playback, easier to observe details
  - Higher fps = faster playback, smoother animation
- `--sample_idx`: Sample index to evaluate (default: 0)
- `--num_predictions`: Number of future steps to predict (default: 100)
- `--fusion_weight`: Fusion weight α for MR-PC (default: 0.8)

## Output Location

Videos are saved to:
```
evaluation_results/cross_scale_evaluation/videos/
├── mrpc_prediction_sample_0.mp4
├── small_prediction_sample_0.mp4
└── comparison_sample_0.mp4
```

## Video Specifications

- **Resolution**: 1500x500 pixels (for individual videos), 1500x1000 pixels (for comparison)
- **Format**: MP4 (H.264 codec)
- **Color scheme**: RdBu_r colormap (diverging red-blue)
- **Color ranges**: Automatically determined from 1st-99th percentile to avoid outliers

## Features

### Fusion Point Indicators

- Fusion events are marked with **[FUSION]** in the title
- These occur at t=31, 41, 51, 61, ... (every 10 steps from t=31)
- Helps identify when large-scale corrections are applied

### Consistent Color Scaling

- All frames use the same color scale per channel
- Ensures visual consistency across time
- Makes it easier to observe magnitude changes

### Channel Organization

- **u**: Streamwise velocity
- **v**: Wall-normal velocity
- **w**: Spanwise velocity

## Requirements

Video generation requires `ffmpeg` to be installed on your system:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Or install via conda
conda install ffmpeg
```

If FFmpeg is not available, the script will automatically fall back to creating GIF files instead of MP4 videos.

## Performance Notes

- Video generation can take several minutes depending on:
  - Number of predictions (`--num_predictions`)
  - Video resolution and quality settings
  - CPU/GPU performance

- For 100 frames at 10 fps:
  - Each video is ~10 seconds long
  - Generation time: ~2-5 minutes

## Tips

1. **For detailed analysis**: Use lower fps (e.g., `--video_fps 5`) to slow down playback
2. **For smooth animation**: Use higher fps (e.g., `--video_fps 20-30`)
3. **For quick preview**: Reduce `--num_predictions` to 50 or fewer frames
4. **Best quality**: Keep default settings and use high-quality video player (VLC, mpv)

## Example Workflow

```bash
# Step 1: Run evaluation with video generation
python evaluation_1plane_cross_scale.py \
    --sample_idx 100 \
    --num_predictions 100 \
    --fusion_weight 0.8 \
    --generate_videos \
    --video_fps 10

# Step 2: View videos
# Navigate to: evaluation_results/cross_scale_evaluation/videos/
# Open with your preferred video player

# Step 3: Compare different fusion weights
python evaluation_1plane_cross_scale.py \
    --sample_idx 100 \
    --fusion_weight 0.5 \
    --generate_videos
```

## Troubleshooting

### Video files are not created or FFmpeg errors occur

- Ensure FFmpeg is properly installed: `ffmpeg -version`
- If FFmpeg is not available, the script will create GIF files instead
- Check disk space is sufficient
- Verify matplotlib is working correctly

### Videos play too fast/slow

- Adjust `--video_fps` parameter
- Lower fps = slower playback
- Most video players also allow speed adjustment

### Colors look washed out

- This is normal if the data has small variations
- The script uses percentile-based scaling (1-99%)
- You can modify `vmin`/`vmax` calculation in the code if needed

### GIF files are created instead of MP4

- This means FFmpeg is not available on your system
- Install FFmpeg following the instructions in the Requirements section
- GIF files work fine but have larger file sizes and lower quality

## Implementation Details

The video generation method (`generate_prediction_videos`) is located in the `CrossScaleEvaluator` class in [evaluation_1plane_cross_scale.py](evaluation_1plane_cross_scale.py):

- Lines 1436-1679: Main video generation method
- Uses `matplotlib.animation.FuncAnimation` for efficient video creation
- Uses `FFMpegWriter` for MP4 output (same as evaluation_1plane.py)
- Automatically falls back to GIF format if FFmpeg is not available
- Supports all visualization features including fusion point indicators
