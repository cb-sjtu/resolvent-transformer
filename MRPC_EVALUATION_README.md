# MR-PC Cross-Scale Evaluation

Multi-Resolution Prediction-Correction (MR-PC) evaluation script for 1-plane Flow Swin Transformer models.

## Overview

This script implements a sophisticated cross-scale prediction strategy that alternates between:
- **Small-scale model** (time_stride=1, t-spacing): High-frequency predictions
- **Large-scale model** (time_stride=5, 5t-spacing): Low-frequency corrections

## MR-PC Strategy

### Phase 1: Warm-up (20 steps)
Build the anchor queue B using only the small-scale model:
```
Initial: x[1,2,3,4,5] (ground truth)
Step 6-25: Small-scale rolling predictions
Anchors collected at: x[5,10,15,20,25]
```

**Data Structures:**
- **S**: Small-step window (length=5, t-spacing) → feeds small-scale model
- **B**: Anchor window (length=5, 5t-spacing) → feeds large-scale model

### Phase 2: Main MR-PC Loop

Each cycle performs:
1. **Small-scale predictions** (4 steps): x[k+1], x[k+2], x[k+3], x[k+4]
2. **Small-scale candidate**: x̂[k+5]
3. **Large-scale prediction**: y[k+5] using B (5t-spaced anchors)
4. **Fusion**: x[k+5] = (1-α)·x̂[k+5] + α·y[k+5]
5. **Update queues**: S slides, B shifts with new anchor x[k+5]

**Example Timeline:**
```
Warm-up: t=1-25 (small-scale only)
Cycle 1:
  - Small: t=26,27,28,29 (direct)
  - Fusion at t=30: (1-α)·small[30] + α·large[30]
Cycle 2:
  - Small: t=31,32,33,34 (direct)
  - Fusion at t=35: (1-α)·small[35] + α·large[35]
...
```

## Usage

### Basic Usage
```bash
python evaluation_1plane_cross_scale.py
```

### Custom Parameters
```bash
python evaluation_1plane_cross_scale.py \
    --small_scale_checkpoint /path/to/small/model.ckpt \
    --large_scale_checkpoint /path/to/large/model.ckpt \
    --sample_idx 0 \
    --num_predictions 100 \
    --fusion_weight 0.5
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--small_scale_checkpoint` | `logs/.../2025-11-02_14-11-12-461089/checkpoints/last.ckpt` | Small-scale model (t spacing) |
| `--large_scale_checkpoint` | `logs/.../2025-11-02_23-13-52-741233/checkpoints/last.ckpt` | Large-scale model (5t spacing) |
| `--sample_idx` | 0 | Test sample index to evaluate |
| `--num_predictions` | 100 | Total number of future steps to predict |
| `--fusion_weight` | 0.5 | Fusion weight α ∈ [0,1] |

### Fusion Weight (α)

Controls the balance between small-scale and large-scale predictions:
- **α = 0.0**: Pure small-scale (no correction)
- **α = 0.3**: Mostly small-scale with light correction
- **α = 0.5**: Equal weighting (default)
- **α = 0.7**: Stronger large-scale correction
- **α = 1.0**: Pure large-scale (replaces small-scale)

**Recommended range**: 0.3 - 0.7 for optimal performance

## Output Visualizations

The script generates 4 types of visualizations in `evaluation_results/cross_scale_evaluation/`, **all comparing MR-PC with pure small-scale and pure large-scale baselines**:

### 1. Temporal Evolution (`temporal_evolution_sample_*_point_*.png`)
**9 separate plots** monitoring different spatial locations in a 3×3 grid (1/4, 1/2, 3/4 positions):

Compares u,v,w evolution at each point over time for all methods:
- **Black solid line**: Ground truth
- **Green line**: MR-PC (ours) with **red stars** at fusion points
- **Blue dashed line**: Pure small-scale baseline (autoregressive, steps by t)
- **Orange dotted line**: Pure large-scale baseline (autoregressive, steps by 5t)
- **Purple dash-dot line**: End of warm-up phase (t=20)

Each plot is labeled with its grid position (e.g., "Row 2/3, Col 3/3")

### 2. Error Analysis (`error_analysis_sample_*.png`)
Four subplots comparing all three methods:
- **MSE per field**: All three methods shown with different line styles
- **Average MSE**: Combined error across all fields (log scale)
- **RMS Relative Error**: Normalized error for each field
- **Cumulative MSE**: Shows accumulated error over time with **improvement percentages**

### 3. Spatial Comparison (`spatial_comparison_*_sample_*.png`)
2D field snapshots at selected time steps with 5 rows:
- **Row 1**: Ground truth fields
- **Row 2**: MR-PC predictions (red border at fusion points)
- **Row 3**: Pure small-scale predictions
- **Row 4**: Pure large-scale predictions
- **Row 5**: Absolute error maps (MR-PC vs GT)

### 4. Fusion Analysis (`fusion_analysis_sample_*_point_*.png`)
**9 separate plots** for each monitored point:

Detailed MR-PC fusion event analysis:
- **Top**: Timeline of fusion events and weights
- **Bottom**: Bar charts comparing small vs large vs fused values for u,v,w at each fusion point

Each plot shows fusion behavior at a specific spatial location

### 5. Energy Spectrum Comparison (`energy_spectrum_comparison_*_sample_*.png`)
**3 plots** (one per field: u, v, w) comparing spectral content:

Time-averaged 1D energy spectra comparison:
- **Left subplot**: Streamwise spectrum E(kx)
- **Right subplot**: Spanwise spectrum E(kz)

Each plot shows:
- **Black solid line**: Ground truth spectrum (if available)
- **Green line**: MR-PC spectrum
- **Blue dashed line**: Pure small-scale spectrum
- **Gray dotted line**: Kolmogorov k⁻⁵/³ reference slope

**Benefits**:
1. **Spectral fidelity**: Check if MR-PC preserves energy distribution across scales
2. **Turbulence physics**: Verify if predictions follow expected spectral decay (Kolmogorov cascade)
3. **Scale representation**: Compare how well each method captures different wavenumber ranges
4. **Energy conservation**: Assess total energy preservation

**Numerical data** also saved as `.npy` files for further analysis:
- `spectrum_mrpc_{field}_kx_sample_*.npy` and `_kz_sample_*.npy`
- `spectrum_small_{field}_kx_sample_*.npy` and `_kz_sample_*.npy`
- `spectrum_gt_{field}_kx_sample_*.npy` and `_kz_sample_*.npy`

## Model Information

### Small-Scale Model (t spacing)
- **Time stride**: 1 (consecutive frames)
- **Training data**: High temporal resolution
- **Strength**: Captures fine-scale dynamics
- **Checkpoint**: `logs/flow_swin_1plane/runs/2025-11-02_14-11-12-461089/`
- **Autoregressive behavior**: Each prediction advances by **t**, window slides by 1 frame

### Large-Scale Model (5t spacing)
- **Time stride**: 5 (every 5th frame)
- **Training data**: Low temporal resolution, larger time steps
- **Strength**: Stable long-term predictions, error correction
- **Checkpoint**: `logs/flow_swin_1plane/runs/2025-11-02_23-13-52-741233/`
- **Autoregressive behavior**: Each prediction advances by **5t**, window slides by 1 anchor position

**Important Note on Time Stepping:**
- Small-scale model: Input [t, t+1, t+2, t+3, t+4] → Predict t+5 (step size: **1×t**)
- Large-scale model: Input [t, t+5, t+10, t+15, t+20] → Predict t+25 (step size: **5×t**)
- This means pure large-scale baseline covers 100 predictions in **500t** total time
- While pure small-scale covers 100 predictions in **100t** total time
- MR-PC combines both: 100 predictions in **100t** with periodic large-scale corrections

## Architecture

Both models use Flow Swin 2D architecture:
- **Input shape**: [128, 128]
- **Sequence length**: 5 frames
- **Prediction horizon**: 1 frame
- **Channels**: 3 (u, v, w)
- **Patch size**: [4, 4]
- **Embed dim**: 128
- **Depths**: [2, 2, 4, 6, 4, 2, 2]
- **Num heads**: 8
- **Window size**: [8, 8]

## Data Configuration

- **Data directory**: `data/preprocessed_flow/`
- **Fields**: u, v, w velocity components
- **File pattern**: `*u-v-w_scale2-3-1_yslice*.h5`
- **Resolution scale**: (2, 3, 1)
- **Y-slice**: 54
- **Normalization**: Enabled with stats from `norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json`

## Key Features

1. **Intelligent Queue Management**: Maintains synchronized S (small) and B (large) windows
2. **Warm-up Phase**: Ensures B has correct 5t-spacing before MR-PC cycles
3. **Adaptive Fusion**: Combines predictions to leverage both model strengths
4. **Comprehensive Visualization**: Multiple plots to analyze performance
5. **WandB Integration**: Optional logging to Weights & Biases (if available)

## Expected Behavior

- **Warm-up phase**: 20 predictions (all small-scale)
- **MR-PC cycles**: ~16 cycles for 100 total predictions
  - Each cycle: 4 small + 1 fused = 5 predictions
  - Total from cycles: 80 predictions (64 small + 16 fused)
- **Final counts**: 84 small-scale + 16 fused = 100 total
- **Baselines**:
  - Pure small-scale: 100 autoregressive steps, each stepping by **t**
  - Pure large-scale: 100 autoregressive steps, each stepping by **5t** (after 20-step warm-up)
- **Output files**:
  - 18 temporal evolution plots (9 with all baselines + 9 without large-scale)
  - 1 error analysis plot (aggregated)
  - 3 spatial comparison plots (one per field: u, v, w)
  - 9 fusion analysis plots (one per monitored point)
  - 3 energy spectrum comparison plots (one per field: u, v, w)
  - 18 numerical spectrum data files (.npy format)

## Tips for Analysis

1. **Check warm-up phase**: All methods should be similar for first 20 steps
2. **Compare temporal evolution**:
   - Green (MR-PC) should track black (GT) better than baselines
   - Blue (small) may diverge due to error accumulation
   - Orange (large) may have larger instantaneous errors but more stable
3. **Error metrics comparison**:
   - Look at cumulative MSE improvement percentages
   - MR-PC should show lower errors than both baselines after warm-up
4. **Spatial patterns**: Compare all 5 rows to see differences in predictions
5. **Fusion impact**: Red stars/borders should coincide with error reduction
6. **Experiment with α**: Try different fusion weights (0.3-0.7) to find optimal balance
7. **Energy spectra analysis**:
   - MR-PC should preserve spectral content better than pure small-scale
   - Check if high-wavenumber energy matches ground truth (small-scale physics)
   - Check if low-wavenumber energy is stable (large-scale structures)
   - Gray k⁻⁵/³ line shows expected Kolmogorov inertial range decay

## Troubleshooting

### Display Issues
If running on headless server, set matplotlib backend:
```bash
export MPLBACKEND=Agg
python evaluation_1plane_cross_scale.py
```

### Memory Issues
Reduce `num_predictions` or process in batches:
```bash
python evaluation_1plane_cross_scale.py --num_predictions 50
```

### Checkpoint Loading Errors
Ensure model configurations match trained models. Check:
- Model architecture parameters
- Number of input channels
- Sequence length

## References

This implementation is based on the multi-resolution prediction-correction strategy for turbulent flow forecasting, combining:
- High-frequency small-scale model for detailed dynamics
- Low-frequency large-scale model for stable long-term predictions
- Weighted fusion at anchor points for optimal accuracy
