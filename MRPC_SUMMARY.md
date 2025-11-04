# MR-PC Evaluation Script - Summary

## What's New: Baseline Comparison

The updated evaluation script now includes **two baseline methods** for comprehensive comparison:

### Three Methods Compared

1. **MR-PC (Multi-Resolution Prediction-Correction)** - Our proposed method
   - Warm-up: 20 steps with small-scale model
   - Main loop: Alternates 4 small-scale + 1 fusion (small+large)
   - Total: 84 small + 16 fused = 100 predictions

2. **Pure Small-Scale Baseline**
   - 100-step autoregressive prediction using only the small-scale model (t-spacing)
   - No large-scale correction
   - Expected to accumulate errors over time

3. **Pure Large-Scale Baseline**
   - 100-step autoregressive prediction using only the large-scale model (5t-spacing)
   - Uses small-scale model for initial warm-up (first 20 steps) to build 5t-spaced anchors
   - Then switches to pure large-scale for remaining predictions
   - Expected to be more stable but with larger instantaneous errors

## Key Improvements

### Visualization Enhancements

#### 1. Temporal Evolution Plot
- **Before**: Only showed MR-PC predictions with small/fused markers
- **After**: Shows all 3 methods on the same plot
  - Green line: MR-PC with red stars at fusion points
  - Blue dashed: Pure small-scale
  - Orange dotted: Pure large-scale
  - Black solid: Ground truth
  - Purple dash-dot: Warm-up phase end

#### 2. Error Analysis Plot
- **Before**: Single method error metrics
- **After**: Comparative analysis across 4 subplots
  - **Top-left**: Per-field MSE for all methods
  - **Top-right**: Average MSE across all fields
  - **Bottom-left**: RMS relative error per field
  - **Bottom-right**: Cumulative MSE with improvement percentages

#### 3. Spatial Comparison
- **Before**: 3 rows (GT, Pred, Error)
- **After**: 5 rows showing all methods
  - Row 1: Ground truth
  - Row 2: MR-PC (red border at fusion points)
  - Row 3: Pure small-scale
  - Row 4: Pure large-scale
  - Row 5: Error (MR-PC vs GT)

## Usage Example

```bash
# Run with default settings (fusion_weight=0.5)
python evaluation_1plane_cross_scale.py

# Try different fusion weights
python evaluation_1plane_cross_scale.py --fusion_weight 0.3  # More small-scale
python evaluation_1plane_cross_scale.py --fusion_weight 0.7  # More large-scale

# Longer prediction horizon
python evaluation_1plane_cross_scale.py --num_predictions 200

# Different sample
python evaluation_1plane_cross_scale.py --sample_idx 5
```

## What to Look For

### Expected Results

1. **Temporal Evolution**:
   - Green (MR-PC) should closely follow black (GT)
   - Blue (small) may drift away from GT due to error accumulation
   - Orange (large) should be stable but may have larger deviations

2. **Error Metrics**:
   - Cumulative MSE should show: MR-PC < Pure Small < Pure Large
   - Or: MR-PC < Pure Large < Pure Small (depends on time horizon)
   - Improvement percentages should be positive

3. **Spatial Fields**:
   - MR-PC (row 2) should look most similar to GT (row 1)
   - Differences between methods more visible at later time steps
   - Fusion points (red borders) may show corrections

## Implementation Details

### Pure Small-Scale Baseline
```python
def pure_small_scale_prediction(initial_frames, num_predictions):
    """Simple autoregressive rolling prediction."""
    S = initial_frames  # [x[1], x[2], x[3], x[4], x[5]]

    for t in range(num_predictions):
        next_pred = small_scale_model(S)
        S = S[1:] + [next_pred]  # Slide window
```

### Pure Large-Scale Baseline
```python
def pure_large_scale_prediction(initial_frames, num_predictions):
    """Large-scale autoregressive after warm-up."""
    # Phase 1: Build 5t-spaced anchors using small model (20 steps)
    # Phase 2: Use large model for remaining predictions

    for t in range(num_predictions):
        next_pred = large_scale_model(B)  # B contains 5t-spaced frames
        B = B[1:] + [next_pred]  # Slide anchor window
```

## Files Modified

1. **`evaluation_1plane_cross_scale.py`**:
   - Added `pure_small_scale_prediction()` method
   - Added `pure_large_scale_prediction()` method
   - Updated `visualize_cross_scale_prediction()` to run all 3 methods
   - Modified all visualization functions to accept and display 3 predictions
   - Enhanced plots with better legends, colors, and annotations

2. **`MRPC_EVALUATION_README.md`**:
   - Updated to reflect baseline comparison features
   - Added interpretation guidelines
   - Updated expected behavior section

## Performance Notes

- **Runtime**: ~3x longer than before (runs 3 prediction sequences)
- **Memory**: Stores 3 prediction arrays instead of 1
- **Disk Space**: Same number of output images, but each image is more information-dense

## Future Enhancements

Possible additions:
1. **Adaptive fusion weight**: α = f(time, error)
2. **Multiple fusion strategies**: Different α per field (u,v,w)
3. **Statistical analysis**: t-tests, confidence intervals across multiple samples
4. **Energy spectrum analysis**: Compare predictions in Fourier space
5. **Video output**: Animated comparison of all 3 methods

## Questions?

For issues or suggestions:
- Check `MRPC_EVALUATION_README.md` for detailed usage
- See code comments in `evaluation_1plane_cross_scale.py`
- Consult error messages and traceback for debugging
