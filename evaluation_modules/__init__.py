"""
Evaluation modules for flow prediction models.

This package contains modular components for model evaluation including:
- Base evaluation framework
- Metrics computation
- Visualization tools
- Video creation utilities
- Time series monitoring
"""

from .base_evaluator import BaseFlowEvaluator
from .metrics import FlowMetrics
from .time_series_monitor import TimeSeriesMonitor
from .utils import (
    compute_smart_relative_error,
    compute_velocity_magnitude,
    create_output_directory,
    ensure_numpy_array,
    ensure_torch_tensor,
    format_metrics_string,
    get_default_monitor_points,
    load_numpy_as_tensor,
    log_image_to_wandb,
    save_tensor_as_numpy,
)
from .video_creation import VideoCreator
from .visualization import FlowVisualizer

__all__ = ["BaseFlowEvaluator", "TimeSeriesMonitor", "FlowMetrics", "FlowVisualizer", "VideoCreator"]
