#!/usr/bin/env python3
"""
Base evaluator for flow prediction models.
Refactored from the original evaluation.py to be more modular.
"""

import contextlib
import warnings
from pathlib import Path

# Add project root to path
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from .metrics import FlowMetrics  # noqa: E402
from .time_series_monitor import TimeSeriesMonitor  # noqa: E402
from .utils import create_output_directory  # noqa: E402
from .video_creation import VideoCreator  # noqa: E402
from .visualization import FlowVisualizer  # noqa: E402

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("W&B not available", stacklevel=2)


class BaseFlowEvaluator:
    """Base evaluator for flow prediction models."""

    def __init__(
        self,
        checkpoint_path: str,
        model_config: DictConfig,
        save_predictions: bool = False,
        monitor_points: list = None,
        output_base_dir: str = "evaluation_outputs",
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            model_config: Model configuration
            save_predictions: Whether to save prediction results
            monitor_points: List of (z_idx, x_idx) tuples for time series monitoring
            output_base_dir: Base directory for all outputs
        """
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config
        self.save_predictions = save_predictions

        # Create output directories - use runs directory from checkpoint path if available
        if "runs/" in checkpoint_path:
            # Extract runs directory path from checkpoint
            # e.g., "/path/logs/flow_swin_2d/runs/2025-09-12_23-16-43-463283/checkpoints/step_30000.ckpt"
            # -> "/path/logs/flow_swin_2d/runs/2025-09-12_23-16-43-463283"
            checkpoint_dir = Path(
                checkpoint_path
            ).parent.parent  # Remove "/checkpoints/step_30000.ckpt"
            self.output_dir = checkpoint_dir / "evaluation_results"
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            # Fallback to custom output directory
            self.output_dir = create_output_directory(output_base_dir)

        self.predictions_dir = (
            self.output_dir / "predictions" if save_predictions else None
        )

        # Initialize components
        self.time_monitor = TimeSeriesMonitor(monitor_points)
        self.metrics_calculator = FlowMetrics()
        self.visualizer = FlowVisualizer(self.output_dir)
        self.video_creator = VideoCreator(self.output_dir)

        # Model and data components
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.wandb_run = None

        # Initialize wandb
        self._init_wandb()

        print("BaseFlowEvaluator initialized")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Save predictions: {save_predictions}")
        print(f"  Monitor points: {len(self.time_monitor.monitor_points)}")

    def _init_wandb(self):
        """Initialize wandb if available."""
        if not WANDB_AVAILABLE:
            self.wandb_run = None
            return

        try:
            self.wandb_run = wandb.init(
                project="flow-evaluation",
                name=f"eval_{Path(self.checkpoint_path).stem}",
                job_type="evaluation",
            )
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            self.wandb_run = None

    def load_model_and_datasets(self):
        """Load model and datasets. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_model_and_datasets")

    def update_monitor_points(self, points: list):
        """Update monitoring points for time series tracking."""
        self.time_monitor.set_monitor_points(points)
        print(f"Updated {len(points)} monitoring points")

    def record_timestep_data(
        self,
        pred_data: torch.Tensor,
        split: str,
        mode: str,
        timestep: int,
        gt_data: torch.Tensor = None,
    ):
        """Record prediction and ground truth data for time series monitoring."""
        self.time_monitor.record_timestep(pred_data, split, mode, timestep, gt_data)

    def evaluate_sample(self, sample_idx: int = 0, split: str = "test"):
        """Evaluate a single sample. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement evaluate_sample")

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("Starting full evaluation...")

        # Load model and datasets
        self.load_model_and_datasets()

        # Run evaluations on different splits
        for split in ["test"]:  # Can extend to ["train", "val", "test"]
            print(f"\n{'=' * 50}")
            print(f"Evaluating on {split} set")
            print(f"{'=' * 50}")

            # Evaluate a few representative samples
            num_samples = min(5, len(getattr(self, f"{split}_dataset")))

            for sample_idx in range(num_samples):
                print(f"\nEvaluating sample {sample_idx}...")
                self.evaluate_sample(sample_idx, split)

        # Generate time series plots
        print("\nGenerating time series plots...")
        self.time_monitor.generate_all_plots(self.output_dir)

        # Save time series data
        print("Saving time series data...")
        self.time_monitor.save_data_csv(self.output_dir)

        print(f"\n✅ Evaluation completed! Results saved to: {self.output_dir}")

    def create_videos(self, sample_idx: int = 0, num_future: int = 30):
        """Create video visualizations."""
        return self.video_creator.create_prediction_video(self, sample_idx, num_future)

    def close_wandb(self):
        """Close wandb run."""
        if self.wandb_run is not None:
            wandb.finish()
            print("Wandb run closed.")

    def __del__(self):
        """Cleanup when evaluator is destroyed."""
        with contextlib.suppress(Exception):
            self.close_wandb()
