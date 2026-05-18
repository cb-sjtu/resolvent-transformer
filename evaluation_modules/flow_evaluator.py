#!/usr/bin/env python3
"""
Concrete Flow Evaluator implementation with time series monitoring.

This replaces the original SimpleModelEvaluator with a more modular design
and adds comprehensive time series point monitoring functionality.
"""

import warnings

import h5py
import numpy as np

# Add project root to path
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from .base_evaluator import BaseFlowEvaluator  # noqa: E402
from .utils import ensure_numpy_array  # noqa: E402

try:
    import importlib.util

    if importlib.util.find_spec("wandb") is not None:
        WANDB_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("W&B not available", stacklevel=2)


class FlowModelEvaluator(BaseFlowEvaluator):
    """
    Comprehensive Flow Model Evaluator with time series monitoring.

    This evaluator provides:
    - Standard model evaluation (AR and TF modes)
    - Time series monitoring at configurable points
    - Comprehensive metrics calculation
    - Multi-modal visualization
    - Video generation
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_config: DictConfig,
        save_predictions: bool = False,
        monitor_points: list = None,
        output_base_dir: str = "evaluation_outputs",
    ):
        """Initialize the flow evaluator."""
        super().__init__(
            checkpoint_path,
            model_config,
            save_predictions,
            monitor_points,
            output_base_dir,
        )

        # Flow-specific configuration
        self.channel_names = ["u", "v", "w"]
        self.input_length = 5  # Will be updated from dataset

    def load_model_and_datasets(self):
        """Load model and datasets."""
        print("Loading model and datasets...")

        # Load model using the same method as original evaluation.py
        self.model = self._load_model()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

        # Load datasets
        print("Loading datasets...")
        self._load_datasets()

        # Update input length from dataset
        if hasattr(self.test_dataset, "input_length"):
            self.input_length = self.test_dataset.input_length
            print(f"Input length: {self.input_length}")

    def _load_model(self):
        """Load the model from checkpoint (copied from original evaluation.py)."""
        print(f"Loading model from {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )

        # Extract hyperparameters from checkpoint
        if "hyper_parameters" in checkpoint:
            print("Found hyperparameters in checkpoint")
        else:
            # Use default parameters based on config
            print("Using default parameters")

        # Load the full Lightning module instead of just the base model
        print("Loading full Lightning module from checkpoint...")
        from src.plmodules.flow_swin_2d_lit_module import FlowSwin2DLitModule

        # Extract hyperparameters from checkpoint to recreate the module
        if "hyper_parameters" in checkpoint:
            # Create the Lightning module with the same config
            model = FlowSwin2DLitModule.load_from_checkpoint(
                self.checkpoint_path, map_location="cpu"
            )
        else:
            # Fallback: create module with current config
            print("No hyperparameters found, using current config...")
            # Create a minimal config for the module
            from omegaconf import OmegaConf

            # Create a config that includes the model
            module_cfg = OmegaConf.create(
                {"model": self.model_config, "loss_fn": "mse"}
            )
            model = FlowSwin2DLitModule(module_cfg)

            # Load the state dict manually
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                print("Model weights loaded successfully!")

        model.eval()  # Set to evaluation mode
        return model

    def _load_datasets(self):
        """Load train/val/test datasets (copied from original evaluation.py)."""
        from src.datasets.flow_sequence_2d.fast_flow_dataset import (
            FastFlowSequence2DDataset,
        )

        data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"

        # Use the same parameters as original evaluation.py
        common_params = {
            "data_dir": data_dir,
            "input_length": 5,  # Match original evaluation.py
            "max_k_steps": 50,  # Load more ground truth steps for comparison (increased for longer predictions)
            "field_names": ["u", "v", "w"],
            "file_pattern": "*.h5",
            "resolution_scale": [2, 3, 1],
            "y_slice": 5,  # Changed from 192 to 5
            "train_ratio": 0.7,
            "valid_ratio": 0.15,
            "test_ratio": 0.15,
            "norm_stats": "norm_stats_u-v-w_scale2-3-1_yslice5.json",
            "enable_normalization": True,
        }

        # Load datasets with specific splits
        print("Setting up test dataset...")
        self.test_dataset = FastFlowSequence2DDataset(**common_params, split="test")
        print(f"Test dataset loaded with {len(self.test_dataset)} samples")

        print("Setting up train dataset...")
        self.train_dataset = FastFlowSequence2DDataset(**common_params, split="train")
        print(f"Train dataset loaded with {len(self.train_dataset)} samples")

        print("Setting up val dataset...")
        self.val_dataset = FastFlowSequence2DDataset(**common_params, split="val")
        print(f"Val dataset loaded with {len(self.val_dataset)} samples")

        print("All datasets loaded successfully!")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")

    def evaluate_sample(
        self,
        sample_idx: int = 0,
        split: str = "test",
        num_future: int = 10,
        save_h5: bool = None,
    ):
        """
        Evaluate a single sample with comprehensive analysis.

        Args:
            sample_idx: Sample index to evaluate
            split: Dataset split ('train', 'val', 'test')
            num_future: Number of future steps to predict
            save_h5: Whether to save predictions as H5 files
        """
        if save_h5 is None:
            save_h5 = self.save_predictions

        print(f"\n🔍 Evaluating sample {sample_idx} from {split} set...")

        # Get dataset
        dataset = getattr(self, f"{split}_dataset")
        if sample_idx >= len(dataset):
            print(f"Warning: Sample {sample_idx} out of range for {split} set")
            return

        # Get sample data
        sample = dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)  # (1, T, C, H, W)

        # Get ground truth sequence if available
        ground_truth_frames = []
        if "label" in sample:
            target_seq = sample["label"]  # (1, max_k_steps, C, H, W)
            print(f"  Target sequence shape: {target_seq.shape}")
            available_gt_steps = min(num_future, target_seq.shape[1])
            ground_truth_frames = [target_seq[0, i] for i in range(available_gt_steps)]
            print(f"  Available ground truth steps: {available_gt_steps}")

        print(f"  Input sequence shape: {input_seq.shape}")
        print(f"  Predicting {num_future} future steps...")

        # Run evaluations
        self._evaluate_autoregressive(
            input_seq, ground_truth_frames, sample_idx, split, num_future, save_h5
        )

        self._evaluate_teacher_forcing(
            input_seq, ground_truth_frames, sample_idx, split, num_future, save_h5
        )

        # Create visualizations
        self._create_sample_visualizations(
            input_seq, ground_truth_frames, sample_idx, split
        )

    def _evaluate_autoregressive(
        self, input_seq, ground_truth_frames, sample_idx, split, num_future, save_h5
    ):
        """Evaluate using autoregressive prediction."""
        print("  🔄 Autoregressive evaluation...")

        # Get the appropriate dataset for denormalization
        dataset = getattr(self, f"{split}_dataset")

        self.model.eval()
        predictions = []
        current_seq = input_seq.clone()

        with torch.no_grad():
            for step in range(num_future):
                # Predict next frame
                next_pred = self.model(current_seq)

                # Handle output shape
                if len(next_pred.shape) == 5:  # (B, T, C, H, W)
                    next_pred = next_pred[:, -1]  # Take last timestep
                elif len(next_pred.shape) == 4:  # (B, C, H, W)
                    pass  # Already correct
                else:
                    raise ValueError(f"Unexpected prediction shape: {next_pred.shape}")

                # Store prediction (denormalized for proper evaluation)
                pred_frame_normalized = next_pred[
                    0
                ]  # Remove batch dimension: (C, H, W)
                pred_frame_denorm = dataset.denormalize(
                    pred_frame_normalized.unsqueeze(0)
                )[0].cpu()  # Denormalize: (C, H, W)
                predictions.append(pred_frame_denorm)

                # Record for time series monitoring (use denormalized data)
                gt_frame = (
                    ground_truth_frames[step]
                    if step < len(ground_truth_frames)
                    else None
                )
                gt_frame_denorm = None
                if gt_frame is not None:
                    gt_frame_denorm = dataset.denormalize(gt_frame.unsqueeze(0))[
                        0
                    ].cpu()  # Denormalize GT: (C, H, W) and move to CPU
                # Debug: Check data before recording (only first few steps)
                if step < 3:
                    print(
                        f"    📝 Recording step {step}: pred shape={pred_frame_denorm.shape}, "
                        f"pred range=[{pred_frame_denorm.min():.6f}, {pred_frame_denorm.max():.6f}]"
                    )

                self.record_timestep_data(
                    pred_frame_denorm, split, "ar", step, gt_frame_denorm
                )

                # Update sequence for next prediction
                next_pred_with_time = next_pred.unsqueeze(1)  # Add time dimension
                current_seq = torch.cat(
                    [
                        current_seq[:, 1:],  # Remove first frame
                        next_pred_with_time,  # Add new prediction
                    ],
                    dim=1,
                )

        # Compute metrics if ground truth available (denormalize ground truth for fair comparison)
        if ground_truth_frames:
            gt_frames_denorm = []
            for gt_frame in ground_truth_frames:
                if gt_frame is not None:
                    gt_denorm = dataset.denormalize(gt_frame.unsqueeze(0))[
                        0
                    ].cpu()  # Denormalize GT
                    gt_frames_denorm.append(gt_denorm)
                else:
                    gt_frames_denorm.append(None)

            self._compute_and_log_metrics(
                predictions, gt_frames_denorm, sample_idx, split, "autoregressive"
            )

        # Save predictions if requested
        if save_h5:
            self._save_predictions_h5(predictions, sample_idx, split, "ar")

        print(
            f"    ✅ Autoregressive evaluation completed ({len(predictions)} predictions)"
        )

    def _evaluate_teacher_forcing(
        self, input_seq, ground_truth_frames, sample_idx, split, num_future, save_h5
    ):
        """Evaluate using teacher forcing (if ground truth available)."""
        if not ground_truth_frames:
            print("  ⚠️ Skipping teacher forcing (no ground truth)")
            return

        print("  📚 Teacher forcing evaluation...")

        # Get the appropriate dataset for denormalization
        dataset = getattr(self, f"{split}_dataset")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for step in range(min(num_future, len(ground_truth_frames))):
                # Use ground truth for input sequence
                if step == 0:
                    current_seq = input_seq.clone()
                else:
                    # Build sequence with ground truth
                    gt_frames = [
                        ground_truth_frames[i].unsqueeze(0) for i in range(step)
                    ]
                    if len(gt_frames) >= self.input_length:
                        # Use last input_length ground truth frames
                        current_seq = torch.stack(
                            gt_frames[-self.input_length :], dim=1
                        ).to(self.device)
                    else:
                        # Pad with original input if needed
                        needed = self.input_length - len(gt_frames)
                        padding = [input_seq[0, i].unsqueeze(0) for i in range(needed)]
                        gt_frames_tensor = [
                            frame.to(self.device) for frame in gt_frames
                        ]
                        all_frames = padding + gt_frames_tensor
                        current_seq = torch.stack(all_frames, dim=1)

                # Predict
                next_pred = self.model(current_seq)

                # Handle output shape
                if len(next_pred.shape) == 5:
                    next_pred = next_pred[:, -1]
                elif len(next_pred.shape) == 4:
                    pass
                else:
                    raise ValueError(f"Unexpected prediction shape: {next_pred.shape}")

                # Store prediction (denormalized for proper evaluation)
                pred_frame_normalized = next_pred[0]  # (C, H, W)
                pred_frame_denorm = dataset.denormalize(
                    pred_frame_normalized.unsqueeze(0)
                )[0].cpu()  # Denormalize: (C, H, W)
                predictions.append(pred_frame_denorm)

                # Record for time series monitoring (use denormalized data)
                gt_frame = (
                    ground_truth_frames[step]
                    if step < len(ground_truth_frames)
                    else None
                )
                gt_frame_denorm = None
                if gt_frame is not None:
                    gt_frame_denorm = dataset.denormalize(gt_frame.unsqueeze(0))[
                        0
                    ].cpu()  # Denormalize GT: (C, H, W) and move to CPU
                self.record_timestep_data(
                    pred_frame_denorm, split, "tf", step, gt_frame_denorm
                )

        # Compute metrics (denormalize ground truth for fair comparison)
        gt_frames_denorm = []
        for gt_frame in ground_truth_frames[: len(predictions)]:
            if gt_frame is not None:
                gt_denorm = dataset.denormalize(gt_frame.unsqueeze(0))[
                    0
                ].cpu()  # Denormalize GT
                gt_frames_denorm.append(gt_denorm)
            else:
                gt_frames_denorm.append(None)

        self._compute_and_log_metrics(
            predictions, gt_frames_denorm, sample_idx, split, "teacher_forcing"
        )

        # Save predictions if requested
        if save_h5:
            self._save_predictions_h5(predictions, sample_idx, split, "tf")

        print(
            f"    ✅ Teacher forcing evaluation completed ({len(predictions)} predictions)"
        )

    def _compute_and_log_metrics(
        self, predictions, ground_truth, sample_idx, split, mode
    ):
        """Compute and log comprehensive metrics."""
        print(f"    📊 Computing metrics for {mode}...")

        all_metrics = []

        for step, (pred, gt) in enumerate(zip(predictions, ground_truth, strict=False)):
            # Ensure same device
            pred = pred.cpu()
            gt = gt.cpu()

            # Compute comprehensive metrics
            metrics = self.metrics_calculator.compute_comprehensive_metrics(
                pred, gt, self.channel_names
            )

            # Add step info
            metrics["step"] = step
            metrics["sample_idx"] = sample_idx
            metrics["split"] = split
            metrics["mode"] = mode

            all_metrics.append(metrics)

            # Log to wandb if available
            if self.wandb_run is not None:
                self.metrics_calculator.log_metrics_to_wandb(
                    metrics, f"{split}/{mode}/sample_{sample_idx}", step
                )

        # Print summary
        if all_metrics:
            final_metrics = all_metrics[-1]
            summary = self.metrics_calculator.format_metrics_summary(final_metrics)
            print(f"    📈 Final step metrics: {summary}")

    def _create_sample_visualizations(
        self, input_seq, ground_truth_frames, sample_idx, split
    ):
        """Create visualizations for the sample."""
        print("    🎨 Creating visualizations...")

        # Get the appropriate dataset for denormalization
        dataset = getattr(self, f"{split}_dataset")

        # Single frame comparison (if ground truth available)
        if ground_truth_frames:
            # Use first prediction vs first ground truth

            # Quick prediction for visualization (both denormalized)
            with torch.no_grad():
                pred_normalized = self.model(input_seq.to(self.device))
                if len(pred_normalized.shape) == 5:
                    pred_normalized = pred_normalized[:, -1]
                pred_normalized = pred_normalized[0]  # (C, H, W)

                # Denormalize prediction for proper visualization
                pred_denorm = dataset.denormalize(pred_normalized.unsqueeze(0))[
                    0
                ].cpu()  # (C, H, W)

                # Denormalize ground truth for proper visualization
                gt_denorm = dataset.denormalize(ground_truth_frames[0].unsqueeze(0))[
                    0
                ].cpu()  # (C, H, W)

            plot_path = self.visualizer.plot_single_frame_comparison(
                pred_denorm, gt_denorm, sample_idx, 0, self.channel_names
            )

            # Log to wandb
            if self.wandb_run is not None:
                self.visualizer.log_plot_to_wandb(
                    self.wandb_run,
                    plot_path,
                    f"{split}/visualization/sample_{sample_idx}",
                    f"Prediction vs Ground Truth - Sample {sample_idx}",
                )

    def _save_predictions_h5(self, predictions, sample_idx, split, mode):
        """Save predictions as H5 files."""
        if not self.predictions_dir:
            return

        split_dir = self.predictions_dir / split / mode
        split_dir.mkdir(exist_ok=True, parents=True)

        for step, pred in enumerate(predictions):
            pred_np = ensure_numpy_array(pred)

            filename = f"pred_sample{sample_idx:05d}_step{step:05d}.h5"
            filepath = split_dir / filename

            with h5py.File(filepath, "w") as f:
                f.create_dataset("data", data=pred_np.astype(np.float32))
                f.attrs["sample_idx"] = sample_idx
                f.attrs["step"] = step
                f.attrs["split"] = split
                f.attrs["mode"] = mode
                f.attrs["field_names"] = self.channel_names
                f.attrs["num_channels"] = len(self.channel_names)

    def create_time_series_summary(self):
        """Create comprehensive time series analysis."""
        print("\n📈 Creating time series analysis...")

        # Generate all plots
        plots_dir = self.time_monitor.generate_all_plots(self.output_dir)

        # Save data as CSV
        csv_dir = self.time_monitor.save_data_csv(self.output_dir)

        # Create summary report
        self._create_monitoring_report()

        return plots_dir, csv_dir

    def _create_monitoring_report(self):
        """Create a comprehensive monitoring report."""
        report_path = self.output_dir / "time_series_report.md"

        with open(report_path, "w") as f:
            f.write("# Time Series Monitoring Report\n\n")
            f.write("## Monitoring Configuration\n\n")
            f.write(f"- **Monitor Points**: {len(self.time_monitor.monitor_points)}\n")
            f.write(
                f"- **Point Locations (z, x)**: {self.time_monitor.monitor_points}\n"
            )
            f.write(f"- **Channels**: {self.channel_names}\n\n")

            f.write("## Generated Outputs\n\n")
            f.write(
                "- **Individual Point Plots**: `time_series_plots/time_series_point_*.png`\n"
            )
            f.write(
                "- **Component Overview Plots**: `time_series_plots/time_series_all_points_*.png`\n"
            )
            f.write("- **Time Series Data**: `time_series_data/*.csv`\n\n")

            f.write("## Data Format\n\n")
            f.write("Each CSV file contains columns:\n")
            f.write("- `timestep`: Time step number\n")

            for component in ["u", "v", "w", "mag"]:
                for i, (z, x) in enumerate(self.time_monitor.monitor_points):
                    f.write(
                        f"- `{component}_point{i}_z{z}_x{x}`: {component.upper()} value at point ({z}, {x})\n"
                    )

        print(f"📄 Monitoring report saved: {report_path}")
