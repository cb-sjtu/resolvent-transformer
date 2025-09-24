#!/usr/bin/env python3
"""
3-Plane Flow Evaluator implementation with modular design.

This extends the modular evaluation architecture to support 3-plane 12-channel models.
Inherits from BaseFlowEvaluator and adds 3-plane specific functionality.
"""

import warnings
from pathlib import Path

import h5py
import numpy as np

# Add project root to path
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from .base_evaluator import BaseFlowEvaluator  # noqa: E402
from .utils import ensure_numpy_array, log_image_to_wandb  # noqa: E402

try:
    import importlib.util

    if importlib.util.find_spec("wandb") is not None:
        import wandb  # noqa: F401

        WANDB_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("W&B not available", stacklevel=2)


class Flow3PlaneEvaluator(BaseFlowEvaluator):
    """
    3-Plane Flow Model Evaluator with modular design.

    This evaluator provides:
    - 3-plane specific visualization (12-channel support)
    - Proper WandB integration with step handling
    - Multi-plane analysis
    - Video generation for 3-plane data
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_config: DictConfig = None,
        save_predictions: bool = False,
        monitor_points: list = None,
        output_base_dir: str = "evaluation_3plane_outputs",
    ):
        """
        Initialize 3-plane evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            model_config: Model configuration (will be loaded if None)
            save_predictions: Whether to save prediction results
            monitor_points: Points to monitor for time series analysis
            output_base_dir: Base directory for outputs
        """
        # Initialize parent class
        super().__init__(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            save_predictions=save_predictions,
            monitor_points=monitor_points,
            output_base_dir=output_base_dir,
        )

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 3-plane specific configuration
        self.num_planes = 3
        self.num_fields_per_plane = 4  # u, v, w, p
        self.total_channels = self.num_planes * self.num_fields_per_plane  # 12
        self.plane_y_positions = [29, 54, 75]  # y-slice positions
        self.field_names = ["u", "v", "w", "p"]

        # Flow-specific configuration (matching 2D version)
        self.channel_names = self.field_names
        self.input_length = 5  # Will be updated from dataset

        # Override time monitor with 3-plane specific one (use "p" instead of "mag")
        self.time_monitor = self._create_3plane_time_monitor(monitor_points)

        print("Initialized 3-plane evaluator:")
        print(f"  - Planes: {self.num_planes}")
        print(f"  - Fields per plane: {self.num_fields_per_plane}")
        print(f"  - Total channels: {self.total_channels}")
        print(f"  - Y positions: {self.plane_y_positions}")

    def load_model_and_datasets(self):
        """Load model and datasets for 3-plane evaluation."""
        print("Loading 3-plane model and datasets...")

        # Load model using the same method as original evaluation.py
        self.model = self._load_model()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

        # Load datasets
        print("Loading 3-plane datasets...")
        self._load_datasets()

        # Update input length from dataset
        if hasattr(self.test_dataset, "input_length"):
            self.input_length = self.test_dataset.input_length
            print(f"Input length: {self.input_length}")

    def _load_model(self):
        """Load the model from checkpoint (adapted from original evaluation.py)."""
        print(f"Loading 3-plane model from {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

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
            model = FlowSwin2DLitModule.load_from_checkpoint(self.checkpoint_path, map_location="cpu")
        else:
            # Fallback: create module with current config
            print("No hyperparameters found, using current config...")
            # Create a minimal config for the module
            from omegaconf import OmegaConf

            # Create a config that includes the model
            module_cfg = OmegaConf.create({"model": self.model_config, "loss_fn": "mse"})
            model = FlowSwin2DLitModule(module_cfg)

            # Load the state dict manually
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                print("Model weights loaded successfully!")

        model.eval()  # Set to evaluation mode
        return model

    def _load_datasets(self):
        """Load 3-plane specific datasets (adapted from original evaluation.py)."""
        from src.datasets.flow_sequence_2d.flow_sequence_3plane import FlowSequence3PlaneDataset

        data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"

        # Use the same parameters as 3-plane training configuration with max_k_steps for GT comparison
        common_params = {
            "data_dir": data_dir,
            "input_length": 5,  # Match 3-plane training
            "max_k_steps": 100,  # Load multiple GT steps for comparison
            "field_names": self.field_names,  # ["u", "v", "w", "p"]
            "file_pattern": "*u-v-w-p_scale4-6-1_yslice*.h5",
            "resolution_scale": [4, 6, 1],
            "y_slices": self.plane_y_positions,  # [29, 54, 75]
            "train_ratio": 0.7,
            "valid_ratio": 0.15,
            "test_ratio": 0.15,
            "norm_stats": "norm_stats_12ch_3plane_u-v-w-p_scale4-6-1.json",
            "enable_normalization": True,
        }

        # Load datasets with specific splits
        print("Setting up 3-plane test dataset...")
        self.test_dataset = FlowSequence3PlaneDataset(**common_params, split="test")
        print(f"Test dataset loaded with {len(self.test_dataset)} samples")

        print("Setting up 3-plane validation dataset...")
        self.val_dataset = FlowSequence3PlaneDataset(**common_params, split="val")
        print(f"Validation dataset loaded with {len(self.val_dataset)} samples")

        print("Setting up 3-plane training dataset...")
        self.train_dataset = FlowSequence3PlaneDataset(**common_params, split="train")
        print(f"Training dataset loaded with {len(self.train_dataset)} samples")

        print(
            f"Dataset sizes - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )
        print(f"Channel info: {self.test_dataset.get_channel_info()['num_total_channels']} total channels")

    def evaluate_3plane_sample(self, sample_idx: int, split: str = "test", num_future: int = 10):
        """
        Evaluate a single sample with 3-plane specific analysis (adapted from 2D evaluator).

        Args:
            sample_idx: Index of sample to evaluate
            split: Dataset split ("train", "val", "test")
            num_future: Number of future steps to predict
        """
        print(f"Evaluating 3-plane sample {sample_idx} from {split} set...")

        # Get dataset
        dataset = getattr(self, f"{split}_dataset")
        if sample_idx >= len(dataset):
            print(f"Warning: Sample {sample_idx} out of range for {split} set")
            return

        # Get sample data
        sample = dataset[sample_idx]
        input_seq = sample["data"]["input_seq"]  # Could be [1, T, C, H, W] or [T, C, H, W]

        # Handle batch dimension properly
        if len(input_seq.shape) == 5:  # [1, T, C, H, W]
            input_seq = input_seq[0]  # Remove first dim: [T, C, H, W]
        elif len(input_seq.shape) == 6:  # [1, 1, T, C, H, W]
            input_seq = input_seq[0, 0]  # Remove first two dims: [T, C, H, W]

        # Add batch dimension
        input_seq = input_seq.unsqueeze(0).to(self.device)  # [1, T, C, H, W]

        # Get ground truth sequence if available
        ground_truth_frames = []
        if "label" in sample:
            target_seq = sample["label"]
            # Handle target sequence shape
            if len(target_seq.shape) == 5:  # [1, num_future, C, H, W]
                target_seq = target_seq[0]  # [num_future, C, H, W]

            available_gt_steps = min(num_future, target_seq.shape[0])
            ground_truth_frames = [target_seq[i] for i in range(available_gt_steps)]
            print(f"  Available ground truth steps: {available_gt_steps}")

        print(f"  Input sequence shape: {input_seq.shape}")
        print(f"  Predicting {num_future} future steps...")

        # Run evaluations (similar to 2D evaluator)
        self._evaluate_3plane_autoregressive(input_seq, ground_truth_frames, sample_idx, split, num_future)
        self._evaluate_3plane_teacher_forcing(input_seq, ground_truth_frames, sample_idx, split, num_future)

    def _evaluate_3plane_autoregressive(self, input_seq, ground_truth_frames, sample_idx, split, num_future):
        """Evaluate using autoregressive prediction (adapted from 2D evaluator)."""
        print("  🔄 3-Plane Autoregressive evaluation...")

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
                pred_frame_normalized = next_pred[0]  # Remove batch dimension: (C, H, W)
                pred_frame_denorm = dataset.denormalize(pred_frame_normalized.unsqueeze(0))[0].cpu()  # (C, H, W)
                predictions.append(pred_frame_denorm)

                # Record for time series monitoring (use denormalized data)
                gt_frame = ground_truth_frames[step] if step < len(ground_truth_frames) else None
                gt_frame_denorm = None
                if gt_frame is not None:
                    gt_frame_denorm = dataset.denormalize(gt_frame.unsqueeze(0))[0].cpu()  # (C, H, W)

                self.record_timestep_data(pred_frame_denorm, split, "ar", step, gt_frame_denorm)

                # Update sequence for next prediction
                next_pred_with_time = next_pred.unsqueeze(1)  # Add time dimension
                current_seq = torch.cat([current_seq[:, 1:], next_pred_with_time], dim=1)

        # Create 3-plane visualization for first prediction
        if predictions:
            predictions_tensor = torch.stack(predictions[: min(10, len(predictions))], dim=0)  # Limit to 10 steps
            targets_tensor = (
                torch.stack(ground_truth_frames[: min(10, len(ground_truth_frames))], dim=0)
                if ground_truth_frames
                else None
            )

            self._create_3plane_visualization(
                predictions=predictions_tensor,
                targets=targets_tensor,
                sample_idx=sample_idx,
                split=split,
                num_future=len(predictions_tensor),
            )

    def _evaluate_3plane_teacher_forcing(self, input_seq, ground_truth_frames, sample_idx, split, num_future):
        """Evaluate using teacher forcing (adapted from 2D evaluator)."""
        print("  📖 3-Plane Teacher forcing evaluation...")

        # Get the appropriate dataset for denormalization
        dataset = getattr(self, f"{split}_dataset")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for step in range(min(num_future, len(ground_truth_frames))):
                # Teacher forcing: use ground truth as input (except for prediction)
                next_pred = self.model(input_seq)

                # Handle output shape
                if len(next_pred.shape) == 5:  # (B, T, C, H, W)
                    next_pred = next_pred[:, -1]  # Take last timestep
                elif len(next_pred.shape) == 4:  # (B, C, H, W)
                    pass  # Already correct
                else:
                    raise ValueError(f"Unexpected prediction shape: {next_pred.shape}")

                # Store prediction (denormalized)
                pred_frame_normalized = next_pred[0]  # Remove batch dimension: (C, H, W)
                pred_frame_denorm = dataset.denormalize(pred_frame_normalized.unsqueeze(0))[0].cpu()  # (C, H, W)
                predictions.append(pred_frame_denorm)

                # Get ground truth for this step
                gt_frame_denorm = None
                if step < len(ground_truth_frames):
                    gt_frame = ground_truth_frames[step]
                    gt_frame_denorm = dataset.denormalize(gt_frame.unsqueeze(0))[0].cpu()

                # Record for time series monitoring
                self.record_timestep_data(pred_frame_denorm, split, "tf", step, gt_frame_denorm)

                # Update input with ground truth for next prediction (teacher forcing)
                if step + 1 < len(ground_truth_frames):
                    gt_frame_next = (
                        ground_truth_frames[step].to(self.device).unsqueeze(0).unsqueeze(1)
                    )  # (1, 1, C, H, W)
                    input_seq = torch.cat([input_seq[:, 1:], gt_frame_next], dim=1)

    def create_time_series_summary(self):
        """Create comprehensive time series analysis (adapted from 2D evaluator)."""
        print("\n📈 Creating 3-plane time series analysis...")

        # Generate all plots
        plots_dir = self.time_monitor.generate_all_plots(self.output_dir)

        # Save data as CSV
        csv_dir = self.time_monitor.save_data_csv(self.output_dir)

        # Create summary report
        self._create_3plane_monitoring_report()

        return plots_dir, csv_dir

    def _create_3plane_monitoring_report(self):
        """Create a comprehensive 3-plane monitoring report."""
        report_path = self.output_dir / "3plane_time_series_report.md"

        with open(report_path, "w") as f:
            f.write("# 3-Plane Time Series Monitoring Report\n\n")
            f.write("## Monitoring Configuration\n\n")
            f.write(f"- **Monitor Points**: {len(self.time_monitor.monitor_points)}\n")
            f.write(f"- **Point Locations (z, x)**: {self.time_monitor.monitor_points}\n")
            f.write(f"- **Field Names**: {self.field_names}\n")
            f.write(f"- **Number of Planes**: {self.num_planes}\n")
            f.write(f"- **Y-slice Positions**: {self.plane_y_positions}\n")
            f.write(f"- **Total Channels**: {self.total_channels}\n\n")

            f.write("## Generated Outputs\n\n")
            f.write("- **Individual Point Plots**: `time_series_plots/time_series_point_*.png`\n")
            f.write("- **Component Overview Plots**: `time_series_plots/time_series_all_points_*.png`\n")
            f.write("- **Time Series Data**: `time_series_data/*.csv`\n\n")

            f.write("## 3-Plane Data Format\n\n")
            f.write("Each CSV file contains columns:\n")
            f.write("- `timestep`: Time step number\n")

            # Generate column descriptions for all channels
            for i, (point_plane_idx, z, x) in enumerate(self.time_monitor.monitor_points):
                y_pos = [29, 54, 75][point_plane_idx]  # Y-slice positions
                for field_name in self.field_names:
                    f.write(
                        f"- `{field_name}_pred_point{i}_plane{point_plane_idx}_y{y_pos}_z{z}_x{x}`: "
                        f"{field_name.upper()} prediction at plane {point_plane_idx} (y={y_pos}) point ({z}, {x})\n"
                    )
                    f.write(
                        f"- `{field_name}_gt_point{i}_plane{point_plane_idx}_y{y_pos}_z{z}_x{x}`: "
                        f"{field_name.upper()} ground truth at plane {point_plane_idx} (y={y_pos}) point ({z}, {x})\n"
                    )

        print(f"📄 3-Plane monitoring report saved: {report_path}")

    def _create_3plane_time_monitor(self, monitor_points):
        """Create a 3-plane specific time series monitor that uses 'p' instead of 'mag'."""
        import matplotlib.pyplot as plt
        import numpy as np

        from evaluation_modules.time_series_monitor import TimeSeriesMonitor

        # Create a custom monitor for 3-plane data
        monitor = TimeSeriesMonitor(monitor_points)

        # Monkey-patch the monitor to use "p" instead of "mag"
        # original_record_timestep = monitor.record_timestep  # (unused)
        # original_extract_point_values = monitor.extract_point_values  # (unused)

        def patched_record_timestep(pred_data, split, mode, timestep, gt_data=None):
            # Only record data from the first sample (when timesteps array is small)
            max_recorded_steps = len(monitor.time_series_data[split][mode]["timesteps"])
            max_steps_per_sample = 100  # Allow up to 100 steps per sample
            if max_recorded_steps >= max_steps_per_sample:
                print(f"    ⏭️ Skipping timestep {timestep} (already have {max_recorded_steps} steps)")
                return

            # Extract prediction values
            pred_values = patched_extract_point_values(pred_data, timestep)

            # Record timestep if not already recorded for this mode
            if timestep not in monitor.time_series_data[split][mode]["timesteps"]:
                monitor.time_series_data[split][mode]["timesteps"].append(timestep)

            # Record prediction values for each point (use "p" instead of "mag")
            for component in ["u", "v", "w", "p"]:
                for i, value in enumerate(pred_values[component]):
                    monitor.time_series_data[split][mode][f"{component}_pred"][i].append(value)
                    # Debug: Log the first few values (only from first sample)
                    if timestep < 3 and i == 0:  # Only log first point for first few timesteps
                        print(f"    🔍 Recorded {split}-{mode} t={timestep} {component}_pred[{i}] = {value:.6f}")

            # Record ground truth values if available
            if gt_data is not None:
                gt_values = patched_extract_point_values(gt_data, timestep)
                for component in ["u", "v", "w", "p"]:
                    for i, value in enumerate(gt_values[component]):
                        monitor.time_series_data[split][mode][f"{component}_gt"][i].append(value)
            else:
                # Fill with None if no ground truth available
                for component in ["u", "v", "w", "p"]:
                    for i in range(len(monitor.monitor_points)):
                        monitor.time_series_data[split][mode][f"{component}_gt"][i].append(None)

        def patched_extract_point_values(flow_data, timestep=0):
            """Extract flow field values at monitoring points for 3-plane data."""
            print(
                f"    🔍 Extract_point_values: data shape={flow_data.shape}, "
                f"data range=[{flow_data.min():.6f}, {flow_data.max():.6f}]"
            )

            # Extract u, v, w, p from 12-channel 3-plane data
            # Channels 0-2: u for planes 0,1,2; Channels 3-5: v; Channels 6-8: w; Channels 9-11: p

            point_values = {"u": [], "v": [], "w": [], "p": []}

            for plane_idx, z_idx, x_idx in monitor.monitor_points:
                # Extract values from the specific plane
                # Channel mapping: u=plane_idx+0, v=plane_idx+3, w=plane_idx+6, p=plane_idx+9
                u_channel = plane_idx + 0  # 0, 1, 2
                v_channel = plane_idx + 3  # 3, 4, 5
                w_channel = plane_idx + 6  # 6, 7, 8
                p_channel = plane_idx + 9  # 9, 10, 11

                u_val = flow_data[u_channel, z_idx, x_idx].item()
                v_val = flow_data[v_channel, z_idx, x_idx].item()
                w_val = flow_data[w_channel, z_idx, x_idx].item()
                p_val = flow_data[p_channel, z_idx, x_idx].item()

                point_values["u"].append(u_val)
                point_values["v"].append(v_val)
                point_values["w"].append(w_val)
                point_values["p"].append(p_val)

            return point_values

        # Apply the patches
        monitor.record_timestep = patched_record_timestep
        monitor.extract_point_values = patched_extract_point_values

        # Also patch the reset_data method to use "p" instead of "mag"
        # original_reset_data = monitor.reset_data  # (unused)

        def patched_reset_data():
            """Reset all monitoring data for 3-plane."""
            monitor.time_series_data = {}
            for split in ["train", "val", "test"]:
                monitor.time_series_data[split] = {}
                for mode in ["ar", "tf"]:
                    monitor.time_series_data[split][mode] = {
                        "timesteps": [],
                    }
                    # Initialize data storage for each component and point (use "p" instead of "mag")
                    for component in ["u", "v", "w", "p"]:
                        monitor.time_series_data[split][mode][f"{component}_pred"] = [
                            [] for _ in range(len(monitor.monitor_points))
                        ]
                        monitor.time_series_data[split][mode][f"{component}_gt"] = [
                            [] for _ in range(len(monitor.monitor_points))
                        ]

        monitor.reset_data = patched_reset_data
        monitor.reset_data()  # Initialize with 3-plane structure

        # Also patch plotting methods to use "p" instead of "mag"
        # original_plot_point_time_series = monitor.plot_point_time_series  # (unused)

        def patched_plot_point_time_series(point_idx, output_dir, split="test"):
            """Plot time series for a specific point (3-plane version with 'p')."""
            plane_idx, z_idx, x_idx = monitor.monitor_points[point_idx]
            y_pos = [29, 54, 75][plane_idx]  # Y-slice positions for planes 0, 1, 2
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Time Series at Plane {plane_idx} (y={y_pos}), Point ({z_idx}, {x_idx}) - {split.upper()} Data",
                fontsize=16,
            )

            components = ["u", "v", "w", "p"]  # Use "p" instead of "mag"
            colors = {"ar": "blue", "tf": "red"}
            labels = {"ar": "Autoregressive", "tf": "Teacher Forcing"}

            for i, component in enumerate(components):
                ax = axes[i // 2, i % 2]
                ax.set_title(f"Component: {component.upper()}")
                ax.set_xlabel("Timestep")
                ax.set_ylabel(f"{component.upper()} Value")

                for mode in ["ar", "tf"]:
                    if split in monitor.time_series_data and mode in monitor.time_series_data[split]:
                        timesteps = monitor.time_series_data[split][mode]["timesteps"]
                        pred_values = monitor.time_series_data[split][mode][f"{component}_pred"][point_idx]
                        gt_values = monitor.time_series_data[split][mode][f"{component}_gt"][point_idx]

                        print(
                            f"    🎨 Plotting {component}_pred: timesteps={len(timesteps)}, "
                            f"pred_values={len(pred_values)}"
                        )
                        if pred_values and len(pred_values) > 0:
                            print(f"    🎨 Pred values sample: {pred_values[:3]}")

                        # Plot predictions
                        if timesteps and pred_values and len(timesteps) == len(pred_values):
                            ax.plot(
                                timesteps,
                                pred_values,
                                color=colors[mode],
                                label=f"{labels[mode]} Pred",
                                linestyle="-",
                                marker="o",
                                markersize=4,
                            )
                            print(f"    ✅ Plotted {len(pred_values)} prediction points for {component}")
                        else:
                            print(
                                f"    ❌ Cannot plot {component}_pred: timesteps={len(timesteps)}, "
                                f"pred_values={len(pred_values) if pred_values else 0}"
                            )

                        # Plot ground truth
                        if gt_values and any(v is not None for v in gt_values):
                            valid_gt = [(t, gt) for t, gt in zip(timesteps, gt_values, strict=False) if gt is not None]
                            if valid_gt:
                                gt_times, gt_vals = zip(*valid_gt, strict=False)
                                ax.plot(
                                    gt_times,
                                    gt_vals,
                                    color=colors[mode],
                                    label=f"{labels[mode]} GT",
                                    linestyle="--",
                                    marker="s",
                                    markersize=4,
                                )

                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = (
                output_dir / f"time_series_point_{point_idx}_plane{plane_idx}_y{y_pos}_z{z_idx}_x{x_idx}_{split}.png"
            )
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Time series plot saved: {output_path}")

        monitor.plot_point_time_series = patched_plot_point_time_series

        # Patch the plot_all_points_component method
        # original_plot_all_points_component = monitor.plot_all_points_component  # (unused)

        def patched_plot_all_points_component(component, output_dir, split="test", mode="ar"):
            """Plot all points for a specific component (3-plane version with 'p')."""
            if component not in ["u", "v", "w", "p"]:
                raise ValueError(f"Component must be one of ['u', 'v', 'w', 'p'], got {component}")

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.set_title(f"All Points - {component.upper()} Component ({split.upper()}, {mode.upper()})")
            ax.set_xlabel("Timestep")
            ax.set_ylabel(f"{component.upper()} Value")

            colors = plt.cm.tab10(np.linspace(0, 1, len(monitor.monitor_points)))

            for i, (plane_idx, z_idx, x_idx) in enumerate(monitor.monitor_points):
                # y_pos = [29, 54, 75][plane_idx]  # Y-slice positions for planes 0, 1, 2 (unused)
                if split in monitor.time_series_data and mode in monitor.time_series_data[split]:
                    timesteps = monitor.time_series_data[split][mode]["timesteps"]
                    pred_values = monitor.time_series_data[split][mode][f"{component}_pred"][i]
                    gt_values = monitor.time_series_data[split][mode][f"{component}_gt"][i]

                    # Plot predictions
                    if timesteps and pred_values and len(timesteps) == len(pred_values):
                        ax.plot(
                            timesteps,
                            pred_values,
                            color=colors[i],
                            label=f"P{plane_idx}({z_idx},{x_idx}) Pred",
                            linestyle="-",
                            marker="o",
                            markersize=2,
                            alpha=0.8,
                        )

                    # Plot ground truth
                    if gt_values and any(v is not None for v in gt_values):
                        valid_gt = [(t, gt) for t, gt in zip(timesteps, gt_values, strict=False) if gt is not None]
                        if valid_gt:
                            gt_times, gt_vals = zip(*valid_gt, strict=False)
                            ax.plot(
                                gt_times,
                                gt_vals,
                                color=colors[i],
                                label=f"P{plane_idx}({z_idx},{x_idx}) GT",
                                linestyle="--",
                                marker="s",
                                markersize=2,
                                alpha=0.6,
                            )

            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = output_dir / f"time_series_all_points_{component}_{split}_{mode}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"All points {component} time series plot saved: {output_path}")

        monitor.plot_all_points_component = patched_plot_all_points_component

        # Patch the generate_all_plots method
        # original_generate_all_plots = monitor.generate_all_plots  # (unused)

        def patched_generate_all_plots(output_dir, split="test"):
            """Generate all time series plots (3-plane version with 'p')."""
            print(f"Generating time series plots for {len(monitor.monitor_points)} monitoring points...")

            ts_dir = Path(output_dir) / "time_series_plots"
            ts_dir.mkdir(parents=True, exist_ok=True)

            # Plot individual points
            for i in range(len(monitor.monitor_points)):
                monitor.plot_point_time_series(i, ts_dir, split)

            # Plot all points for each component (use "p" instead of "mag")
            for component in ["u", "v", "w", "p"]:
                for mode in ["ar", "tf"]:
                    monitor.plot_all_points_component(component, ts_dir, split, mode)

            print(f"All time series plots saved to: {ts_dir}")
            return str(ts_dir)

        monitor.generate_all_plots = patched_generate_all_plots

        # Patch the save_data_csv method
        # original_save_data_csv = monitor.save_data_csv  # (unused)

        def patched_save_data_csv(output_dir):
            """Save time series data as CSV files (3-plane version with 'p')."""
            data_dir = Path(output_dir) / "time_series_data"
            data_dir.mkdir(parents=True, exist_ok=True)

            for split in ["train", "val", "test"]:
                for mode in ["ar", "tf"]:
                    if split in monitor.time_series_data and mode in monitor.time_series_data[split]:
                        data_dict = {"timestep": monitor.time_series_data[split][mode]["timesteps"]}

                        # Add data for each point and component (use "p" instead of "mag")
                        num_timesteps = len(data_dict["timestep"])
                        for component in ["u", "v", "w", "p"]:
                            for i, (plane_idx, z_idx, x_idx) in enumerate(monitor.monitor_points):
                                y_pos = [29, 54, 75][plane_idx]  # Y-slice positions for planes 0, 1, 2
                                # Prediction data
                                pred_col_name = f"{component}_pred_point{i}_plane{plane_idx}_y{y_pos}_z{z_idx}_x{x_idx}"
                                pred_data = monitor.time_series_data[split][mode][f"{component}_pred"][i]
                                data_dict[pred_col_name] = pred_data + [None] * (num_timesteps - len(pred_data))

                                # Ground truth data
                                gt_col_name = f"{component}_gt_point{i}_plane{plane_idx}_y{y_pos}_z{z_idx}_x{x_idx}"
                                gt_data = monitor.time_series_data[split][mode][f"{component}_gt"][i]
                                data_dict[gt_col_name] = gt_data + [None] * (num_timesteps - len(gt_data))

                        # Save to CSV
                        if data_dict["timestep"]:  # Only save if we have data
                            import pandas as pd

                            df = pd.DataFrame(data_dict)
                            csv_path = data_dir / f"time_series_{split}_{mode}.csv"
                            df.to_csv(csv_path, index=False)
                            print(f"Time series data saved: {csv_path}")

            return str(data_dir)

        monitor.save_data_csv = patched_save_data_csv

        return monitor

    def _create_3plane_visualization(self, predictions, targets, sample_idx, split, num_future):
        """Create comprehensive 3-plane visualization."""
        import matplotlib.pyplot as plt

        # Create output directory for this sample
        sample_output_dir = self.output_dir / f"{split}_sample_{sample_idx}"
        sample_output_dir.mkdir(exist_ok=True)

        # Create multi-plane comparison plot
        fig, axes = plt.subplots(self.num_planes, self.num_fields_per_plane, figsize=(16, 12))
        fig.suptitle(f"3-Plane Prediction vs Target (Sample {sample_idx}, {split})", fontsize=16)

        for plane_idx in range(self.num_planes):
            for field_idx in range(self.num_fields_per_plane):
                ax = axes[plane_idx, field_idx]

                # Calculate channel index
                channel_idx = plane_idx * self.num_fields_per_plane + field_idx

                # Get prediction and target for last time step
                pred_field = predictions[-1, channel_idx].numpy()
                target_field = targets[-1, channel_idx].numpy()

                # Create side-by-side comparison
                combined = np.concatenate([pred_field, target_field], axis=1)

                im = ax.imshow(combined, cmap="RdBu_r", vmin=-2, vmax=2)
                ax.set_title(
                    f"Plane {plane_idx} (y={self.plane_y_positions[plane_idx]}) - {self.field_names[field_idx]}"
                )
                ax.set_xlabel("Prediction | Target")
                ax.axis("off")

                plt.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout()

        # Save plot
        plot_path = sample_output_dir / f"3plane_comparison_{sample_idx}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Log to WandB using the new strategy
        if self.wandb_run:
            log_image_to_wandb(
                self.wandb_run,
                f"3plane_prediction_sample_{sample_idx}_{split}",
                plot_path,
                f"3-plane prediction vs target for sample {sample_idx}",
            )

        plt.close()
        print(f"Saved 3-plane visualization: {plot_path}")

    def _save_3plane_predictions(self, predictions, targets, sample_idx, split):
        """Save 3-plane predictions to H5 file."""
        output_path = self.output_dir / f"{split}_sample_{sample_idx}_predictions.h5"

        with h5py.File(output_path, "w") as f:
            f.create_dataset("predictions", data=ensure_numpy_array(predictions))
            f.create_dataset("targets", data=ensure_numpy_array(targets))
            f.attrs["num_planes"] = self.num_planes
            f.attrs["num_fields_per_plane"] = self.num_fields_per_plane
            f.attrs["plane_y_positions"] = self.plane_y_positions
            f.attrs["field_names"] = [name.encode() for name in self.field_names]

        print(f"Saved 3-plane predictions: {output_path}")

    def create_comprehensive_3plane_analysis(self):
        """Create comprehensive analysis specific to 3-plane data."""
        print("Creating comprehensive 3-plane analysis...")

        # Use the metrics module to compute 3-plane specific metrics
        if hasattr(self, "metrics"):
            self.metrics.compute_3plane_specific_metrics()

        # Create cross-plane comparison visualizations
        self._create_cross_plane_analysis()

        print("3-plane analysis completed!")

    def _create_cross_plane_analysis(self):
        """Create analysis comparing data across different planes."""
        import matplotlib.pyplot as plt

        # This would analyze relationships between different y-planes
        print("Creating cross-plane analysis...")

        # Placeholder for now - would implement plane-to-plane correlation analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Cross-plane analysis\nComing soon!", ha="center", va="center", fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        plot_path = self.output_dir / "cross_plane_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Cross-plane analysis saved: {plot_path}")
