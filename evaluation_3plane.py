#!/usr/bin/env python3
"""
Evaluation script for 3-plane 3-channel Flow Swin Transformer implementation.
Loads the best model checkpoint and generates comprehensive visualizations for all planes and channels.
Only processes u,v,w velocity fields (pressure channel removed).
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

warnings.filterwarnings("ignore")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Plots will only be saved locally.")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.datasets.flow_sequence_2d.flow_sequence_3plane import FlowSequence3PlaneDataset  # noqa: E402


class ThreePlaneModelEvaluator:
    """Evaluator for the 3-plane 3-channel Flow Swin Transformer model (u,v,w only)."""

    def __init__(self, checkpoint_path: str, model_cfg: DictConfig, save_predictions: bool = False):
        """Initialize the evaluator.

        Args:
            checkpoint_path: Path to the model checkpoint
            model_cfg: Model configuration from Hydra
            save_predictions: Whether to save prediction results as H5 files
        """
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.save_predictions = save_predictions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")
        print(f"Checkpoint path: {checkpoint_path}")

        # Initialize wandb for evaluation logging
        if WANDB_AVAILABLE:
            log_dir_name = self._extract_log_dir_name()
            training_run_id = self._extract_wandb_run_id()

            if training_run_id:
                print(f"Found training run ID: {training_run_id}")
                try:
                    # Try to resume the training run
                    self.wandb_run = wandb.init(
                        project="turbulence_swin_3plane",
                        id=training_run_id,
                        resume="allow",
                        tags=["evaluation", "flow", "swin", "3plane", "9channel", "uvw"],
                    )
                    print("Successfully resumed training wandb run for evaluation logging")
                except Exception as e:
                    print(f"Could not resume training run: {e}")
                    # Fallback: create a new linked run
                    self.wandb_run = wandb.init(
                        project="turbulence_swin_3plane",
                        name=f"evaluation_{log_dir_name}",
                        tags=["evaluation", "flow", "swin", "3plane", "9channel", "uvw"],
                        config={
                            "checkpoint_path": checkpoint_path,
                            "device": str(self.device),
                            "log_dir": log_dir_name,
                            "evaluation_type": "post_training",
                            "training_run_id": training_run_id,
                        },
                    )
            else:
                print("No training wandb run ID found, creating new evaluation run...")
                # Create new evaluation run
                self.wandb_run = wandb.init(
                    project="turbulence_swin_3plane",
                    name=f"evaluation_{log_dir_name}",
                    tags=["evaluation", "flow", "swin", "3plane", "9channel", "uvw"],
                    config={
                        "checkpoint_path": checkpoint_path,
                        "device": str(self.device),
                        "log_dir": log_dir_name,
                        "evaluation_type": "post_training",
                    },
                )
        else:
            self.wandb_run = None

        # Load model
        self.model = self._load_model()

        # Setup datasets - using the same configuration as training
        self.train_dataset, self.val_dataset, self.test_dataset = self._setup_datasets()

        # Create output directory
        self.output_dir = Path("evaluation_results") / f"evaluation_results_{self._extract_log_dir_name()}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def _load_model(self) -> nn.Module:
        """Load the model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Load the full Lightning module
        print("Loading full Lightning module from checkpoint...")
        from src.plmodules.flow_swin_2d_lit_module import FlowSwin2DLitModule

        if "hyper_parameters" in checkpoint:
            # Create the Lightning module with the same config
            model = FlowSwin2DLitModule.load_from_checkpoint(self.checkpoint_path, map_location="cpu")
        else:
            # Fallback: create module with current config
            print("No hyperparameters found, using current config...")
            from omegaconf import OmegaConf

            module_cfg = OmegaConf.create({"model": self.model_cfg, "loss_fn": "mse"})
            model = FlowSwin2DLitModule(module_cfg)

            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                print("Model weights loaded successfully!")

        model.eval()
        model.to(self.device)
        return model

    def _setup_datasets(self):
        """Setup the 3-plane datasets for evaluation."""
        print("Setting up 3-plane datasets...")

        # Dataset configuration matching training
        data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"
        field_names = ["u", "v", "w"]  # Only velocity fields (removed pressure)
        file_pattern = "*u-v-w-p_scale4-6-1_yslice*.h5"  # Still read uvwp files, but only load u,v,w
        resolution_scale = (4, 6, 1)
        y_slices = [29, 54, 75]  # 与归一化统计对应的y平面
        norm_stats_file = "norm_stats_9ch_3plane_u-v-w_scale4-6-1.json"

        # Create datasets for all splits
        train_dataset = FlowSequence3PlaneDataset(
            data_dir=data_dir,
            input_length=5,
            field_names=field_names,
            file_pattern=file_pattern,
            resolution_scale=resolution_scale,
            y_slices=y_slices,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="train",
            enable_normalization=True,
            norm_stats=norm_stats_file,
        )

        val_dataset = FlowSequence3PlaneDataset(
            data_dir=data_dir,
            input_length=5,
            field_names=field_names,
            file_pattern=file_pattern,
            resolution_scale=resolution_scale,
            y_slices=y_slices,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="val",
            enable_normalization=True,
            norm_stats=norm_stats_file,
        )

        test_dataset = FlowSequence3PlaneDataset(
            data_dir=data_dir,
            input_length=5,
            field_names=field_names,
            file_pattern=file_pattern,
            resolution_scale=resolution_scale,
            y_slices=y_slices,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="test",
            enable_normalization=True,
            norm_stats=norm_stats_file,
        )

        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"Channel info: {train_dataset.get_channel_info()['num_total_channels']} total channels")

        return train_dataset, val_dataset, test_dataset

    def _extract_log_dir_name(self) -> str:
        """Extract log directory name from checkpoint path."""
        checkpoint_path = Path(self.checkpoint_path)
        run_dir = checkpoint_path.parent.parent.name
        return run_dir

    def _extract_wandb_run_id(self) -> str:
        """Extract wandb run ID from the training logs."""
        checkpoint_path = Path(self.checkpoint_path)
        run_dir = checkpoint_path.parent.parent
        wandb_dir = run_dir / "wandb"

        if wandb_dir.exists():
            latest_run_link = wandb_dir / "latest-run"
            if latest_run_link.exists() and latest_run_link.is_symlink():
                run_name = latest_run_link.readlink().name
                if run_name.startswith("run-") and "-" in run_name:
                    return run_name.split("-")[-1]

            for item in wandb_dir.iterdir():
                if item.is_dir() and item.name.startswith("run-"):
                    run_id = item.name.split("-")[-1]
                    return run_id

        return None

    def generate_sequence_prediction(self, input_seq: torch.Tensor, num_predictions: int = 5) -> torch.Tensor:
        """Generate TRUE autoregressive sequence predictions.

        IMPORTANT: This implements TRUE autoregressive prediction, where:
        - The sliding window moves by time_stride frames (not just 1 frame)
        - Uses previous predictions as input for future predictions
        - Matches the training data's temporal spacing
        """
        self.model.eval()

        # Get time_stride from dataset
        time_stride = 1  # Default value
        try:
            if hasattr(self.test_dataset, "get_time_stride"):
                time_stride = self.test_dataset.get_time_stride()
            else:
                channel_info = self.test_dataset.get_channel_info()
                time_stride = channel_info.get("time_stride", 1)
            print(f"Using time_stride={time_stride} for autoregressive prediction")
        except Exception:
            print("Warning: Could not get time_stride, using default=1")

        with torch.no_grad():
            predictions = []
            current_input = input_seq.clone()

            for _ in range(num_predictions):
                # Predict next frame
                next_pred = self.model(current_input)  # (B, C, H, W)
                predictions.append(next_pred)

                # TRUE AUTOREGRESSIVE: Slide window by 1 frame (always remove oldest frame)
                # Model requires fixed input_length, so always remove exactly 1 frame
                # Example: [0,2,4,6,8] → predict 10, then [2,4,6,8,10] → predict 12
                current_input = torch.cat([current_input[:, 1:], next_pred.unsqueeze(1)], dim=1)

            # Stack predictions: (B, T_pred, C, H, W)
            pred_seq = torch.stack(predictions, dim=1)

        return pred_seq

    def visualize_3plane_prediction(self, sample_idx: int = 0, num_future: int = 20):
        """Visualize 3-plane 3-channel prediction with comprehensive comparison (u,v,w only)."""
        print(f"Visualizing 3-plane sample {sample_idx} with {num_future} future steps...")

        # Get sample and generate predictions
        sample = self.test_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)
        # target_seq = sample["label"]  # Ground truth targets (unused)

        # Generate predictions
        pred_seq = self.generate_sequence_prediction(input_seq, num_future)

        # Get ground truth sequence for proper comparison
        ground_truth_frames = []
        for i in range(num_future):
            if sample_idx + i < len(self.test_dataset):
                sample_i = self.test_dataset[sample_idx + i]
                target_i = sample_i["label"]  # (1, 1, C, H, W) or similar
                target_denorm = self.test_dataset.denormalize(target_i)
                target_frame = target_denorm.cpu().numpy()[0, 0]  # (C, H, W)
                ground_truth_frames.append(target_frame)
            else:
                # Repeat last frame if not enough samples
                if ground_truth_frames:
                    ground_truth_frames.append(ground_truth_frames[-1])

        # Denormalize predictions
        input_seq_denorm = self.test_dataset.denormalize(input_seq)
        pred_seq_denorm = self.test_dataset.denormalize(pred_seq)

        # Move to CPU
        input_seq = input_seq_denorm.cpu().numpy()[0]  # (T, C, H, W)
        pred_seq = pred_seq_denorm.cpu().numpy()[0]  # (T_pred, C, H, W)

        # Get channel info
        channel_info = self.test_dataset.get_channel_info()
        field_names = channel_info["field_names"]  # ["u", "v", "w"]
        y_slices = channel_info["y_slices"]  # [29, 54, 75]
        num_planes = channel_info["num_planes"]  # 3

        # Create separate visualizations for each channel (9 total: 3 planes × 3 fields)
        self._create_channel_visualizations(
            input_seq, pred_seq, ground_truth_frames, field_names, y_slices, num_planes, sample_idx, num_future
        )

        # Create comprehensive comparison visualization
        self._create_comprehensive_comparison(
            input_seq, pred_seq, ground_truth_frames, field_names, y_slices, num_planes, sample_idx, num_future
        )

        # Save detailed error analysis
        self._save_detailed_error_analysis(pred_seq, ground_truth_frames, sample_idx, field_names, y_slices)

    def _create_channel_visualizations(
        self, input_seq, pred_seq, ground_truth_frames, field_names, y_slices, num_planes, sample_idx, num_future
    ):
        """Create separate visualization for each channel with 20 steps."""
        display_steps = min(num_future, 20)

        for plane_idx in range(num_planes):
            y_slice = y_slices[plane_idx]

            for field_idx, field_name in enumerate(field_names):
                channel_idx = plane_idx * len(field_names) + field_idx

                # Create figure for this channel: 3 rows (GT, Pred, Error) × timesteps
                fig, axes = plt.subplots(3, display_steps, figsize=(2 * display_steps, 8))
                if display_steps == 1:
                    axes = axes.reshape(3, 1)

                print(f"\nChannel: Plane{plane_idx} {field_name.upper()} (y={y_slice})")
                print("Step | MSE      | MAE      | RMS-Rel Error")
                print("-" * 40)

                # Calculate channel-specific colorbar range across all timesteps
                all_data = []
                for t in range(display_steps):
                    if t < len(ground_truth_frames):
                        all_data.append(ground_truth_frames[t][channel_idx])
                    if t < pred_seq.shape[0]:
                        all_data.append(pred_seq[t][channel_idx])

                if all_data:
                    # All fields are velocity components
                    cmap = "RdBu_r"
                    vmax = max([abs(data.min()) for data in all_data] + [abs(data.max()) for data in all_data])
                    vmin = -vmax
                else:
                    cmap = "RdBu_r"
                    vmin, vmax = -1, 1

                for t in range(display_steps):
                    # Ground truth
                    if t < len(ground_truth_frames):
                        gt_data = ground_truth_frames[t][channel_idx]
                        im1 = axes[0, t].imshow(gt_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                        axes[0, t].set_title(f"GT t+{t + 1}", fontsize=8)
                    else:
                        axes[0, t].axis("off")
                        axes[0, t].set_title("GT N/A", fontsize=8)

                    # Prediction
                    if t < pred_seq.shape[0]:
                        pred_data = pred_seq[t][channel_idx]
                        im2 = axes[1, t].imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                        axes[1, t].set_title(f"Pred t+{t + 1}", fontsize=8)

                        # Error
                        if t < len(ground_truth_frames):
                            gt_data = ground_truth_frames[t][channel_idx]
                            error = np.abs(pred_data - gt_data)
                            im3 = axes[2, t].imshow(error, cmap="Reds", origin="lower")

                            # Calculate metrics
                            mse = np.mean((pred_data - gt_data) ** 2)
                            mae = np.mean(error)
                            target_rms = np.sqrt(np.mean(gt_data**2))
                            rms_rel_error = np.sqrt(mse) / (target_rms + 1e-8)

                            print(f"t+{t + 1:2d} | {mse:.6f} | {mae:.6f} | {rms_rel_error:.6f}")
                            axes[2, t].set_title(f"Error t+{t + 1}\nMAE: {mae:.4f}", fontsize=8)
                        else:
                            axes[2, t].axis("off")
                            axes[2, t].set_title("Error N/A", fontsize=8)
                    else:
                        axes[1, t].axis("off")
                        axes[1, t].set_title("Pred N/A", fontsize=8)
                        axes[2, t].axis("off")
                        axes[2, t].set_title("Error N/A", fontsize=8)

                    # Remove axis ticks for cleaner look
                    for row in range(3):
                        axes[row, t].set_xticks([])
                        axes[row, t].set_yticks([])

                    # Add colorbar for first column
                    if t == 0:
                        if t < len(ground_truth_frames):
                            plt.colorbar(im1, ax=axes[0, t], fraction=0.046, pad=0.04)
                        if t < pred_seq.shape[0]:
                            plt.colorbar(im2, ax=axes[1, t], fraction=0.046, pad=0.04)
                            if t < len(ground_truth_frames):
                                plt.colorbar(im3, ax=axes[2, t], fraction=0.046, pad=0.04)

                # Set row labels
                axes[0, 0].set_ylabel("Ground Truth", fontsize=10)
                axes[1, 0].set_ylabel("Prediction", fontsize=10)
                axes[2, 0].set_ylabel("Error", fontsize=10)

                plt.suptitle(f"Plane {plane_idx} - {field_name.upper()} (y={y_slice}) - 20 Steps", fontsize=12)
                plt.tight_layout()

                # Save individual channel visualization
                output_path = self.output_dir / f"channel_plane{plane_idx}_{field_name}_sample_{sample_idx}.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Saved channel visualization: {output_path}")

                # Log to wandb if available
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"channel_plane{plane_idx}_{field_name}_sample_{sample_idx}": wandb.Image(str(output_path))}
                    )

                plt.close()  # Close to save memory

    def _create_comprehensive_comparison(
        self, input_seq, pred_seq, ground_truth_frames, field_names, y_slices, num_planes, sample_idx, num_future
    ):
        """Create comprehensive comparison visualization showing all channels and planes."""
        # Limit display for overview
        display_steps = min(num_future, 10)

        # Create comprehensive visualization
        # Rows: 3 planes × 3 fields = 9 rows
        # Cols: timesteps
        fig, axes = plt.subplots(
            num_planes * len(field_names),
            display_steps,
            figsize=(2.5 * display_steps, 1.8 * num_planes * len(field_names)),
        )

        if display_steps == 1:
            axes = axes.reshape(-1, 1)
        for t in range(display_steps):
            for plane_idx in range(num_planes):
                y_slice = y_slices[plane_idx]

                for field_idx, field_name in enumerate(field_names):
                    row_idx = plane_idx * len(field_names) + field_idx
                    channel_idx = plane_idx * len(field_names) + field_idx

                    ax = axes[row_idx, t]

                    if t < pred_seq.shape[0]:
                        # Show predictions
                        data = pred_seq[t][channel_idx]  # (H, W)
                        title_suffix = f"Pred t+{t + 1}"
                    else:
                        # Show black if no prediction
                        data = np.zeros((pred_seq.shape[-2], pred_seq.shape[-1]))
                        title_suffix = "N/A"

                    # Determine colormap and range based on field (all are velocity components)
                    cmap = "RdBu_r"
                    if np.any(data):
                        vmax = max(abs(data.min()), abs(data.max()))
                        vmin = -vmax
                    else:
                        vmin, vmax = -1, 1

                    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

                    # Set title
                    if t == 0:
                        ax.set_title(f"{title_suffix}\nP{plane_idx} {field_name} y={y_slice}", fontsize=6)
                    else:
                        ax.set_title(f"{title_suffix}", fontsize=6)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Add colorbar for first column
                    if t == 0 and np.any(data):
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"3-Plane 3-Channel Prediction Overview (u,v,w) - Sample {sample_idx}", fontsize=14)
        plt.tight_layout()

        # Save comprehensive figure
        output_path = self.output_dir / f"3plane_comprehensive_sample_{sample_idx}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comprehensive visualization: {output_path}")

        # Log to wandb if available
        if self.wandb_run:
            self.wandb_run.log(
                {
                    f"3plane_comprehensive_sample_{sample_idx}": wandb.Image(str(output_path)),
                    "sample_idx": sample_idx,
                    "num_future_steps": num_future,
                }
            )

        plt.close()

    def create_3plane_animation(self, sample_idx: int = 0, num_future: int = 20):
        """Create animation showing 3-plane evolution over time."""
        print(f"Creating 3-plane animation for sample {sample_idx}...")

        # Get sample and generate predictions
        sample = self.test_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)

        # Generate longer prediction sequence
        pred_seq = self.generate_sequence_prediction(input_seq, num_future)

        # Denormalize
        input_seq_denorm = self.test_dataset.denormalize(input_seq)
        pred_seq_denorm = self.test_dataset.denormalize(pred_seq)

        # Combine input and predictions
        input_seq = input_seq_denorm.cpu().numpy()[0]  # (T_in, C, H, W)
        pred_seq = pred_seq_denorm.cpu().numpy()[0]  # (T_pred, C, H, W)

        # Concatenate: use last input frame + all predictions
        full_sequence = np.concatenate([input_seq[-1:], pred_seq], axis=0)  # (T_total, C, H, W)

        # Get channel info
        channel_info = self.test_dataset.get_channel_info()
        field_names = channel_info["field_names"]
        y_slices = channel_info["y_slices"]
        num_planes = channel_info["num_planes"]

        # Create figure for animation
        fig, axes = plt.subplots(num_planes, len(field_names), figsize=(4 * len(field_names), 3 * num_planes))

        if num_planes == 1:
            axes = axes.reshape(1, -1)
        if len(field_names) == 1:
            axes = axes.reshape(-1, 1)

        # Initialize plots
        ims = []
        titles = []

        for plane_idx in range(num_planes):
            for field_idx, field_name in enumerate(field_names):
                ax = axes[plane_idx, field_idx]
                channel_idx = plane_idx * len(field_names) + field_idx

                # Use first frame to set up plot
                first_frame = full_sequence[0, channel_idx]

                # All fields are velocity components
                cmap = "RdBu_r"
                vmax = max(abs(first_frame.min()), abs(first_frame.max()))
                vmin = -vmax

                im = ax.imshow(first_frame, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", animated=True)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                title = ax.set_title(f"Plane{plane_idx} ({field_name}) y={y_slices[plane_idx]}, t=0")
                titles.append(title)
                ims.append(im)

                ax.set_xlabel("x")
                ax.set_ylabel("z")

        def animate(frame):
            """Animation function."""
            for plane_idx in range(num_planes):
                for field_idx, field_name in enumerate(field_names):
                    idx = plane_idx * len(field_names) + field_idx
                    channel_idx = plane_idx * len(field_names) + field_idx

                    # Update image data
                    ims[idx].set_array(full_sequence[frame, channel_idx])

                    # Update title
                    titles[idx].set_text(f"Plane{plane_idx} ({field_name}) y={y_slices[plane_idx]}, t={frame}")

            return ims + titles

        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(full_sequence), interval=200, blit=True, repeat=True)

        # Save animation
        output_path = self.output_dir / f"3plane_animation_sample_{sample_idx}.mp4"
        writer = animation.FFMpegWriter(fps=5, metadata=dict(artist="FlowSwin3Plane"), bitrate=1800)
        anim.save(output_path, writer=writer)
        print(f"Saved animation: {output_path}")

        # Log to wandb if available
        if self.wandb_run:
            # 使用和evaluation_new.py一致的策略：不指定step，让WandB自动处理
            self.wandb_run.log(
                {
                    f"3plane_animation_sample_{sample_idx}": wandb.Video(str(output_path)),
                }
            )

        plt.close()

    def run_comprehensive_evaluation(self, num_samples: int = 1, num_future: int = 20):
        """Run comprehensive evaluation with both autoregressive and teacher forcing modes."""
        print(f"Running comprehensive evaluation on {num_samples} samples with {num_future} future steps...")

        # Test data evaluation (autoregressive)
        print("\n" + "=" * 60)
        print("TEST DATA EVALUATION (Autoregressive)")
        print("=" * 60)
        for i in range(num_samples):
            if i < len(self.test_dataset):
                print(f"\n=== Evaluating Sample {i} (Autoregressive) ===")
                self.visualize_3plane_prediction(sample_idx=i, num_future=num_future)

                if i == 0:  # Create animation only for first sample
                    self.create_3plane_animation(sample_idx=i, num_future=num_future)

        # Training data evaluation (teacher forcing)
        print("\n" + "=" * 60)
        print("TRAINING DATA EVALUATION (Teacher Forcing)")
        print("=" * 60)
        self.visualize_teacher_forcing(split="train", sample_idx=0, num_future=num_future)

        # Validation data evaluation (teacher forcing)
        print("\n" + "=" * 60)
        print("VALIDATION DATA EVALUATION (Teacher Forcing)")
        print("=" * 60)
        self.visualize_teacher_forcing(split="val", sample_idx=0, num_future=num_future)

        # Training data evaluation (autoregressive)
        print("\n" + "=" * 60)
        print("TRAINING DATA EVALUATION (Autoregressive)")
        print("=" * 60)
        self.visualize_autoregressive(split="train", sample_idx=0, num_future=num_future)

        # Validation data evaluation (autoregressive)
        print("\n" + "=" * 60)
        print("VALIDATION DATA EVALUATION (Autoregressive)")
        print("=" * 60)
        self.visualize_autoregressive(split="val", sample_idx=0, num_future=num_future)

        # Energy Spectra Analysis
        print("\n" + "=" * 60)
        print("ENERGY SPECTRA ANALYSIS")
        print("=" * 60)
        self._run_energy_spectra_analysis(num_future=num_future)

        print(f"\nEvaluation complete! Results saved to: {self.output_dir}")
        print("\nGenerated visualizations:")
        print("- Individual channel images (9 files per sample: 3 planes × 3 fields [u,v,w])")
        print("- Comprehensive overview images")
        print("- Detailed error analysis files")
        print("- Animations for temporal evolution")
        print("- Teacher forcing comparisons")
        print("- Autoregressive predictions")
        print("- Energy spectra analysis (1D and 2D spectra)")

    def visualize_teacher_forcing(self, split: str = "train", sample_idx: int = 0, num_future: int = 20):
        """Evaluate using teacher forcing (ground truth as input) for train/val data."""
        print(f"Evaluating {split} data with teacher forcing...")

        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        else:
            raise ValueError("Only 'train' and 'val' splits supported for teacher forcing")

        # Collect predictions and ground truth
        ground_truth_frames = []
        predictions = []

        print("\nTeacher Forcing Results:")
        print("Step | Plane | Field | MSE      | MAE      | RMS-Rel Error")
        print("-" * 60)

        for i in range(num_future):
            if sample_idx + i < len(dataset):
                sample = dataset[sample_idx + i]
                input_seq = sample["data"]["input_seq"].to(self.device)
                target = sample["label"]  # (1, 1, C, H, W)

                # Denormalize target for comparison
                target_denorm = dataset.denormalize(target).cpu().numpy()[0, 0]  # (C, H, W)
                ground_truth_frames.append(target_denorm)

                # Teacher forcing prediction
                with torch.no_grad():
                    pred = self.model(input_seq)  # (1, C, H, W)
                    pred_denorm = dataset.denormalize(pred).cpu().numpy()[0]  # (C, H, W)
                    predictions.append(pred_denorm)

                # Calculate per-channel errors for first few steps
                if i < 10:  # Print first 10 steps
                    channel_info = dataset.get_channel_info()
                    field_names = channel_info["field_names"]  # ["u", "v", "w"]
                    # y_slices = channel_info["y_slices"]  # [29, 54, 75] (unused)

                    for plane_idx in range(3):
                        for field_idx, field_name in enumerate(field_names):
                            channel_idx = plane_idx * len(field_names) + field_idx

                            pred_data = pred_denorm[channel_idx]
                            target_data = target_denorm[channel_idx]

                            mse = np.mean((pred_data - target_data) ** 2)
                            mae = np.mean(np.abs(pred_data - target_data))
                            target_rms = np.sqrt(np.mean(target_data**2))
                            rms_rel_error = np.sqrt(mse) / (target_rms + 1e-8)

                            print(
                                f"{i + 1:4d} | {plane_idx:5d} | {field_name:5s} | "
                                f"{mse:.6f} | {mae:.6f} | {rms_rel_error:.6f}"
                            )

        # Save teacher forcing errors to file
        self._save_teacher_forcing_errors(ground_truth_frames, predictions, split, sample_idx)

        # Create detailed teacher forcing visualization
        self._create_teacher_forcing_visualization(ground_truth_frames, predictions, split, sample_idx)

        return ground_truth_frames, predictions

    def visualize_autoregressive(self, split: str = "train", sample_idx: int = 0, num_future: int = 20):
        """Evaluate using autoregressive prediction for train/val data."""
        print(f"\nEvaluating {split} data with autoregressive prediction...")

        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        else:
            raise ValueError("Only 'train' and 'val' splits supported for autoregressive")

        # Get ground truth sequence
        ground_truth_frames = []
        for i in range(num_future):
            if sample_idx + i < len(dataset):
                sample_i = dataset[sample_idx + i]
                target_i = sample_i["label"]  # (1, 1, C, H, W)
                target_denorm = dataset.denormalize(target_i)
                target_frame = target_denorm.cpu().numpy()[0, 0]  # (C, H, W)
                ground_truth_frames.append(target_frame)
            else:
                if ground_truth_frames:
                    ground_truth_frames.append(ground_truth_frames[-1])

        # Get initial sample for autoregressive prediction
        sample = dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)

        # Generate autoregressive predictions
        pred_seq = self.generate_sequence_prediction(input_seq, num_future)
        pred_seq_denorm = dataset.denormalize(pred_seq)
        pred_seq = pred_seq_denorm.cpu().numpy()[0]  # (T_pred, C, H, W)

        print("\nAutoregressive Results:")
        print("Step | Plane | Field | MSE      | MAE      | RMS-Rel Error")
        print("-" * 60)

        # Calculate metrics
        channel_info = dataset.get_channel_info()
        field_names = channel_info["field_names"]
        y_slices = channel_info["y_slices"]

        num_steps = min(len(ground_truth_frames), pred_seq.shape[0], 10)
        for i in range(num_steps):
            pred_frame = pred_seq[i]
            gt_frame = ground_truth_frames[i]

            for plane_idx in range(3):
                for field_idx, field_name in enumerate(field_names):
                    channel_idx = plane_idx * len(field_names) + field_idx

                    pred_data = pred_frame[channel_idx]
                    target_data = gt_frame[channel_idx]

                    mse = np.mean((pred_data - target_data) ** 2)
                    mae = np.mean(np.abs(pred_data - target_data))
                    target_rms = np.sqrt(np.mean(target_data**2))
                    rms_rel_error = np.sqrt(mse) / (target_rms + 1e-8)

                    print(
                        f"{i + 1:4d} | {plane_idx:5d} | {field_name:5s} | {mse:.6f} | {mae:.6f} | {rms_rel_error:.6f}"
                    )

        # Save autoregressive errors and create visualization
        self._save_detailed_error_analysis(
            pred_seq, ground_truth_frames, f"{split}_{sample_idx}", field_names, y_slices
        )
        self._create_autoregressive_visualization(ground_truth_frames, pred_seq, split, sample_idx)

        return ground_truth_frames, pred_seq

    def _save_teacher_forcing_errors(self, ground_truth_frames, predictions, split, sample_idx):
        """Save detailed teacher forcing errors to Excel and text files."""
        import pandas as pd

        if not ground_truth_frames or not predictions:
            return

        # Calculate detailed errors for all timesteps and channels
        error_data = {"timestep": [], "plane": [], "field": [], "channel": [], "mse": [], "mae": [], "rms_rel": []}

        channel_info = self.test_dataset.get_channel_info()  # Use test dataset channel info
        field_names = channel_info["field_names"]
        # y_slices = channel_info["y_slices"]  # (unused)

        for t, (pred, target) in enumerate(zip(predictions, ground_truth_frames, strict=False)):
            for plane_idx in range(3):
                for field_idx, field_name in enumerate(field_names):
                    channel_idx = plane_idx * len(field_names) + field_idx

                    pred_data = pred[channel_idx]
                    target_data = target[channel_idx]

                    mse = np.mean((pred_data - target_data) ** 2)
                    mae = np.mean(np.abs(pred_data - target_data))
                    target_rms = np.sqrt(np.mean(target_data**2))
                    rms_rel_error = np.sqrt(mse) / (target_rms + 1e-8)

                    error_data["timestep"].append(t + 1)
                    error_data["plane"].append(plane_idx)
                    error_data["field"].append(field_name)
                    error_data["channel"].append(f"plane_{plane_idx}_{field_name}")
                    error_data["mse"].append(mse)
                    error_data["mae"].append(mae)
                    error_data["rms_rel"].append(rms_rel_error)

        # Create DataFrame
        df = pd.DataFrame(error_data)

        # Save to text file (CSV format)
        txt_path = self.output_dir / f"errors_teacher_forcing_{split}_sample_{sample_idx}.txt"
        df.to_csv(txt_path, sep="\t", index=False)

        # Try to save to Excel if openpyxl is available
        try:
            excel_path = self.output_dir / f"errors_teacher_forcing_{split}_sample_{sample_idx}.xlsx"
            df.to_excel(excel_path, index=False)
            print(f"Saved teacher forcing errors to: {excel_path} and {txt_path}")
        except ImportError:
            print(
                f"Saved teacher forcing errors to: {txt_path} "
                "(Excel not available - install openpyxl for .xlsx support)"
            )

    def _create_3plane_comparison_plot(self, ground_truth_frames, predictions, mode, sample_idx):
        """Create detailed plane-wise comparison visualization."""
        import matplotlib.pyplot as plt

        if not ground_truth_frames or not predictions:
            return

        # Use first timestep for visualization
        pred = predictions[0]
        target = ground_truth_frames[0]

        # Create 3x4 subplot (3 planes x 4 fields)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f"3-Plane Comparison: {mode} (Sample {sample_idx})", fontsize=16)

        channel_info = self.test_dataset.get_channel_info()
        field_names = channel_info["field_names"]
        y_slices = channel_info["y_slices"]

        for plane_idx in range(3):
            for field_idx, field_name in enumerate(field_names):
                ax = axes[plane_idx, field_idx]
                channel_idx = plane_idx * len(field_names) + field_idx

                # Create side-by-side comparison: prediction | ground truth
                pred_field = pred[channel_idx]
                target_field = target[channel_idx]
                combined = np.concatenate([pred_field, target_field], axis=1)

                # Use colormap for velocity fields
                cmap = "RdBu_r"
                vmax = max(abs(combined.min()), abs(combined.max()))
                vmin = -vmax

                im = ax.imshow(combined, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

                ax.set_title(f"Plane {plane_idx} (y={y_slices[plane_idx]}) - {field_name.upper()}\nPred | GT")
                ax.axis("off")

                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout()

        # Save plot
        save_path = self.output_dir / f"3plane_comparison_{mode}_sample_{sample_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {mode} visualization: {save_path}")

    def _save_detailed_error_analysis(self, pred_seq, ground_truth_frames, sample_idx, field_names, y_slices):
        """Save detailed error analysis for autoregressive predictions."""
        import pandas as pd

        # Calculate detailed errors for all timesteps and channels
        error_data = {
            "timestep": [],
            "plane": [],
            "field": [],
            "y_slice": [],
            "channel": [],
            "mse": [],
            "mae": [],
            "rms_rel": [],
            "max_error": [],
            "mean_gt": [],
            "std_gt": [],
        }

        num_timesteps = min(len(ground_truth_frames), pred_seq.shape[0])

        for t in range(num_timesteps):
            pred_frame = pred_seq[t]  # (C, H, W)
            gt_frame = ground_truth_frames[t]  # (C, H, W)

            for plane_idx in range(3):
                for field_idx, field_name in enumerate(field_names):
                    channel_idx = plane_idx * len(field_names) + field_idx
                    y_slice = y_slices[plane_idx]

                    pred_data = pred_frame[channel_idx]
                    gt_data = gt_frame[channel_idx]

                    # Calculate comprehensive error metrics
                    error = pred_data - gt_data
                    abs_error = np.abs(error)

                    mse = np.mean(error**2)
                    mae = np.mean(abs_error)
                    max_error = np.max(abs_error)

                    # Ground truth statistics
                    mean_gt = np.mean(gt_data)
                    std_gt = np.std(gt_data)

                    # RMS relative error
                    gt_rms = np.sqrt(np.mean(gt_data**2))
                    rms_rel_error = np.sqrt(mse) / (gt_rms + 1e-8)

                    error_data["timestep"].append(t + 1)
                    error_data["plane"].append(plane_idx)
                    error_data["field"].append(field_name)
                    error_data["y_slice"].append(y_slice)
                    error_data["channel"].append(f"plane_{plane_idx}_{field_name}")
                    error_data["mse"].append(mse)
                    error_data["mae"].append(mae)
                    error_data["rms_rel"].append(rms_rel_error)
                    error_data["max_error"].append(max_error)
                    error_data["mean_gt"].append(mean_gt)
                    error_data["std_gt"].append(std_gt)

        # Create DataFrame
        df = pd.DataFrame(error_data)

        # Save to text file (CSV format)
        txt_path = self.output_dir / f"detailed_errors_autoregressive_sample_{sample_idx}.txt"
        df.to_csv(txt_path, sep="\t", index=False)

        # Save summary statistics
        summary_path = self.output_dir / f"error_summary_sample_{sample_idx}.txt"
        with open(summary_path, "w") as f:
            f.write(f"Error Analysis Summary for Sample {sample_idx}\n")
            f.write("=" * 50 + "\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total timesteps analyzed: {num_timesteps}\n")
            f.write(f"Average MSE across all channels: {df['mse'].mean():.6f}\n")
            f.write(f"Average MAE across all channels: {df['mae'].mean():.6f}\n")
            f.write(f"Average RMS-Rel Error: {df['rms_rel'].mean():.6f}\n\n")

            # Per-field statistics
            f.write("Per-Field Statistics:\n")
            for field in field_names:
                field_data = df[df["field"] == field]
                if not field_data.empty:
                    f.write(f"  {field.upper()}:\n")
                    f.write(f"    MSE: {field_data['mse'].mean():.6f} ± {field_data['mse'].std():.6f}\n")
                    f.write(f"    MAE: {field_data['mae'].mean():.6f} ± {field_data['mae'].std():.6f}\n")
                    f.write(f"    RMS-Rel: {field_data['rms_rel'].mean():.6f} ± {field_data['rms_rel'].std():.6f}\n")

            f.write("\n")

            # Per-plane statistics
            f.write("Per-Plane Statistics:\n")
            for plane_idx in range(3):
                y_slice = y_slices[plane_idx]
                plane_data = df[df["plane"] == plane_idx]
                if not plane_data.empty:
                    f.write(f"  Plane {plane_idx} (y={y_slice}):\n")
                    f.write(f"    MSE: {plane_data['mse'].mean():.6f} ± {plane_data['mse'].std():.6f}\n")
                    f.write(f"    MAE: {plane_data['mae'].mean():.6f} ± {plane_data['mae'].std():.6f}\n")
                    f.write(f"    RMS-Rel: {plane_data['rms_rel'].mean():.6f} ± {plane_data['rms_rel'].std():.6f}\n")

        # Try to save to Excel if openpyxl is available
        try:
            excel_path = self.output_dir / f"detailed_errors_autoregressive_sample_{sample_idx}.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Detailed_Errors", index=False)

                # Create summary sheet
                summary_stats = []

                # Overall stats
                summary_stats.append(["Metric", "Value"])
                summary_stats.append(["Total Timesteps", num_timesteps])
                summary_stats.append(["Overall MSE", df["mse"].mean()])
                summary_stats.append(["Overall MAE", df["mae"].mean()])
                summary_stats.append(["Overall RMS-Rel", df["rms_rel"].mean()])
                summary_stats.append(["", ""])

                # Per-field stats
                summary_stats.append(["Per-Field Statistics", ""])
                for field in field_names:
                    field_data = df[df["field"] == field]
                    if not field_data.empty:
                        summary_stats.append([f"{field.upper()} MSE", field_data["mse"].mean()])
                        summary_stats.append([f"{field.upper()} MAE", field_data["mae"].mean()])
                        summary_stats.append([f"{field.upper()} RMS-Rel", field_data["rms_rel"].mean()])

                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name="Summary", index=False, header=False)

            print(f"Saved detailed error analysis to: {excel_path}, {txt_path}, and {summary_path}")
        except ImportError:
            print(
                f"Saved detailed error analysis to: {txt_path} and {summary_path} "
                "(Excel not available - install openpyxl for .xlsx support)"
            )

        return df

    def _create_teacher_forcing_visualization(self, ground_truth_frames, predictions, split, sample_idx):
        """Create detailed teacher forcing visualization."""
        channel_info = self.test_dataset.get_channel_info()
        field_names = channel_info["field_names"]
        y_slices = channel_info["y_slices"]

        # Limit display steps
        display_steps = min(len(predictions), 15)

        # Create one visualization per channel
        for plane_idx in range(3):
            y_slice = y_slices[plane_idx]

            for field_idx, field_name in enumerate(field_names):
                channel_idx = plane_idx * len(field_names) + field_idx

                # Create figure: 3 rows (GT, Pred, Error) × timesteps
                fig, axes = plt.subplots(3, display_steps, figsize=(2 * display_steps, 8))
                if display_steps == 1:
                    axes = axes.reshape(3, 1)

                # Calculate channel-specific colorbar range
                all_data = []
                for t in range(display_steps):
                    all_data.append(ground_truth_frames[t][channel_idx])
                    all_data.append(predictions[t][channel_idx])

                # All fields are velocity components
                cmap = "RdBu_r"
                vmax = max([abs(data.min()) for data in all_data] + [abs(data.max()) for data in all_data])
                vmin = -vmax

                for t in range(display_steps):
                    gt_data = ground_truth_frames[t][channel_idx]
                    pred_data = predictions[t][channel_idx]
                    error = np.abs(pred_data - gt_data)

                    # Ground truth
                    im1 = axes[0, t].imshow(gt_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                    axes[0, t].set_title(f"GT t+{t + 1}", fontsize=8)
                    axes[0, t].set_xticks([])
                    axes[0, t].set_yticks([])

                    # Prediction
                    im2 = axes[1, t].imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                    axes[1, t].set_title(f"TF t+{t + 1}", fontsize=8)
                    axes[1, t].set_xticks([])
                    axes[1, t].set_yticks([])

                    # Error
                    im3 = axes[2, t].imshow(error, cmap="Reds", origin="lower")
                    mae = np.mean(error)
                    axes[2, t].set_title(f"Err\nMAE:{mae:.3f}", fontsize=8)
                    axes[2, t].set_xticks([])
                    axes[2, t].set_yticks([])

                    # Add colorbar for first column
                    if t == 0:
                        plt.colorbar(im1, ax=axes[0, t], fraction=0.046, pad=0.04)
                        plt.colorbar(im2, ax=axes[1, t], fraction=0.046, pad=0.04)
                        plt.colorbar(im3, ax=axes[2, t], fraction=0.046, pad=0.04)

                # Set row labels
                axes[0, 0].set_ylabel("Ground Truth", fontsize=10)
                axes[1, 0].set_ylabel("Teacher Forcing", fontsize=10)
                axes[2, 0].set_ylabel("Error", fontsize=10)

                plt.suptitle(
                    f"{split.upper()} - Plane {plane_idx} - {field_name.upper()} (y={y_slice}) - Teacher Forcing",
                    fontsize=12,
                )
                plt.tight_layout()

                # Save visualization
                output_path = self.output_dir / f"{split}_tf_plane{plane_idx}_{field_name}_sample_{sample_idx}.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Saved TF visualization: {output_path}")

                # Log to wandb
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"{split}_tf_plane{plane_idx}_{field_name}_sample_{sample_idx}": wandb.Image(str(output_path))}
                    )

                plt.close()

    def _create_autoregressive_visualization(self, ground_truth_frames, pred_seq, split, sample_idx):
        """Create detailed autoregressive visualization."""
        channel_info = self.test_dataset.get_channel_info()
        field_names = channel_info["field_names"]
        y_slices = channel_info["y_slices"]

        # Limit display steps
        display_steps = min(len(ground_truth_frames), pred_seq.shape[0], 15)

        # Create one visualization per channel
        for plane_idx in range(3):
            y_slice = y_slices[plane_idx]

            for field_idx, field_name in enumerate(field_names):
                channel_idx = plane_idx * len(field_names) + field_idx

                # Create figure: 3 rows (GT, Pred, Error) × timesteps
                fig, axes = plt.subplots(3, display_steps, figsize=(2 * display_steps, 8))
                if display_steps == 1:
                    axes = axes.reshape(3, 1)

                # Calculate channel-specific colorbar range
                all_data = []
                for t in range(display_steps):
                    all_data.append(ground_truth_frames[t][channel_idx])
                    all_data.append(pred_seq[t][channel_idx])

                # All fields are velocity components
                cmap = "RdBu_r"
                vmax = max([abs(data.min()) for data in all_data] + [abs(data.max()) for data in all_data])
                vmin = -vmax

                for t in range(display_steps):
                    gt_data = ground_truth_frames[t][channel_idx]
                    pred_data = pred_seq[t][channel_idx]
                    error = np.abs(pred_data - gt_data)

                    # Ground truth
                    im1 = axes[0, t].imshow(gt_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                    axes[0, t].set_title(f"GT t+{t + 1}", fontsize=8)
                    axes[0, t].set_xticks([])
                    axes[0, t].set_yticks([])

                    # Prediction
                    im2 = axes[1, t].imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                    axes[1, t].set_title(f"AR t+{t + 1}", fontsize=8)
                    axes[1, t].set_xticks([])
                    axes[1, t].set_yticks([])

                    # Error
                    im3 = axes[2, t].imshow(error, cmap="Reds", origin="lower")
                    mae = np.mean(error)
                    axes[2, t].set_title(f"Err\nMAE:{mae:.3f}", fontsize=8)
                    axes[2, t].set_xticks([])
                    axes[2, t].set_yticks([])

                    # Add colorbar for first column
                    if t == 0:
                        plt.colorbar(im1, ax=axes[0, t], fraction=0.046, pad=0.04)
                        plt.colorbar(im2, ax=axes[1, t], fraction=0.046, pad=0.04)
                        plt.colorbar(im3, ax=axes[2, t], fraction=0.046, pad=0.04)

                # Set row labels
                axes[0, 0].set_ylabel("Ground Truth", fontsize=10)
                axes[1, 0].set_ylabel("Autoregressive", fontsize=10)
                axes[2, 0].set_ylabel("Error", fontsize=10)

                plt.suptitle(
                    f"{split.upper()} - Plane {plane_idx} - {field_name.upper()} (y={y_slice}) - Autoregressive",
                    fontsize=12,
                )
                plt.tight_layout()

                # Save visualization
                output_path = self.output_dir / f"{split}_ar_plane{plane_idx}_{field_name}_sample_{sample_idx}.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Saved AR visualization: {output_path}")

                # Log to wandb
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"{split}_ar_plane{plane_idx}_{field_name}_sample_{sample_idx}": wandb.Image(str(output_path))}
                    )

                plt.close()

    def _compute_energy_spectra(self, frames, field_names, y_slices, dx=1.0, dz=1.0):
        """
        Compute energy spectra from prediction frames.

        Args:
            frames: Tensor of shape (T, C, H, W) where C = 3 planes × len(field_names)
            field_names: List of field names (e.g., ["u", "v", "w"] or ["u", "v", "w", "p"])
            y_slices: List of y-slice positions [29, 54, 75]
            dx: Grid spacing in x direction
            dz: Grid spacing in z direction

        Returns:
            dict: Dictionary containing spectra data for each plane and field
        """
        print("Computing energy spectra...")

        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()

        T, _C, H, W = frames.shape
        spectra_results = {}

        # Compute frequency grids (no fftshift here, only in visualization)
        kx = np.fft.fftfreq(W, dx)  # x-direction wavenumbers
        kz = np.fft.fftfreq(H, dz)  # z-direction wavenumbers

        # Only keep positive frequencies for plotting
        kx_pos = kx[kx > 0]
        kz_pos = kz[kz > 0]

        for plane_idx in range(3):
            y_slice = y_slices[plane_idx]
            spectra_results[f"plane{plane_idx}"] = {"y_slice": y_slice, "fields": {}}

            for field_idx, field_name in enumerate(field_names):
                print(f"  Processing plane {plane_idx} ({field_name}) at y={y_slice}")

                # Extract channel data for this plane and field
                channel_idx = plane_idx * len(field_names) + field_idx
                field_data = frames[:, channel_idx, :, :]  # Shape: (T, H, W)

                # Time-averaged energy spectrum
                spectrum_2d_sum = np.zeros((H, W))

                for t in range(T):
                    # Remove plane average (mean) before FFT
                    data_slice = field_data[t] - np.mean(field_data[t])

                    # Compute 2D FFT with proper normalization
                    fft_2d = np.fft.fft2(data_slice) / (H * W)

                    # Compute energy spectrum: E(kx, kz) = 0.5 * |q_hat|^2
                    spectrum_2d = 0.5 * np.abs(fft_2d) ** 2
                    spectrum_2d_sum += spectrum_2d

                # Time average (do NOT apply fftshift here, only for visualization)
                spectrum_2d_avg = spectrum_2d_sum / T

                # Compute 1D spectra by integration
                # Streamwise spectrum E(kx) = sum over kz
                spectrum_kx = np.sum(spectrum_2d_avg, axis=0)  # Sum over H (kz direction)

                # Spanwise spectrum E(kz) = sum over kx
                spectrum_kz = np.sum(spectrum_2d_avg, axis=1)  # Sum over W (kx direction)

                # Store results
                spectra_results[f"plane{plane_idx}"]["fields"][field_name] = {
                    "spectrum_2d": spectrum_2d_avg,
                    "spectrum_kx": spectrum_kx,
                    "spectrum_kz": spectrum_kz,
                    "kx": kx,
                    "kz": kz,
                    "kx_pos": kx_pos,
                    "kz_pos": kz_pos,
                }

        print("Energy spectra computation completed.")
        return spectra_results

    def _plot_energy_spectra(self, spectra, field_names, y_slices, mode="prediction"):
        """
        Plot and save energy spectra.

        Args:
            spectra: Dictionary containing spectra data from _compute_energy_spectra
            field_names: List of field names
            y_slices: List of y-slice positions
            mode: String identifier for the plot type ("prediction" or "ground_truth")
        """
        print(f"Plotting energy spectra for {mode}...")

        for plane_idx in range(3):
            y_slice = y_slices[plane_idx]
            plane_data = spectra[f"plane{plane_idx}"]

            for field_name in field_names:
                field_data = plane_data["fields"][field_name]

                # Extract data
                spectrum_2d = field_data["spectrum_2d"]
                spectrum_kx = field_data["spectrum_kx"]
                spectrum_kz = field_data["spectrum_kz"]
                kx = field_data["kx"]
                kz = field_data["kz"]
                kx_pos = field_data["kx_pos"]
                kz_pos = field_data["kz_pos"]

                # Save numerical data
                base_name = f"spectrum_{mode}_plane{plane_idx}_{field_name}"
                np.save(self.output_dir / f"{base_name}_2d.npy", spectrum_2d)
                np.save(self.output_dir / f"{base_name}_kx.npy", spectrum_kx)
                np.save(self.output_dir / f"{base_name}_kz.npy", spectrum_kz)

                # Create combined figure with all spectrum visualizations
                fig = plt.figure(figsize=(18, 6))

                # Prepare data for plotting
                kx_pos_mask = kx > 0
                kz_pos_mask = kz > 0
                spectrum_kx_pos = spectrum_kx[kx_pos_mask]
                spectrum_kz_pos = spectrum_kz[kz_pos_mask]
                # For 2D visualization, apply fftshift to center zero frequency
                spectrum_2d_shift = np.fft.fftshift(spectrum_2d)
                kx_shift = np.fft.fftshift(kx)
                kz_shift = np.fft.fftshift(kz)
                kx_2d, kz_2d = np.meshgrid(kx_shift, kz_shift)
                spectrum_log = np.log10(spectrum_2d_shift + 1e-12)

                # Subplot 1: 1D streamwise spectrum E(kx)
                plt.subplot(1, 3, 1)
                plt.loglog(kx_pos, spectrum_kx_pos, "b-", linewidth=2)
                plt.xlabel("Streamwise Wavenumber kx")
                plt.ylabel("Energy Spectrum E(kx)")
                plt.title("Streamwise Spectrum")
                plt.grid(True, alpha=0.3)

                # Subplot 2: 1D spanwise spectrum E(kz)
                plt.subplot(1, 3, 2)
                plt.loglog(kz_pos, spectrum_kz_pos, "r-", linewidth=2)
                plt.xlabel("Spanwise Wavenumber kz")
                plt.ylabel("Energy Spectrum E(kz)")
                plt.title("Spanwise Spectrum")
                plt.grid(True, alpha=0.3)

                # Subplot 3: 2D spectrum heatmap
                plt.subplot(1, 3, 3)
                contour = plt.contourf(kx_2d, kz_2d, spectrum_log, levels=50, cmap="viridis")
                plt.colorbar(contour, label="log₁₀(Energy Spectrum)")
                plt.xlabel("Streamwise Wavenumber kx")
                plt.ylabel("Spanwise Wavenumber kz")
                plt.title("2D Spectrum")

                # Overall title for the figure
                fig.suptitle(
                    f"{mode.title()} - Plane {plane_idx} ({field_name.upper()}) - Energy Spectra (y={y_slice})",
                    fontsize=14,
                    fontweight="bold",
                )

                plt.tight_layout()

                # Save combined figure
                output_path = self.output_dir / f"{base_name}_combined.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"  Saved combined spectrum plot: {output_path}")

        print(f"Energy spectra plotting for {mode} completed.")

    def _run_energy_spectra_analysis(self, num_future: int = 100, sample_idx: int = 0):
        """
        Run energy spectra analysis on both autoregressive predictions and ground truth.

        Args:
            num_future: Number of future steps to generate for analysis (default 50)
            sample_idx: Index of the sample to analyze
        """
        print(f"Generating {num_future} steps for energy spectra analysis...")

        # Get channel info
        channel_info = self.test_dataset.get_channel_info()
        field_names = channel_info["field_names"]  # e.g., ["u", "v", "w"] or ["u", "v", "w", "p"]
        y_slices = channel_info["y_slices"]  # [29, 54, 75]

        # Generate autoregressive predictions
        print("Generating autoregressive predictions...")
        sample = self.test_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)

        # Generate autoregressive predictions
        pred_frames = self.generate_sequence_prediction(input_seq, num_future)  # (B, T_pred, C, H, W)

        # Denormalize predictions
        pred_frames_denorm = self.test_dataset.denormalize(pred_frames).cpu().numpy()  # (B, T_pred, C, H, W)

        # Remove batch dimension: (T_pred, C, H, W)
        pred_frames_array = pred_frames_denorm[0]  # Take first (and only) batch

        # Collect corresponding ground truth frames
        print("Collecting ground truth frames...")
        ground_truth_frames = []
        for i in range(num_future):
            if sample_idx + i + 1 < len(self.test_dataset):  # +1 because first prediction is t+1
                gt_sample = self.test_dataset[sample_idx + i + 1]
                gt_frame = gt_sample["data"]["input_seq"][:, -1:, :, :, :]  # Get last frame
                gt_frame_denorm = self.test_dataset.denormalize(gt_frame).cpu().numpy()[0, 0]  # (C, H, W)
                ground_truth_frames.append(gt_frame_denorm)
            else:
                # If we run out of ground truth data, duplicate the last frame
                if ground_truth_frames:
                    ground_truth_frames.append(ground_truth_frames[-1])
                else:
                    # Use a zero frame as fallback
                    zero_frame = np.zeros_like(pred_frames_array[0])
                    ground_truth_frames.append(zero_frame)

        # Convert to numpy array: (T, C, H, W)
        gt_frames_array = np.stack(ground_truth_frames, axis=0)

        print(f"Prediction frames shape: {pred_frames_array.shape}")
        print(f"Ground truth frames shape: {gt_frames_array.shape}")

        # Verify shapes are consistent
        assert pred_frames_array.shape == gt_frames_array.shape, (
            f"Shape mismatch: pred {pred_frames_array.shape} vs gt {gt_frames_array.shape}"
        )

        # Compute energy spectra for predictions
        print("\nComputing energy spectra for predictions...")
        pred_spectra = self._compute_energy_spectra(pred_frames_array, field_names, y_slices, dx=1.0, dz=1.0)

        # Compute energy spectra for ground truth
        print("\nComputing energy spectra for ground truth...")
        gt_spectra = self._compute_energy_spectra(gt_frames_array, field_names, y_slices, dx=1.0, dz=1.0)

        # Plot energy spectra
        print("\nPlotting energy spectra...")
        self._plot_energy_spectra(pred_spectra, field_names, y_slices, mode="prediction")
        self._plot_energy_spectra(gt_spectra, field_names, y_slices, mode="ground_truth")

        # Generate comparison plots
        print("\nGenerating comparison plots...")
        self._plot_spectra_comparison(pred_spectra, gt_spectra, field_names, y_slices)

        print(f"Energy spectra analysis complete! Results saved to: {self.output_dir}")

    def _plot_spectra_comparison(self, pred_spectra, gt_spectra, field_names, y_slices):
        """
        Plot combined comparison between prediction and ground truth spectra.

        Args:
            pred_spectra: Prediction spectra from _compute_energy_spectra
            gt_spectra: Ground truth spectra from _compute_energy_spectra
            field_names: List of field names
            y_slices: List of y-slice positions
        """
        print("Creating combined spectra comparison plots...")

        for plane_idx in range(3):
            y_slice = y_slices[plane_idx]

            for field_name in field_names:
                # Extract data
                pred_data = pred_spectra[f"plane{plane_idx}"]["fields"][field_name]
                gt_data = gt_spectra[f"plane{plane_idx}"]["fields"][field_name]

                # Create combined figure with 4 subplots
                fig = plt.figure(figsize=(20, 10))

                # Prepare data for plotting (fix indexing after fftshift)
                kx = pred_data["kx"]
                kz = pred_data["kz"]
                kx_pos = pred_data["kx_pos"]
                kz_pos = pred_data["kz_pos"]

                # Get positive frequency masks and corresponding spectra
                kx_pos_mask = kx > 0
                kz_pos_mask = kz > 0

                pred_spectrum_kx_pos = pred_data["spectrum_kx"][kx_pos_mask]
                gt_spectrum_kx_pos = gt_data["spectrum_kx"][kx_pos_mask]
                pred_spectrum_kz_pos = pred_data["spectrum_kz"][kz_pos_mask]
                gt_spectrum_kz_pos = gt_data["spectrum_kz"][kz_pos_mask]

                # Subplot 1: 1D streamwise spectrum comparison E(kx)
                plt.subplot(2, 2, 1)
                plt.loglog(kx_pos, pred_spectrum_kx_pos, "b-", linewidth=2, label="Prediction")
                plt.loglog(kx_pos, gt_spectrum_kx_pos, "r--", linewidth=2, label="Ground Truth")
                plt.xlabel("Streamwise Wavenumber kx")
                plt.ylabel("Energy Spectrum E(kx)")
                plt.title("Streamwise Spectrum Comparison")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Subplot 2: 1D spanwise spectrum comparison E(kz)
                plt.subplot(2, 2, 2)
                plt.loglog(kz_pos, pred_spectrum_kz_pos, "b-", linewidth=2, label="Prediction")
                plt.loglog(kz_pos, gt_spectrum_kz_pos, "r--", linewidth=2, label="Ground Truth")
                plt.xlabel("Spanwise Wavenumber kz")
                plt.ylabel("Energy Spectrum E(kz)")
                plt.title("Spanwise Spectrum Comparison")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Prepare 2D spectrum data with consistent colorbar scale
                pred_spectrum_2d = pred_data["spectrum_2d"]
                gt_spectrum_2d = gt_data["spectrum_2d"]

                # For 2D visualization, apply fftshift to center zero frequency
                pred_spectrum_2d_shift = np.fft.fftshift(pred_spectrum_2d)
                gt_spectrum_2d_shift = np.fft.fftshift(gt_spectrum_2d)
                kx_shift = np.fft.fftshift(kx)
                kz_shift = np.fft.fftshift(kz)

                # Use log scale for better visualization and find common range
                pred_spectrum_log = np.log10(pred_spectrum_2d_shift + 1e-12)
                gt_spectrum_log = np.log10(gt_spectrum_2d_shift + 1e-12)

                # Set consistent color scale
                vmin = min(np.min(pred_spectrum_log), np.min(gt_spectrum_log))
                vmax = max(np.max(pred_spectrum_log), np.max(gt_spectrum_log))

                # Create meshgrid for plotting
                kx_2d, kz_2d = np.meshgrid(kx_shift, kz_shift)

                # Subplot 3: 2D prediction spectrum
                plt.subplot(2, 2, 3)
                contour1 = plt.contourf(
                    kx_2d, kz_2d, pred_spectrum_log, levels=50, cmap="viridis", vmin=vmin, vmax=vmax
                )
                plt.colorbar(contour1, label="log₁₀(Energy Spectrum)")
                plt.xlabel("Streamwise Wavenumber kx")
                plt.ylabel("Spanwise Wavenumber kz")
                plt.title("2D Spectrum - Prediction")

                # Subplot 4: 2D ground truth spectrum
                plt.subplot(2, 2, 4)
                contour2 = plt.contourf(kx_2d, kz_2d, gt_spectrum_log, levels=50, cmap="viridis", vmin=vmin, vmax=vmax)
                plt.colorbar(contour2, label="log₁₀(Energy Spectrum)")
                plt.xlabel("Streamwise Wavenumber kx")
                plt.ylabel("Spanwise Wavenumber kz")
                plt.title("2D Spectrum - Ground Truth")

                # Overall title for the figure
                fig.suptitle(
                    f"Spectrum Comparison - Plane {plane_idx} ({field_name.upper()}) at y={y_slice}",
                    fontsize=14,
                    fontweight="bold",
                )

                plt.tight_layout()

                # Save combined comparison figure
                output_path = self.output_dir / f"spectrum_comparison_plane{plane_idx}_{field_name}_combined.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"  Saved combined spectrum comparison: {output_path}")

        print("Combined spectra comparison plots completed.")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 3-plane Flow Swin Transformer")
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to evaluate")
    parser.add_argument("--num_future", type=int, default=100, help="Number of future steps to predict")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions as H5 files")

    args = parser.parse_args()

    # Use command line argument or default path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Default to the hardcoded path if no argument provided
        checkpoint_path = (
            "/home/sh/CB/icon-thewell-dev/logs/flow_swin_3plane/runs/"
            "2025-10-15_22-32-26-187481/checkpoints/step_49800.ckpt"
        )

    # Load model config (simplified for direct usage)
    from omegaconf import OmegaConf

    # Create a basic model config for 3-plane model (9 channels: 3 planes × 3 fields)
    model_cfg = OmegaConf.create(
        {
            "input_shape": [128, 128],
            "sequence_length": 5,
            "prediction_horizon": 1,
            "num_channels": 9,
            "patch_size": [4, 4],
            "embed_dim": 128,
            "depths": [2, 2, 4, 6, 4, 2, 2],
            "num_heads": 8,
            "window_size": [8, 8],
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "drop_rate": 0.1,
            "attn_drop_rate": 0.1,
            "drop_path_rate": 0.1,
        }
    )

    # Create evaluator and run evaluation
    evaluator = ThreePlaneModelEvaluator(
        checkpoint_path=checkpoint_path, model_cfg=model_cfg, save_predictions=args.save_predictions
    )

    evaluator.run_comprehensive_evaluation(num_samples=args.num_samples, num_future=args.num_future)


if __name__ == "__main__":
    main()
