#!/usr/bin/env python3
"""
Simple evaluation script for 2D Flow Swin Transformer implementation.
Loads the best model checkpoint and generates visualizations and videos.
"""

import os
import sys
import warnings
from pathlib import Path

import hydra
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

from src.datasets.flow_sequence_2d.fast_flow_dataset import FastFlowSequence2DDataset  # noqa: E402


class SimpleModelEvaluator:
    """Simple evaluator for the 2D Flow Swin Transformer model."""

    def __init__(self, checkpoint_path: str, model_cfg: DictConfig):
        """Initialize the evaluator.

        Args:
            checkpoint_path: Path to the model checkpoint
            model_cfg: Model configuration from Hydra
        """
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model weights
        self.model = self._load_model()
        self.model.eval()

        # Setup data
        self.test_dataset = self._setup_test_dataset()
        self.train_dataset = self._setup_train_dataset()
        self.val_dataset = self._setup_val_dataset()

        # Create output directory using log directory name
        log_dir_name = self._extract_log_dir_name()
        self.output_dir = Path(f"evaluation_results/evaluation_results_{log_dir_name}")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize wandb for evaluation logging
        if WANDB_AVAILABLE:
            # Try to resume the training run if possible
            training_run_id = self._extract_wandb_run_id()

            if training_run_id:
                print(f"Found training wandb run ID: {training_run_id}")
                try:
                    # Resume the existing training run for evaluation
                    self.wandb_run = wandb.init(
                        project="turbulence_swin",  # Same project as training
                        id=training_run_id,  # Resume the same run
                        resume="allow",  # Allow resuming the run
                        tags=["evaluation", "flow", "swin", "2d"],  # Add evaluation tag
                    )
                    print("Successfully resumed training wandb run for evaluation")
                except Exception as e:
                    print(f"Failed to resume training run: {e}")
                    print("Creating new evaluation run...")
                    # Fallback: create a new linked run
                    self.wandb_run = wandb.init(
                        project="turbulence_swin",  # Same project as training
                        name=f"evaluation_{log_dir_name}",
                        tags=["evaluation", "flow", "swin", "2d"],
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
                    project="turbulence_swin",  # Same project as training
                    name=f"evaluation_{log_dir_name}",
                    tags=["evaluation", "flow", "swin", "2d"],
                    config={
                        "checkpoint_path": checkpoint_path,
                        "device": str(self.device),
                        "log_dir": log_dir_name,
                        "evaluation_type": "post_training",
                    },
                )
        else:
            self.wandb_run = None

    def _load_model(self) -> nn.Module:
        """Load the model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Extract hyperparameters from checkpoint
        if "hyper_parameters" in checkpoint:
            print("Found hyperparameters in checkpoint")
        else:
            # Use default parameters based on config
            print("Using default parameters")

        # Create model with parameters from config
        print("Instantiating model from config...")
        model = hydra.utils.instantiate(self.model_cfg)

        # Load state dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'net.' prefix if present
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("net."):
                    new_key = key[4:]  # Remove 'net.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            model.load_state_dict(new_state_dict)
            print("Model weights loaded successfully!")
        else:
            print("No state_dict found in checkpoint")

        model.to(self.device)
        return model

    def _extract_log_dir_name(self) -> str:
        """Extract log directory name from checkpoint path."""
        # Extract the run directory name from checkpoint path
        # Example: /logs/flow_swin_2d/runs/2025-08-03_18-52-37-797221/checkpoints/last.ckpt
        checkpoint_path = Path(self.checkpoint_path)
        run_dir = checkpoint_path.parent.parent.name  # Get the run directory name
        return run_dir

    def _extract_wandb_run_id(self) -> str:
        """Extract wandb run ID from the training logs."""
        # Look for wandb run ID in the log directory
        checkpoint_path = Path(self.checkpoint_path)
        run_dir = checkpoint_path.parent.parent
        wandb_dir = run_dir / "wandb"

        if wandb_dir.exists():
            # Look for the latest-run symlink or run directories
            latest_run_link = wandb_dir / "latest-run"
            if latest_run_link.exists() and latest_run_link.is_symlink():
                run_name = latest_run_link.readlink().name
                # Extract run ID from run name like "run-20250806_122524-h77kz7la"
                if run_name.startswith("run-") and "-" in run_name:
                    return run_name.split("-")[-1]  # Get the last part (run ID)

            # Fallback: look for run directories directly
            for item in wandb_dir.iterdir():
                if item.is_dir() and item.name.startswith("run-"):
                    run_id = item.name.split("-")[-1]
                    return run_id

        return None

    def _setup_test_dataset(self):
        """Setup test dataset."""
        print("Setting up test dataset...")

        data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"

        test_dataset = FastFlowSequence2DDataset(
            data_dir=data_dir,
            input_length=3,
            max_k_steps=1,  # For evaluation, we only need single frame targets
            field_name="u",
            file_pattern="*.h5",
            resolution_scale=[2, 3, 1],
            y_slice=5,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="test",
        )

        print(f"Test dataset loaded with {len(test_dataset)} samples")
        return test_dataset

    def _setup_train_dataset(self):
        """Setup train dataset."""
        print("Setting up train dataset...")

        data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"

        train_dataset = FastFlowSequence2DDataset(
            data_dir=data_dir,
            input_length=3,
            max_k_steps=1,  # For evaluation, we only need single frame targets
            field_name="u",
            file_pattern="*.h5",
            resolution_scale=[2, 3, 1],
            y_slice=5,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="train",
        )

        print(f"Train dataset loaded with {len(train_dataset)} samples")
        return train_dataset

    def _setup_val_dataset(self):
        """Setup validation dataset."""
        print("Setting up validation dataset...")

        data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"

        val_dataset = FastFlowSequence2DDataset(
            data_dir=data_dir,
            input_length=3,
            max_k_steps=1,  # For evaluation, we only need single frame targets
            field_name="u",
            file_pattern="*.h5",
            resolution_scale=[2, 3, 1],
            y_slice=5,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="val",
        )

        print(f"Validation dataset loaded with {len(val_dataset)} samples")
        return val_dataset

    def _log_to_wandb(self, key: str, value, step: int = None):
        """Log metrics or images to wandb if available."""
        if self.wandb_run is not None:
            if step is not None:
                wandb.log({key: value}, step=step)
            else:
                wandb.log({key: value})

    def _log_image_to_wandb(self, key: str, image_path: str, caption: str = ""):
        """Log an image file to wandb if available."""
        if self.wandb_run is not None:
            wandb.log({key: wandb.Image(str(image_path), caption=caption)})

    def evaluate_model(self, num_samples: int = 100) -> dict[str, float]:
        """Evaluate the model on test data.

        Args:
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"Evaluating model on {num_samples} samples...")

        # Custom collate function to handle dictionaries
        def collate_fn(batch):
            # Extract input sequences and targets from batch
            input_seqs = []
            targets = []
            for item in batch:
                # Remove the extra batch dimension that the dataset adds
                input_seq = item["data"]["input_seq"].squeeze(0)  # (input_length, 1, H, W)
                target_seq = item["label"].squeeze(0)  # (max_k_steps, 1, H, W)
                # For evaluation, take only the first target frame
                target = target_seq[0]  # (1, H, W)
                input_seqs.append(input_seq)
                targets.append(target)

            return torch.stack(input_seqs), torch.stack(targets)

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn
        )

        metrics = {"mse": 0.0, "mae": 0.0, "rel_error": 0.0, "count": 0}

        with torch.no_grad():
            for _i, (input_seq, target) in enumerate(dataloader):
                if metrics["count"] >= num_samples:
                    break

                input_seq = input_seq.to(self.device)
                target = target.to(self.device)

                # Forward pass
                pred = self.model(input_seq)

                # Calculate metrics
                mse = torch.nn.functional.mse_loss(pred, target)
                mae = torch.nn.functional.l1_loss(pred, target)
                rel_error = torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8))

                batch_size = target.shape[0]
                metrics["mse"] += mse.item() * batch_size
                metrics["mae"] += mae.item() * batch_size
                metrics["rel_error"] += rel_error.item() * batch_size
                metrics["count"] += batch_size

        # Average metrics
        for key in ["mse", "mae", "rel_error"]:
            metrics[key] /= metrics["count"]

        print(f"Test Results (on {metrics['count']} samples):")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Rel Error: {metrics['rel_error']:.6f}")

        # Log metrics to wandb
        self._log_to_wandb("test/mse", metrics["mse"])
        self._log_to_wandb("test/mae", metrics["mae"])
        self._log_to_wandb("test/relative_error", metrics["rel_error"])
        self._log_to_wandb("test/num_samples", metrics["count"])

        return metrics

    def generate_sequence_prediction(self, input_seq: torch.Tensor, num_predictions: int = 10) -> torch.Tensor:
        """Generate autoregressive predictions.

        Args:
            input_seq: Input sequence of shape (B, T, C, H, W)
            num_predictions: Number of future timesteps to predict

        Returns:
            Predicted sequence of shape (B, num_predictions, C, H, W)
        """
        predictions = []
        current_seq = input_seq.clone()

        with torch.no_grad():
            for i in range(num_predictions):
                # Predict next timestep
                next_pred = self.model(current_seq)  # Model output shape

                # Debug: Check shapes on first iteration
                if i == 0:
                    print(f"Input sequence shape: {current_seq.shape}")
                    print(f"Model output shape: {next_pred.shape}")
                    print(f"Input sequence data range: [{current_seq.min():.4f}, {current_seq.max():.4f}]")
                    print(f"Model output data range: [{next_pred.min():.4f}, {next_pred.max():.4f}]")

                # Ensure next_pred has the right shape for concatenation
                if len(next_pred.shape) == 4:  # (B, C, H, W)
                    next_pred = next_pred.unsqueeze(1)  # (B, 1, C, H, W)
                elif len(next_pred.shape) == 5 and next_pred.shape[1] != 1:
                    # If it has multiple timesteps, take the last one
                    next_pred = next_pred[:, -1:, :, :, :]  # (B, 1, C, H, W)

                predictions.append(next_pred)

                # Update sequence for next prediction
                current_seq = torch.cat([current_seq[:, 1:], next_pred], dim=1)

        return torch.cat(predictions, dim=1)  # (B, num_predictions, C, H, W)

    def visualize_sample_prediction(self, sample_idx: int = 0, num_future: int = 30):
        """Visualize a sample prediction with ground truth comparison."""
        print(f"Visualizing sample {sample_idx}...")

        # Get multiple consecutive samples to get ground truth for future timesteps
        ground_truth_frames = []
        for i in range(num_future + 1):
            if sample_idx + i < len(self.test_dataset):
                sample = self.test_dataset[sample_idx + i]
                target_seq = sample["label"].cpu().numpy()[0]  # (max_k_steps, C, H, W)
                # Take the first target frame
                target = target_seq[0]  # (C, H, W)
                ground_truth_frames.append(target[0])  # Take channel 0

                # Debug: Print shapes for first few frames
                if i < 3:
                    print(f"Ground truth {i} shape: {target[0].shape}")
                    print(f"Ground truth {i} data range: [{target[0].min():.4f}, {target[0].max():.4f}]")
            else:
                # If we run out of samples, repeat the last one
                ground_truth_frames.append(ground_truth_frames[-1])

        # Get the initial sample for prediction
        sample = self.test_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)  # Already has batch dimension

        # Generate predictions
        pred_seq = self.generate_sequence_prediction(input_seq, num_future)

        # Move to CPU for visualization
        input_seq = input_seq.cpu().numpy()[0]  # (T, C, H, W)
        pred_seq = pred_seq.cpu().numpy()[0]  # (T_pred, C, H, W)

        # Calculate per-timestep colorbar ranges
        # Each timestep gets its own range based on truth vs prediction at that time
        timestep_ranges = {}
        timestep_error_ranges = {}

        for t in range(num_future + 1):
            if t == 0:
                # Last input timestep
                data = input_seq[-1, 0]
                pred_data = data  # Same as ground truth for input
                error = np.zeros_like(data)
            else:
                # Ground truth and predictions
                data = ground_truth_frames[t]
                pred_data = pred_seq[t - 1, 0]
                error = np.abs(data - pred_data)

            # Calculate range for this timestep (combining truth and prediction)
            timestep_vmin = min(data.min(), pred_data.min())
            timestep_vmax = max(data.max(), pred_data.max())
            timestep_ranges[t] = (timestep_vmin, timestep_vmax)

            # Calculate error range for this timestep
            timestep_error_ranges[t] = (0, error.max() if error.max() > 0 else 0.1)

        # Create figure
        fig, axes = plt.subplots(3, num_future + 1, figsize=(3 * (num_future + 1), 9))

        # Calculate and print quantitative metrics
        print("\nQuantitative Results:")
        print("Step | MSE     | MAE     | Rel Error")
        print("-" * 35)

        for t in range(num_future + 1):
            if t == 0:
                # Show last input timestep
                data = input_seq[-1, 0]  # Last input frame
                title_prefix = "Last Input"
                pred_data = data  # Same as ground truth for input
                error = np.zeros_like(data)
            else:
                # Show ground truth and predictions
                data = ground_truth_frames[t]  # Ground truth for t+t
                pred_data = pred_seq[t - 1, 0]  # Prediction for t+t
                error = np.abs(data - pred_data)
                title_prefix = f"True t+{t}"

                # Calculate metrics for this timestep
                mse = np.mean(error**2)
                mae = np.mean(error)
                rel_error = np.mean(error / (np.abs(data) + 1e-8))
                print(f"t+{t:2d} | {mse:.5f} | {mae:.5f} | {rel_error:.5f}")

            # Get timestep-specific ranges
            vmin, vmax = timestep_ranges[t]
            error_vmin, error_vmax = timestep_error_ranges[t]

            # Ground truth (using timestep-specific colorbar range)
            im1 = axes[0, t].imshow(data, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            axes[0, t].set_title(title_prefix)
            axes[0, t].axis("off")
            plt.colorbar(im1, ax=axes[0, t], fraction=0.046, pad=0.04)

            # Prediction (using same timestep-specific colorbar range)
            im2 = axes[1, t].imshow(pred_data, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            axes[1, t].set_title(f"Pred t+{t}" if t > 0 else "Last Input")
            axes[1, t].axis("off")
            plt.colorbar(im2, ax=axes[1, t], fraction=0.046, pad=0.04)

            # Error (using timestep-specific error range)
            im3 = axes[2, t].imshow(error, cmap="Reds", aspect="auto", vmin=error_vmin, vmax=error_vmax)
            if t > 0:
                axes[2, t].set_title(f"Error t+{t} (MAE: {np.mean(error):.4f})")
            else:
                axes[2, t].set_title("Error (0)")
            axes[2, t].axis("off")
            plt.colorbar(im3, ax=axes[2, t], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save figure
        save_path = self.output_dir / f"sample_{sample_idx}_prediction.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
        print("Per-timestep ranges used (allows better temporal visualization):")

        # Log image to wandb
        self._log_image_to_wandb(
            f"test/sample_{sample_idx}_prediction",
            save_path,
            f"Autoregressive prediction for sample {sample_idx} over {num_future} timesteps",
        )

        plt.show()

    def create_prediction_video(self, sample_idx: int = 0, num_future: int = 30):
        """Create a video showing autoregressive prediction vs ground truth."""
        print(f"Creating video for sample {sample_idx}...")

        # Get ground truth frames for the future timesteps
        ground_truth_frames = []
        for i in range(num_future + 5):  # Get input + future frames
            if sample_idx + i < len(self.test_dataset):
                sample = self.test_dataset[sample_idx + i]
                input_seq = sample["data"]["input_seq"].cpu().numpy()[0]  # (T, C, H, W)
                target_seq = sample["label"].cpu().numpy()[0]  # (max_k_steps, C, H, W)
                # Take the first target frame
                target = target_seq[0]  # (C, H, W)

                if i == 0:
                    # For first sample, add all input frames
                    for j in range(input_seq.shape[0]):
                        ground_truth_frames.append(input_seq[j, 0])
                # Add the target frame
                ground_truth_frames.append(target[0])
            else:
                # If we run out of samples, repeat the last one
                if ground_truth_frames:
                    ground_truth_frames.append(ground_truth_frames[-1])

        print(f"Collected {len(ground_truth_frames)} ground truth frames")

        # Get the initial sample for prediction
        sample = self.test_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)

        # Generate predictions
        pred_seq = self.generate_sequence_prediction(input_seq, num_future)

        # Move to CPU
        input_seq = input_seq.cpu().numpy()[0]  # (T, C, H, W)
        pred_seq = pred_seq.cpu().numpy()[0]  # (T_pred, C, H, W)

        # Combine input sequence and predictions
        pred_full_sequence = np.concatenate([input_seq, pred_seq], axis=0)
        input_len = input_seq.shape[0]

        # Trim ground truth to match prediction sequence length
        ground_truth_frames = ground_truth_frames[: len(pred_full_sequence)]

        # Calculate per-frame colorbar ranges for dynamic visualization
        frame_ranges = {}
        frame_error_ranges = {}

        for frame_idx in range(len(ground_truth_frames)):
            truth_data = ground_truth_frames[frame_idx]
            pred_data = pred_full_sequence[frame_idx, 0]
            error = np.abs(truth_data - pred_data)

            # Calculate range for this frame (combining truth and prediction)
            frame_vmin = min(truth_data.min(), pred_data.min())
            frame_vmax = max(truth_data.max(), pred_data.max())
            frame_ranges[frame_idx] = (frame_vmin, frame_vmax)

            # Calculate error range for this frame
            frame_error_ranges[frame_idx] = (0, error.max() if error.max() > 0 else 0.1)

        print("Video using dynamic per-frame color scaling for better temporal visualization")

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Initialize plots with first frame ranges
        frame_0_vmin, frame_0_vmax = frame_ranges[0]
        frame_0_error_vmin, frame_0_error_vmax = frame_error_ranges[0]

        im1 = axes[0].imshow(
            ground_truth_frames[0], cmap="viridis", aspect="auto", vmin=frame_0_vmin, vmax=frame_0_vmax
        )
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(
            pred_full_sequence[0, 0], cmap="viridis", aspect="auto", vmin=frame_0_vmin, vmax=frame_0_vmax
        )
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Error plot with first frame error range
        initial_error = np.abs(ground_truth_frames[0] - pred_full_sequence[0, 0])
        im3 = axes[2].imshow(
            initial_error, cmap="Reds", aspect="auto", vmin=frame_0_error_vmin, vmax=frame_0_error_vmax
        )
        axes[2].set_title("Absolute Error")
        axes[2].axis("off")
        cb3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Time text
        time_text = fig.suptitle("Timestep: 0 (Input)", fontsize=16)

        def animate(frame):
            # Get frame-specific ranges for dynamic color scaling
            frame_vmin, frame_vmax = frame_ranges[frame]
            frame_error_vmin, frame_error_vmax = frame_error_ranges[frame]

            # Update ground truth with dynamic per-frame ranges
            truth_data = ground_truth_frames[frame].copy()
            im1.set_array(truth_data)
            im1.set_clim(vmin=frame_vmin, vmax=frame_vmax)

            # Update prediction with SAME per-frame ranges (crucial for comparison)
            pred_data = pred_full_sequence[frame, 0].copy()
            im2.set_array(pred_data)
            im2.set_clim(vmin=frame_vmin, vmax=frame_vmax)  # Same range as ground truth

            # Update error with frame-specific error range
            error = np.abs(truth_data - pred_data)
            im3.set_array(error.copy())
            im3.set_clim(vmin=frame_error_vmin, vmax=frame_error_vmax)

            # Update colorbars to reflect the new ranges
            cb1.set_clim(vmin=frame_vmin, vmax=frame_vmax)
            cb2.set_clim(vmin=frame_vmin, vmax=frame_vmax)  # Same range as cb1
            cb3.set_clim(vmin=frame_error_vmin, vmax=frame_error_vmax)

            # Update title with range information
            if frame < input_len:
                time_text.set_text(f"Timestep: {frame} (Input) - Range: [{frame_vmin:.3f}, {frame_vmax:.3f}]")
            else:
                pred_step = frame - input_len + 1
                mae = np.mean(error)
                time_text.set_text(
                    f"Timestep: {frame} (Prediction t+{pred_step}) - MAE: {mae:.4f} - "
                    f"Range: [{frame_vmin:.3f}, {frame_vmax:.3f}]"
                )

            # Don't return colorbars since they cause issues with blit
            return [im1, im2, im3]

        # Create animation
        total_frames = len(ground_truth_frames)
        print(f"Creating animation with {total_frames} frames")

        if total_frames == 0:
            print("Error: No frames to animate!")
            return

        # Use a simpler animation approach to avoid matplotlib issues
        anim = animation.FuncAnimation(fig, animate, frames=range(total_frames), interval=500, blit=False, repeat=False)

        # Save video in multiple formats
        video_path_mp4 = self.output_dir / f"sample_{sample_idx}_evolution.mp4"
        video_path_gif = self.output_dir / f"sample_{sample_idx}_evolution.gif"

        # Save as GIF first (always works)
        try:
            print("Saving animation as GIF...")
            anim.save(video_path_gif, writer="pillow", fps=2)
            print(f"Video saved as GIF: {video_path_gif}")

            # Log GIF to wandb
            if self.wandb_run is not None:
                self._log_to_wandb(
                    f"test/sample_{sample_idx}_evolution", wandb.Video(str(video_path_gif), fps=2, format="gif")
                )
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print("Trying alternative approach...")
            # Alternative approach: manually create frames
            try:
                import io

                from PIL import Image

                frames = []
                for frame_idx in range(total_frames):
                    animate(frame_idx)  # Update the plot
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    frames.append(Image.open(buf))

                # Save as GIF using PIL
                frames[0].save(video_path_gif, save_all=True, append_images=frames[1:], duration=500, loop=0)
                print(f"Video saved as GIF using manual approach: {video_path_gif}")
            except Exception as e2:
                print(f"Manual approach also failed: {e2}")
                return

        # Try to convert GIF to MP4 using imageio (alternative to ffmpeg)
        try:
            import imageio

            # Read GIF frames
            gif_reader = imageio.get_reader(video_path_gif)
            frames = []
            for frame in gif_reader:
                frames.append(frame)

            # Save as MP4
            imageio.mimsave(video_path_mp4, frames, fps=2, codec="libx264")
            print(f"Video converted and saved as MP4: {video_path_mp4}")

        except ImportError:
            print("imageio not available for MP4 conversion")
        except Exception as e:
            print(f"Failed to convert to MP4: {e}")

            # Alternative: try with matplotlib's HTML writer to create a web-viewable video
            try:
                video_path_html = self.output_dir / f"sample_{sample_idx}_evolution.html"
                anim.save(video_path_html, writer="html", fps=2)
                print(f"Video saved as HTML: {video_path_html}")
            except Exception as e2:
                print(f"HTML fallback also failed: {e2}")

        plt.show()

    def evaluate_with_teacher_forcing(self, split: str = "train", sample_idx: int = 0, num_future: int = 30):
        """Evaluate using teacher forcing (ground truth as input) for train/val data.

        Args:
            split: "train" or "val"
            sample_idx: Starting sample index
            num_future: Number of future timesteps to predict
        """
        print(f"Evaluating {split} data with teacher forcing...")

        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        else:
            raise ValueError("Only 'train' and 'val' splits supported for teacher forcing")

        # Collect ground truth frames and predictions
        ground_truth_frames = []
        predictions = []

        for i in range(num_future + 1):
            if sample_idx + i < len(dataset):
                sample = dataset[sample_idx + i]
                input_seq = sample["data"]["input_seq"].to(self.device)  # (1, T, C, H, W)
                target_seq = sample["label"].to(self.device)  # (1, max_k_steps, C, H, W)
                # Take the first target frame
                target = target_seq[:, 0]  # (1, C, H, W)

                # Store ground truth
                ground_truth_frames.append(target.cpu().numpy()[0, 0])  # (H, W)

                # Predict using ground truth input (teacher forcing)
                with torch.no_grad():
                    pred = self.model(input_seq)
                    predictions.append(pred.cpu().numpy()[0, 0])  # (H, W)
            else:
                break

        return ground_truth_frames, predictions

    def visualize_teacher_forcing(self, split: str = "train", sample_idx: int = 0, num_future: int = 30):
        """Visualize teacher forcing results for train/val data."""
        print(f"Visualizing {split} data with teacher forcing...")

        ground_truth_frames, predictions = self.evaluate_with_teacher_forcing(split, sample_idx, num_future)

        # Create figure
        fig, axes = plt.subplots(3, min(len(predictions), 6), figsize=(18, 9))
        if len(predictions) == 1:
            axes = axes.reshape(3, 1)

        # Calculate and print quantitative metrics
        print(f"\n{split.upper()} Teacher Forcing Results:")
        print("Step | MSE     | MAE     | Rel Error")
        print("-" * 35)

        # Calculate per-timestep colorbar ranges for teacher forcing visualization
        num_display = min(len(predictions), 6)
        tf_timestep_ranges = {}
        tf_timestep_error_ranges = {}

        for t in range(num_display):
            data = ground_truth_frames[t]
            pred_data = predictions[t]
            error = np.abs(data - pred_data)

            # Calculate range for this timestep (combining truth and prediction)
            timestep_vmin = min(data.min(), pred_data.min())
            timestep_vmax = max(data.max(), pred_data.max())
            tf_timestep_ranges[t] = (timestep_vmin, timestep_vmax)

            # Calculate error range for this timestep
            tf_timestep_error_ranges[t] = (0, error.max() if error.max() > 0 else 0.1)

        for t in range(num_display):
            data = ground_truth_frames[t]
            pred_data = predictions[t]
            error = np.abs(data - pred_data)

            # Calculate metrics
            mse = np.mean(error**2)
            mae = np.mean(error)
            rel_error = np.mean(error / (np.abs(data) + 1e-8))
            print(f"t+{t + 1:2d} | {mse:.5f} | {mae:.5f} | {rel_error:.5f}")

            # Get timestep-specific ranges
            tf_vmin, tf_vmax = tf_timestep_ranges[t]
            tf_error_vmin, tf_error_vmax = tf_timestep_error_ranges[t]

            # Ground truth (using timestep-specific colorbar range)
            im1 = axes[0, t].imshow(data, cmap="viridis", aspect="auto", vmin=tf_vmin, vmax=tf_vmax)
            axes[0, t].set_title(f"True t+{t + 1}")
            axes[0, t].axis("off")
            plt.colorbar(im1, ax=axes[0, t], fraction=0.046, pad=0.04)

            # Prediction (using same timestep-specific colorbar range)
            im2 = axes[1, t].imshow(pred_data, cmap="viridis", aspect="auto", vmin=tf_vmin, vmax=tf_vmax)
            axes[1, t].set_title(f"Pred t+{t + 1} (TF)")
            axes[1, t].axis("off")
            plt.colorbar(im2, ax=axes[1, t], fraction=0.046, pad=0.04)

            # Error (using timestep-specific error range)
            im3 = axes[2, t].imshow(error, cmap="Reds", aspect="auto", vmin=tf_error_vmin, vmax=tf_error_vmax)
            axes[2, t].set_title(f"Error t+{t + 1} (MAE: {mae:.4f})")
            axes[2, t].axis("off")
            plt.colorbar(im3, ax=axes[2, t], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save figure
        save_path = self.output_dir / f"{split}_teacher_forcing_prediction.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Teacher forcing visualization saved to {save_path}")

        # Log image to wandb
        self._log_image_to_wandb(
            f"{split}/teacher_forcing_prediction",
            save_path,
            f"Teacher forcing prediction for {split} data over {num_future} timesteps",
        )

        plt.show()

    def create_teacher_forcing_video(self, split: str = "train", sample_idx: int = 0, num_future: int = 30):
        """Create video showing teacher forcing results."""
        print(f"Creating {split} teacher forcing video...")

        ground_truth_frames, predictions = self.evaluate_with_teacher_forcing(split, sample_idx, num_future)

        # Calculate per-frame colorbar ranges for teacher forcing video dynamic visualization
        tf_frame_ranges = {}
        tf_frame_error_ranges = {}

        for frame_idx in range(len(ground_truth_frames)):
            truth_data = ground_truth_frames[frame_idx]
            pred_data = predictions[frame_idx]
            error = np.abs(truth_data - pred_data)

            # Calculate range for this frame (combining truth and prediction)
            frame_vmin = min(truth_data.min(), pred_data.min())
            frame_vmax = max(truth_data.max(), pred_data.max())
            tf_frame_ranges[frame_idx] = (frame_vmin, frame_vmax)

            # Calculate error range for this frame
            tf_frame_error_ranges[frame_idx] = (0, error.max() if error.max() > 0 else 0.1)

        print("Teacher forcing video using dynamic per-frame color scaling")

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Initialize plots with first frame ranges
        tf_frame_0_vmin, tf_frame_0_vmax = tf_frame_ranges[0]
        tf_frame_0_error_vmin, tf_frame_0_error_vmax = tf_frame_error_ranges[0]

        im1 = axes[0].imshow(
            ground_truth_frames[0], cmap="viridis", aspect="auto", vmin=tf_frame_0_vmin, vmax=tf_frame_0_vmax
        )
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        tf_cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(predictions[0], cmap="viridis", aspect="auto", vmin=tf_frame_0_vmin, vmax=tf_frame_0_vmax)
        axes[1].set_title("Prediction (Teacher Forcing)")
        axes[1].axis("off")
        tf_cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Error plot with first frame error range
        initial_error = np.abs(ground_truth_frames[0] - predictions[0])
        im3 = axes[2].imshow(
            initial_error, cmap="Reds", aspect="auto", vmin=tf_frame_0_error_vmin, vmax=tf_frame_0_error_vmax
        )
        axes[2].set_title("Absolute Error")
        axes[2].axis("off")
        tf_cb3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Time text
        time_text = fig.suptitle("Timestep: 1 (Teacher Forcing)", fontsize=16)

        def animate(frame):
            # Get frame-specific ranges for dynamic color scaling
            tf_frame_vmin, tf_frame_vmax = tf_frame_ranges[frame]
            tf_frame_error_vmin, tf_frame_error_vmax = tf_frame_error_ranges[frame]

            # Update ground truth with dynamic per-frame ranges
            im1.set_array(ground_truth_frames[frame])
            im1.set_clim(vmin=tf_frame_vmin, vmax=tf_frame_vmax)

            # Update prediction with SAME per-frame ranges (crucial for comparison)
            im2.set_array(predictions[frame])
            im2.set_clim(vmin=tf_frame_vmin, vmax=tf_frame_vmax)  # Same range as ground truth

            # Update error with frame-specific error range
            error = np.abs(ground_truth_frames[frame] - predictions[frame])
            im3.set_array(error)
            im3.set_clim(vmin=tf_frame_error_vmin, vmax=tf_frame_error_vmax)

            # Update colorbars to reflect the new ranges
            tf_cb1.set_clim(vmin=tf_frame_vmin, vmax=tf_frame_vmax)
            tf_cb2.set_clim(vmin=tf_frame_vmin, vmax=tf_frame_vmax)  # Same range as tf_cb1
            tf_cb3.set_clim(vmin=tf_frame_error_vmin, vmax=tf_frame_error_vmax)

            # Update title with range information
            mae = np.mean(error)
            time_text.set_text(
                f"Timestep: {frame + 1} (Teacher Forcing) - MAE: {mae:.4f} - "
                f"Range: [{tf_frame_vmin:.3f}, {tf_frame_vmax:.3f}]"
            )

            # Don't return colorbars since they cause issues with blit
            return [im1, im2, im3]

        # Create animation
        total_frames = len(ground_truth_frames)
        print(f"Creating teacher forcing animation with {total_frames} frames")

        if total_frames == 0:
            print("Error: No frames to animate!")
            return

        # Use a simpler animation approach to avoid matplotlib issues
        anim = animation.FuncAnimation(fig, animate, frames=range(total_frames), interval=500, blit=False, repeat=False)

        # Save video in multiple formats
        video_path_mp4 = self.output_dir / f"{split}_teacher_forcing_evolution.mp4"
        video_path_gif = self.output_dir / f"{split}_teacher_forcing_evolution.gif"

        # Save as GIF first (always works)
        try:
            print("Saving teacher forcing animation as GIF...")
            anim.save(video_path_gif, writer="pillow", fps=2)
            print(f"Video saved as GIF: {video_path_gif}")

            # Log GIF to wandb
            if self.wandb_run is not None:
                self._log_to_wandb(
                    f"{split}/teacher_forcing_evolution", wandb.Video(str(video_path_gif), fps=2, format="gif")
                )
        except Exception as e:
            print(f"Error saving teacher forcing GIF: {e}")
            print("Trying alternative approach...")
            # Alternative approach: manually create frames
            try:
                import io

                from PIL import Image

                frames = []
                for frame_idx in range(total_frames):
                    animate(frame_idx)  # Update the plot
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    frames.append(Image.open(buf))

                # Save as GIF using PIL
                frames[0].save(video_path_gif, save_all=True, append_images=frames[1:], duration=500, loop=0)
                print(f"Teacher forcing video saved as GIF using manual approach: {video_path_gif}")
            except Exception as e2:
                print(f"Teacher forcing manual approach also failed: {e2}")
                return

        # Try to convert GIF to MP4 using imageio
        try:
            import imageio

            # Read GIF frames
            gif_reader = imageio.get_reader(video_path_gif)
            frames = []
            for frame in gif_reader:
                frames.append(frame)

            # Save as MP4
            imageio.mimsave(video_path_mp4, frames, fps=2, codec="libx264")
            print(f"Video converted and saved as MP4: {video_path_mp4}")

        except Exception as e:
            print(f"Failed to convert to MP4: {e}")

        plt.show()

    def run_evaluation(self):
        """Run complete evaluation for all data splits."""
        print("Starting comprehensive evaluation...")

        # Evaluate test data (autoregressive)
        print("\n" + "=" * 60)
        print("TEST DATA EVALUATION (Autoregressive)")
        print("=" * 60)
        metrics = self.evaluate_model(num_samples=50)

        # Save test results
        results_path = self.output_dir / "evaluation_results.txt"
        with open(results_path, "w") as f:
            f.write("Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("TEST DATA (Autoregressive):\n")
            f.write(f"MSE: {metrics['mse']:.6f}\n")
            f.write(f"MAE: {metrics['mae']:.6f}\n")
            f.write(f"Relative Error: {metrics['rel_error']:.6f}\n")
            f.write(f"Samples: {metrics['count']}\n\n")

        # Generate test visualizations (autoregressive)
        self.visualize_sample_prediction(sample_idx=0, num_future=30)
        self.create_prediction_video(sample_idx=0, num_future=30)

        # Evaluate training data (teacher forcing)
        print("\n" + "=" * 60)
        print("TRAINING DATA EVALUATION (Teacher Forcing)")
        print("=" * 60)
        self.visualize_teacher_forcing(split="train", sample_idx=0, num_future=30)
        self.create_teacher_forcing_video(split="train", sample_idx=0, num_future=30)

        # Evaluate validation data (teacher forcing)
        print("\n" + "=" * 60)
        print("VALIDATION DATA EVALUATION (Teacher Forcing)")
        print("=" * 60)
        self.visualize_teacher_forcing(split="val", sample_idx=0, num_future=30)
        self.create_teacher_forcing_video(split="val", sample_idx=0, num_future=30)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print("- Test (autoregressive): sample_0_prediction.png, sample_0_evolution.mp4")
        print("- Train (teacher forcing): train_teacher_forcing_prediction.png, train_teacher_forcing_evolution.mp4")
        print("- Val (teacher forcing): val_teacher_forcing_prediction.png, val_teacher_forcing_evolution.mp4")

        # Log summary metrics to wandb
        self._log_to_wandb("evaluation/status", "completed")

    def close_wandb(self):
        """Close wandb run."""
        if self.wandb_run is not None:
            wandb.finish()
            print("Wandb run closed.")


def main():
    """Main function."""
    import argparse

    # Parse command line arguments BEFORE Hydra initialization
    parser = argparse.ArgumentParser(description="Evaluate Flow Swin 2D model")
    parser.add_argument("checkpoint_path", nargs="?", default=None, help="Path to model checkpoint")

    # Parse known args to separate our checkpoint path from Hydra overrides
    args, hydra_overrides = parser.parse_known_args()

    # Use command line argument or default path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Default to the hardcoded path if no argument provided
        checkpoint_path = (
            "/home/sh/CB/icon-thewell-dev/logs/flow_swin_2d/runs/2025-08-11_22-40-39-792051/checkpoints/last.ckpt"
        )

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Initialize Hydra with the remaining overrides
    with hydra.initialize(version_base="1.3", config_path="configs"):
        cfg = hydra.compose(config_name="train_flow_swin_2d", overrides=hydra_overrides)

        evaluator = SimpleModelEvaluator(checkpoint_path, cfg.model)
        try:
            evaluator.run_evaluation()
        finally:
            evaluator.close_wandb()  # Ensure wandb is properly closed


if __name__ == "__main__":
    main()
