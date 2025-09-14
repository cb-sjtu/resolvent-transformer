#!/usr/bin/env python3
"""
Simple evaluation script for 2D Flow Swin Transformer implementation.
Loads the best model checkpoint and generates visualizations and videos.
"""

import os
import sys
import warnings
from pathlib import Path

import h5py
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

    def __init__(self, checkpoint_path: str, model_cfg: DictConfig, save_predictions: bool = False):
        """Initialize the evaluator.

        Args:
            checkpoint_path: Path to the model checkpoint
            model_cfg: Model configuration from Hydra
            save_predictions: Whether to save prediction results as H5 files
        """
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_predictions = save_predictions

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

        # Create predictions save directory if needed
        if self.save_predictions:
            # Save predictions in the same logs directory as the checkpoint
            checkpoint_path = Path(self.checkpoint_path)
            self.logs_dir = checkpoint_path.parent.parent  # logs/flow_swin_2d/runs/{run_name}
            self.predictions_dir = self.logs_dir / "predictions"
            self.predictions_dir.mkdir(exist_ok=True, parents=True)
            print(f"Predictions will be saved to: {self.predictions_dir}")
        else:
            self.predictions_dir = None

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
            module_cfg = OmegaConf.create({"model": self.model_cfg, "loss_fn": "mse"})
            model = FlowSwin2DLitModule(module_cfg)

            # Load the state dict manually
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                print("Model weights loaded successfully!")

        model.eval()  # Set to evaluation mode
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
            field_names=["u", "v", "w"],  # Use 3-channel uvw data
            file_pattern="*.h5",
            resolution_scale=[2, 3, 1],
            y_slice=5,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="test",
            norm_stats="norm_stats_u-v-w_scale2-3-1_yslice5.json",
            enable_normalization=True,
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
            field_names=["u", "v", "w"],  # Use 3-channel uvw data
            file_pattern="*.h5",
            resolution_scale=[2, 3, 1],
            y_slice=5,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="train",
            norm_stats="norm_stats_u-v-w_scale2-3-1_yslice5.json",
            enable_normalization=True,
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
            field_names=["u", "v", "w"],  # Use 3-channel uvw data
            file_pattern="*.h5",
            resolution_scale=[2, 3, 1],
            y_slice=5,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="val",
            norm_stats="norm_stats_u-v-w_scale2-3-1_yslice5.json",
            enable_normalization=True,
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

    def _compute_velocity_magnitude(self, velocity_data: np.ndarray) -> np.ndarray:
        """Compute velocity magnitude from u, v, w components.

        Args:
            velocity_data: Velocity data with shape (C, H, W) where C >= 3 for u, v, w

        Returns:
            Velocity magnitude with shape (H, W)
        """
        if velocity_data.ndim != 3 or velocity_data.shape[0] < 3:
            raise ValueError(f"Expected 3D array with at least 3 channels, got shape {velocity_data.shape}")

        u, v, w = velocity_data[0], velocity_data[1], velocity_data[2]
        magnitude = np.sqrt(u**2 + v**2 + w**2)
        return magnitude

    def _compute_smart_relative_error(self, pred, target, channel_names=None):
        """Compute smart relative error that handles different channel magnitudes properly.

        Args:
            pred: Predicted tensor (C, H, W) or (..., C, H, W)
            target: Target tensor (same shape as pred)
            channel_names: List of channel names for reference

        Returns:
            dict with per-channel and overall relative errors
        """
        if channel_names is None:
            channel_names = ["u", "v", "w"]

        # Convert to torch tensors if needed
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        # Ensure we have the right shapes
        if len(pred.shape) > 3:
            pred = pred.view(-1, *pred.shape[-3:])  # Flatten to (N, C, H, W)
            target = target.view(-1, *target.shape[-3:])
            pred = pred.mean(0)  # Average over batch dimension -> (C, H, W)
            target = target.mean(0)

        results = {}

        for c in range(min(pred.shape[0], len(channel_names))):
            channel_name = channel_names[c]
            pred_c = pred[c]
            target_c = target[c]

            # Compute absolute error
            abs_error = torch.abs(pred_c - target_c)

            # Compute different types of relative error
            # 1. Traditional relative error (can be problematic for small values)
            traditional_rel_error = torch.mean(abs_error / (torch.abs(target_c) + 1e-8))

            # 2. RMS-normalized relative error (more stable for small values)
            target_rms = torch.sqrt(torch.mean(target_c**2))
            rms_rel_error = torch.mean(abs_error) / (target_rms + 1e-8)

            # 3. Range-normalized relative error
            target_range = torch.max(target_c) - torch.min(target_c)
            range_rel_error = torch.mean(abs_error) / (target_range + 1e-8)

            results[channel_name] = {
                "traditional": traditional_rel_error.item(),
                "rms_normalized": rms_rel_error.item(),
                "range_normalized": range_rel_error.item(),
                "mae": torch.mean(abs_error).item(),
                "mse": torch.mean(abs_error**2).item(),
            }

        # Overall relative error using L2 norm (consistent with training)
        pred_flat = pred.flatten(start_dim=1)  # (C, H*W)
        target_flat = target.flatten(start_dim=1)
        target_norm = torch.norm(target_flat, dim=1, keepdim=True)
        error_norm = torch.norm(pred_flat - target_flat, dim=1, keepdim=True)
        overall_rel_error = (error_norm / (target_norm + 1e-8)).mean()

        results["overall"] = {"l2_normalized": overall_rel_error.item()}

        return results

    def _save_prediction_as_h5(self, prediction: np.ndarray, split: str, sample_idx: int, timestep: int):
        """Save prediction result as H5 file in the same format as original data.

        Args:
            prediction: Prediction data array with shape (C, H, W) for 3-channel or (H, W) for single channel
            split: Data split ('train', 'val', 'test')
            sample_idx: Sample index
            timestep: Timestep number (1-based)
        """
        if not self.save_predictions or self.predictions_dir is None:
            return

        # Create split directory
        split_dir = self.predictions_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)

        # Handle multi-channel data (C, H, W)
        if prediction.ndim == 3 and prediction.shape[0] == 3:
            # Save 3-channel data as combined file
            filename = f"pred_u-v-w_scale2-3-1_yslice5_s{sample_idx:05d}_t{timestep:05d}.h5"
            filepath = split_dir / filename

            with h5py.File(filepath, "w") as f:
                data_to_save = prediction.astype(np.float32)  # (C, H, W)
                f.create_dataset("data", data=data_to_save, dtype=np.float32)
                f.attrs["field_names"] = ["u", "v", "w"]
                f.attrs["num_channels"] = 3

            print(f"Saved 3-channel prediction: {filepath}")

            # Also save individual channels
            channel_names = ["u", "v", "w"]
            for i, channel_name in enumerate(channel_names):
                channel_filename = f"pred_{channel_name}_scale2-3-1_yslice5_s{sample_idx:05d}_t{timestep:05d}.h5"
                channel_filepath = split_dir / channel_filename

                with h5py.File(channel_filepath, "w") as f:
                    data_to_save = prediction[i].astype(np.float32)  # (H, W)
                    f.create_dataset("data", data=data_to_save, dtype=np.float32)
                    f.attrs["field_names"] = [channel_name]
                    f.attrs["num_channels"] = 1

                print(f"Saved {channel_name} channel: {channel_filepath}")

        else:
            # Handle legacy single channel data
            if prediction.ndim == 2:
                data_to_save = prediction.astype(np.float32)
            else:
                # If prediction has extra dimensions, take the first channel
                data_to_save = (
                    prediction[0].astype(np.float32) if prediction.ndim > 2 else prediction.astype(np.float32)
                )

            # Use legacy filename for backward compatibility
            filename = f"pred_u_scale2-3-1_yslice5_s{sample_idx:05d}_t{timestep:05d}.h5"
            filepath = split_dir / filename

            with h5py.File(filepath, "w") as f:
                f.create_dataset("data", data=data_to_save, dtype=np.float32)

            print(f"Saved legacy single-channel prediction: {filepath}")

    def _save_predictions_batch(self, predictions: np.ndarray, split: str, start_sample_idx: int, start_timestep: int):
        """Save a batch of predictions as H5 files.

        Args:
            predictions: Prediction data array with shape (batch_size, timesteps, C, H, W) or (batch_size, C, H, W)
            split: Data split ('train', 'val', 'test')
            start_sample_idx: Starting sample index
            start_timestep: Starting timestep number (1-based)
        """
        if not self.save_predictions or self.predictions_dir is None:
            return

        if predictions.ndim == 4:
            # Single timestep predictions: (batch_size, C, H, W)
            for i in range(predictions.shape[0]):
                self._save_prediction_as_h5(predictions[i], split, start_sample_idx + i, start_timestep)
        elif predictions.ndim == 5:
            # Multi timestep predictions: (batch_size, timesteps, C, H, W)
            for i in range(predictions.shape[0]):
                for t in range(predictions.shape[1]):
                    self._save_prediction_as_h5(predictions[i, t], split, start_sample_idx + i, start_timestep + t)
        elif predictions.ndim == 3:
            # Legacy single channel: (batch_size, H, W)
            for i in range(predictions.shape[0]):
                self._save_prediction_as_h5(predictions[i], split, start_sample_idx + i, start_timestep)

    def evaluate_model(self, num_samples: int = 100, debug_comparison: bool = False) -> dict[str, float]:
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
                # Dataset returns (1, T, C, H, W) for input_seq and (1, max_k_steps, C, H, W) for label
                input_seq = item["data"]["input_seq"][0]  # Remove outer batch dim: (T, C, H, W)
                target_seq = item["label"][0]  # Remove outer batch dim: (max_k_steps, C, H, W)
                # For evaluation, take only the first target frame
                target = target_seq[0]  # (C, H, W)
                input_seqs.append(input_seq)
                targets.append(target)

            return torch.stack(input_seqs), torch.stack(targets)

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn
        )

        metrics = {"mse": 0.0, "mae": 0.0, "rel_error": 0.0, "count": 0}

        # Debug comparison: test single sample vs batch processing
        if debug_comparison:
            print("=== DEBUG COMPARISON ===")
            # Test single sample like train/val
            sample = self.test_dataset[0]
            input_seq_single = sample["data"]["input_seq"].to(self.device)  # (1, T, C, H, W)
            target_single = sample["label"][:, 0].to(self.device)  # (1, C, H, W)

            with torch.no_grad():
                pred_single = self.model(input_seq_single, return_delta=False)
                pred_single_denorm = self.test_dataset.denormalize(pred_single)
                target_single_denorm = self.test_dataset.denormalize(target_single)
                mae_single = torch.nn.functional.l1_loss(pred_single_denorm, target_single_denorm)
                print(f"Single sample MAE: {mae_single.item():.6f}")

            # Test batch processing
            first_batch = next(iter(dataloader))
            input_seq_batch, target_batch = first_batch
            input_seq_batch = input_seq_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            with torch.no_grad():
                pred_batch = self.model(input_seq_batch, return_delta=False)
                pred_batch_denorm = self.test_dataset.denormalize(pred_batch)
                target_batch_denorm = self.test_dataset.denormalize(target_batch)
                mae_batch = torch.nn.functional.l1_loss(pred_batch_denorm[0:1], target_batch_denorm[0:1])
                print(f"First batch sample MAE: {mae_batch.item():.6f}")

                # Check if data is the same
                print(f"Input shapes - Single: {input_seq_single.shape}, Batch: {input_seq_batch[0:1].shape}")
                print(f"Target shapes - Single: {target_single.shape}, Batch: {target_batch[0:1].shape}")
                print(f"Input data close: {torch.allclose(input_seq_single, input_seq_batch[0:1], atol=1e-6)}")
                print(f"Target data close: {torch.allclose(target_single, target_batch[0:1], atol=1e-6)}")
            print("=== END DEBUG ===\n")

        with torch.no_grad():
            # Use direct dataset iteration to get original indices
            for idx in range(min(num_samples, len(self.test_dataset))):
                sample = self.test_dataset[idx]
                input_seq = sample["data"]["input_seq"].to(self.device)  # (1, T, C, H, W)
                target = sample["label"][:, 0].to(self.device)  # (1, C, H, W)

                # Forward pass with residual prediction
                pred = self.model(input_seq, return_delta=False)  # Get composed prediction u_{t+1} = u_t + Δu

                # IMPORTANT: Denormalize predictions for fair comparison with targets
                # Both pred and target are normalized, so we need to denormalize both for proper metrics
                pred_denorm = self.test_dataset.denormalize(pred)
                target_denorm = self.test_dataset.denormalize(target)

                # Save predictions as H5 files if enabled
                if self.save_predictions:
                    pred_np = pred_denorm.cpu().numpy()  # Shape: (1, C, H, W)
                    # Get the real timestep from the dataset indices
                    real_timestep = (
                        self.test_dataset.indices[idx] + self.test_dataset.input_length + 1
                    )  # +1 because prediction is next timestep
                    self._save_prediction_as_h5(
                        pred_np[0],  # Take first batch, keep all channels (C, H, W)
                        "test",
                        idx,  # Use sample index
                        real_timestep,  # Use real timestep as timestep
                    )

                # Calculate metrics in denormalized space
                mse = torch.nn.functional.mse_loss(pred_denorm, target_denorm)
                mae = torch.nn.functional.l1_loss(pred_denorm, target_denorm)
                # Use the same relative error calculation as training (global L2 norm)
                target_flat = target_denorm.flatten(start_dim=2)
                pred_flat = pred_denorm.flatten(start_dim=2)
                target_norm = torch.norm(target_flat, dim=2, keepdim=True)
                error_norm = torch.norm(pred_flat - target_flat, dim=2, keepdim=True)
                rel_error = (error_norm / (target_norm + 1e-8)).mean()

                metrics["mse"] += mse.item()
                metrics["mae"] += mae.item()
                metrics["rel_error"] += rel_error.item()
                metrics["count"] += 1

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

    def generate_sequence_prediction(self, input_seq: torch.Tensor, num_predictions: int = 5) -> torch.Tensor:
        """Generate autoregressive predictions using residual prediction.

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
                # Residual prediction: model outputs u_{t+1} = u_t + Δu
                next_pred = self.model(current_seq, return_delta=False)  # u_{t+1} [B, C, H, W]

                # NOTE: next_pred is normalized, but we keep it normalized for autoregressive feedback
                # Denormalization happens at the end for evaluation/visualization

                # Debug: Check shapes on first iteration
                if i == 0:
                    print(f"Input sequence shape: {current_seq.shape}")
                    print(f"Prediction shape: {next_pred.shape}")
                    print(f"Input sequence data range: [{current_seq.min():.4f}, {current_seq.max():.4f}]")
                    print(f"Prediction data range: [{next_pred.min():.4f}, {next_pred.max():.4f}]")

                    # Debug: Check delta prediction to understand the issue
                    x_last = current_seq[:, -1]  # Last frame
                    delta_pred = self.model(current_seq, return_delta=True)  # Get delta only
                    print(f"Last frame (u_t) range: [{x_last.min():.4f}, {x_last.max():.4f}]")
                    print(f"Delta prediction (Δu) range: [{delta_pred.min():.4f}, {delta_pred.max():.4f}]")
                    print(
                        f"Expected next_pred = u_t + Δu: "
                        f"[{(x_last + delta_pred).min():.4f}, {(x_last + delta_pred).max():.4f}]"
                    )

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
        """Visualize a sample prediction with ground truth comparison showing u, v, w components and magnitude."""
        print(f"Visualizing sample {sample_idx} with multi-channel support...")

        # Get ground truth from the same sequence (consecutive timesteps)
        ground_truth_frames = []

        # Try to get a sample with longer max_k_steps if available, otherwise use consecutive samples
        sample = self.test_dataset[sample_idx]
        target_seq = sample["label"]  # (1, max_k_steps, C, H, W)

        if target_seq.shape[1] >= num_future:
            # If we have enough future steps in this sample, use them
            target_seq_denorm = self.test_dataset.denormalize(target_seq)
            target_seq_denorm = target_seq_denorm.cpu().numpy()[0]  # (max_k_steps, C, H, W)

            for i in range(num_future + 1):
                if i < target_seq_denorm.shape[0]:
                    ground_truth_frames.append(target_seq_denorm[i])  # Keep all channels (C, H, W)
                else:
                    # Repeat last frame if not enough
                    ground_truth_frames.append(ground_truth_frames[-1])
        else:
            # Fallback: use consecutive samples (original approach)
            print("WARNING: Using consecutive samples as ground truth - not true autoregressive evaluation!")
            for i in range(num_future + 1):
                if sample_idx + i < len(self.test_dataset):
                    sample_i = self.test_dataset[sample_idx + i]
                    target_seq_i = sample_i["label"]  # (1, max_k_steps, C, H, W)

                    # IMPORTANT: Denormalize ground truth for proper comparison
                    target_seq_denorm_i = self.test_dataset.denormalize(target_seq_i)
                    target_seq_denorm_i = target_seq_denorm_i.cpu().numpy()[0]  # (max_k_steps, C, H, W)

                    # Take the first target frame
                    target = target_seq_denorm_i[0]  # (C, H, W)
                    ground_truth_frames.append(target)  # Keep all channels
                else:
                    # If we run out of samples, repeat the last one
                    ground_truth_frames.append(ground_truth_frames[-1])

        # Get the initial sample for prediction
        sample = self.test_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)

        # Generate predictions
        pred_seq = self.generate_sequence_prediction(input_seq, num_future)

        # IMPORTANT: Denormalize both input and predictions for proper visualization
        input_seq_denorm = self.test_dataset.denormalize(input_seq)
        pred_seq_denorm = self.test_dataset.denormalize(pred_seq)

        # Move to CPU for visualization
        input_seq = input_seq_denorm.cpu().numpy()[0]  # (T, C, H, W)
        pred_seq = pred_seq_denorm.cpu().numpy()[0]  # (T_pred, C, H, W)

        # Limit display to reasonable number of timesteps
        display_steps = min(num_future + 1, 10)

        # Create figure with 5 rows: u, v, w, magnitude, error_magnitude
        fig, axes = plt.subplots(5, display_steps, figsize=(3 * display_steps, 15))
        if display_steps == 1:
            axes = axes.reshape(5, 1)

        channel_names = ["u", "v", "w"]
        print(f"\nQuantitative Results for channels {channel_names} (Per-channel normalization):")
        print("Step | Channel | MSE     | MAE     | RMS-Rel Error")
        print("-" * 50)

        for t in range(display_steps):
            if t == 0:
                # Show last input timestep
                truth_data = input_seq[-1]  # (C, H, W)
                pred_data = truth_data  # Same as ground truth for input
                title_suffix = "Last Input"
            else:
                # Show ground truth and predictions
                truth_data = ground_truth_frames[t]  # (C, H, W)
                pred_data = pred_seq[t - 1]  # (C, H, W)
                title_suffix = f"t+{t}"

                # Calculate metrics for each channel using smart relative error
                smart_errors = self._compute_smart_relative_error(
                    torch.from_numpy(pred_data), torch.from_numpy(truth_data), channel_names
                )

                for _c, channel_name in enumerate(channel_names):
                    if channel_name in smart_errors:
                        error_info = smart_errors[channel_name]
                        print(
                            f"t+{t:2d} | {channel_name:7s} | {error_info['mse']:.5f} | "
                            f"{error_info['mae']:.5f} | {error_info['rms_normalized']:.5f}"
                        )

            # Plot u, v, w channels separately
            for c, channel_name in enumerate(channel_names):
                if c < truth_data.shape[0] and c < pred_data.shape[0]:
                    # Truth vs prediction for this channel
                    truth_channel = truth_data[c]
                    pred_channel = pred_data[c]

                    # Calculate channel-specific colorbar range
                    vmin = min(truth_channel.min(), pred_channel.min())
                    vmax = max(truth_channel.max(), pred_channel.max())

                    # Show ground truth
                    if t == 0:
                        im = axes[c, t].imshow(truth_channel, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
                        axes[c, t].set_title(f"{channel_name.upper()}: {title_suffix}")
                    else:
                        # Show prediction
                        im = axes[c, t].imshow(pred_channel, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
                        axes[c, t].set_title(f"{channel_name.upper()}: Pred {title_suffix}")

                    axes[c, t].axis("off")
                    plt.colorbar(im, ax=axes[c, t], fraction=0.046, pad=0.04)
                else:
                    axes[c, t].axis("off")
                    axes[c, t].set_title(f"{channel_name.upper()}: N/A")

            # Plot velocity magnitude
            if truth_data.shape[0] >= 3 and pred_data.shape[0] >= 3:
                truth_magnitude = self._compute_velocity_magnitude(truth_data)
                pred_magnitude = self._compute_velocity_magnitude(pred_data)

                # Calculate magnitude colorbar range
                mag_vmin = min(truth_magnitude.min(), pred_magnitude.min())
                mag_vmax = max(truth_magnitude.max(), pred_magnitude.max())

                if t == 0:
                    im_mag = axes[3, t].imshow(
                        truth_magnitude,
                        cmap="plasma",
                        aspect="auto",
                        vmin=mag_vmin,
                        vmax=mag_vmax,
                    )
                    axes[3, t].set_title(f"Magnitude: {title_suffix}")
                else:
                    im_mag = axes[3, t].imshow(
                        pred_magnitude,
                        cmap="plasma",
                        aspect="auto",
                        vmin=mag_vmin,
                        vmax=mag_vmax,
                    )
                    axes[3, t].set_title(f"Magnitude: Pred {title_suffix}")

                axes[3, t].axis("off")
                plt.colorbar(im_mag, ax=axes[3, t], fraction=0.046, pad=0.04)

                # Plot magnitude error
                if t > 0:
                    mag_error = np.abs(truth_magnitude - pred_magnitude)
                    axes[4, t].imshow(mag_error, cmap="Reds", aspect="auto", vmin=0, vmax=mag_error.max())
                    mae_mag = np.mean(mag_error)
                    axes[4, t].set_title(f"Mag Error {title_suffix} (MAE: {mae_mag:.4f})")
                    print(
                        f"t+{t:2d} | {'Mag':7s} | {np.mean(mag_error**2):.5f} | {mae_mag:.5f} | "
                        f"{np.mean(mag_error / (truth_magnitude + 1e-8)):.5f}"
                    )
                else:
                    axes[4, t].imshow(np.zeros_like(truth_magnitude), cmap="Reds", aspect="auto", vmin=0, vmax=0.1)
                    axes[4, t].set_title("Mag Error (0)")

                axes[4, t].axis("off")
                plt.colorbar(axes[4, t].images[0], ax=axes[4, t], fraction=0.046, pad=0.04)
            else:
                axes[3, t].axis("off")
                axes[3, t].set_title("Magnitude: N/A")
                axes[4, t].axis("off")
                axes[4, t].set_title("Mag Error: N/A")

        plt.tight_layout()

        # Save figure
        save_path = self.output_dir / f"sample_{sample_idx}_multi_channel_prediction.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Multi-channel visualization saved to {save_path}")

        # Log image to wandb
        self._log_image_to_wandb(
            f"test/sample_{sample_idx}_multi_channel_prediction",
            save_path,
            f"Multi-channel (u,v,w,magnitude) prediction for sample {sample_idx} over {display_steps - 1} timesteps",
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

                # IMPORTANT: Denormalize ground truth data
                input_seq_raw = sample["data"]["input_seq"]  # (1, T, C, H, W)
                target_seq_raw = sample["label"]  # (1, max_k_steps, C, H, W)

                input_seq_denorm = self.test_dataset.denormalize(input_seq_raw).cpu().numpy()[0]  # (T, C, H, W)
                target_seq_denorm = (
                    self.test_dataset.denormalize(target_seq_raw).cpu().numpy()[0]
                )  # (max_k_steps, C, H, W)

                # Take the first target frame
                target = target_seq_denorm[0]  # (C, H, W)

                if i == 0:
                    # For first sample, add all input frames
                    for j in range(input_seq_denorm.shape[0]):
                        ground_truth_frames.append(input_seq_denorm[j])  # Keep all channels (C, H, W)
                # Add the target frame
                ground_truth_frames.append(target)  # Keep all channels (C, H, W)
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

        # IMPORTANT: Denormalize for proper visualization
        input_seq_denorm = self.test_dataset.denormalize(input_seq)
        pred_seq_denorm = self.test_dataset.denormalize(pred_seq)

        # Move to CPU
        input_seq = input_seq_denorm.cpu().numpy()[0]  # (T, C, H, W)
        pred_seq = pred_seq_denorm.cpu().numpy()[0]  # (T_pred, C, H, W)

        # Combine input sequence and predictions
        pred_full_sequence = np.concatenate([input_seq, pred_seq], axis=0)
        input_len = input_seq.shape[0]

        # Trim ground truth to match prediction sequence length
        ground_truth_frames = ground_truth_frames[: len(pred_full_sequence)]

        # Calculate per-frame colorbar ranges for dynamic visualization
        frame_ranges = {}
        frame_error_ranges = {}

        for frame_idx in range(len(ground_truth_frames)):
            truth_data = ground_truth_frames[frame_idx]  # (C, H, W)
            pred_data = pred_full_sequence[frame_idx]  # (C, H, W)

            # Compute velocity magnitude for visualization
            if truth_data.shape[0] >= 3 and pred_data.shape[0] >= 3:
                truth_magnitude = self._compute_velocity_magnitude(truth_data)  # (H, W)
                pred_magnitude = self._compute_velocity_magnitude(pred_data)  # (H, W)
            else:
                # Fallback to first channel if not enough channels
                truth_magnitude = truth_data[0] if truth_data.ndim == 3 else truth_data
                pred_magnitude = pred_data[0] if pred_data.ndim == 3 else pred_data

            error = np.abs(truth_magnitude - pred_magnitude)

            # Calculate range for this frame (combining truth and prediction magnitude)
            frame_vmin = min(truth_magnitude.min(), pred_magnitude.min())
            frame_vmax = max(truth_magnitude.max(), pred_magnitude.max())
            frame_ranges[frame_idx] = (frame_vmin, frame_vmax)

            # Calculate error range for this frame
            frame_error_ranges[frame_idx] = (0, error.max() if error.max() > 0 else 0.1)

        print("Video using dynamic per-frame color scaling for better temporal visualization")

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Initialize plots with first frame ranges
        frame_0_vmin, frame_0_vmax = frame_ranges[0]
        frame_0_error_vmin, frame_0_error_vmax = frame_error_ranges[0]

        # Compute initial velocity magnitude for display
        truth_data_0 = ground_truth_frames[0]  # (C, H, W)
        pred_data_0 = pred_full_sequence[0]  # (C, H, W)

        if truth_data_0.shape[0] >= 3 and pred_data_0.shape[0] >= 3:
            initial_truth_magnitude = self._compute_velocity_magnitude(truth_data_0)
            initial_pred_magnitude = self._compute_velocity_magnitude(pred_data_0)
        else:
            initial_truth_magnitude = truth_data_0[0] if truth_data_0.ndim == 3 else truth_data_0
            initial_pred_magnitude = pred_data_0[0] if pred_data_0.ndim == 3 else pred_data_0

        im1 = axes[0].imshow(
            initial_truth_magnitude, cmap="viridis", aspect="auto", vmin=frame_0_vmin, vmax=frame_0_vmax
        )
        axes[0].set_title("Ground Truth (Velocity Magnitude)")
        axes[0].axis("off")
        cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(
            initial_pred_magnitude, cmap="viridis", aspect="auto", vmin=frame_0_vmin, vmax=frame_0_vmax
        )
        axes[1].set_title("Prediction (Velocity Magnitude)")
        axes[1].axis("off")
        cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Error plot with first frame error range
        initial_error = np.abs(initial_truth_magnitude - initial_pred_magnitude)
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

            # Get multi-channel data
            truth_data = ground_truth_frames[frame]  # (C, H, W)
            pred_data = pred_full_sequence[frame]  # (C, H, W)

            # Compute velocity magnitude for display
            if truth_data.shape[0] >= 3 and pred_data.shape[0] >= 3:
                truth_magnitude = self._compute_velocity_magnitude(truth_data)  # (H, W)
                pred_magnitude = self._compute_velocity_magnitude(pred_data)  # (H, W)
            else:
                # Fallback to first channel
                truth_magnitude = truth_data[0] if truth_data.ndim == 3 else truth_data
                pred_magnitude = pred_data[0] if pred_data.ndim == 3 else pred_data

            # Update ground truth with velocity magnitude
            im1.set_array(truth_magnitude)
            im1.set_clim(vmin=frame_vmin, vmax=frame_vmax)

            # Update prediction with velocity magnitude (same range as ground truth)
            im2.set_array(pred_magnitude)
            im2.set_clim(vmin=frame_vmin, vmax=frame_vmax)

            # Update error with frame-specific error range
            error = np.abs(truth_magnitude - pred_magnitude)
            im3.set_array(error.copy())
            im3.set_clim(vmin=frame_error_vmin, vmax=frame_error_vmax)

            # Update colorbars to reflect the new ranges (use mappable instead of colorbar)
            cb1.mappable.set_clim(vmin=frame_vmin, vmax=frame_vmax)
            cb2.mappable.set_clim(vmin=frame_vmin, vmax=frame_vmax)  # Same range as cb1
            cb3.mappable.set_clim(vmin=frame_error_vmin, vmax=frame_error_vmax)

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

                # IMPORTANT: Denormalize both ground truth and predictions
                target_denorm = dataset.denormalize(target).cpu().numpy()[0, 0]  # (H, W)
                ground_truth_frames.append(target_denorm)

                # Predict using ground truth input (teacher forcing)
                with torch.no_grad():
                    pred = self.model(input_seq, return_delta=False)  # Use residual prediction
                    pred_denorm = dataset.denormalize(pred).cpu().numpy()[0]  # (C, H, W)
                    predictions.append(pred_denorm[0])  # Store first channel for visualization compatibility

                    # Save prediction as H5 file if enabled
                    if self.save_predictions:
                        # Get the real timestep from the dataset indices
                        real_timestep = dataset.indices[sample_idx + i] + dataset.input_length + 1
                        self._save_prediction_as_h5(
                            pred_denorm,  # Save all channels (C, H, W)
                            split,
                            real_timestep,  # Use real timestep as sample index
                            i + 1,  # Timestep 1-based for prediction sequence
                        )
            else:
                break

        return ground_truth_frames, predictions

    def evaluate_with_autoregressive(self, split: str = "train", sample_idx: int = 0, num_future: int = 30):
        """Evaluate using autoregressive prediction (TFR=0.0) for train/val data.

        Args:
            split: "train" or "val"
            sample_idx: Starting sample index
            num_future: Number of future timesteps to predict
        """
        print(f"Evaluating {split} data with autoregressive prediction (TFR=0.0)...")

        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        else:
            raise ValueError("Only 'train' and 'val' splits supported for autoregressive")

        # Get ground truth frames for the future timesteps
        ground_truth_frames = []
        for i in range(num_future + 5):  # Get input + future frames
            if sample_idx + i < len(dataset):
                sample = dataset[sample_idx + i]

                # IMPORTANT: Denormalize ground truth data
                input_seq_raw = sample["data"]["input_seq"]  # (1, T, C, H, W)
                target_seq_raw = sample["label"]  # (1, max_k_steps, C, H, W)

                input_seq_denorm = dataset.denormalize(input_seq_raw).cpu().numpy()[0]  # (T, C, H, W)
                target_seq_denorm = dataset.denormalize(target_seq_raw).cpu().numpy()[0]  # (max_k_steps, C, H, W)

                # Take the first target frame
                target = target_seq_denorm[0]  # (C, H, W)

                if i == 0:
                    # For first sample, add all input frames
                    for j in range(input_seq_denorm.shape[0]):
                        ground_truth_frames.append(input_seq_denorm[j])  # Keep all channels (C, H, W)
                # Add the target frame
                ground_truth_frames.append(target)  # Keep all channels (C, H, W)
            else:
                # If we run out of samples, repeat the last one
                if ground_truth_frames:
                    ground_truth_frames.append(ground_truth_frames[-1])

        print(f"Collected {len(ground_truth_frames)} ground truth frames for {split}")

        # Get the initial sample for autoregressive prediction
        sample = dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)

        # Generate autoregressive predictions
        pred_seq = self.generate_sequence_prediction(input_seq, num_future)

        # IMPORTANT: Denormalize predictions for fair comparison
        input_seq_denorm = dataset.denormalize(input_seq)
        pred_seq_denorm = dataset.denormalize(pred_seq)

        # Save autoregressive predictions as H5 files if enabled
        if self.save_predictions:
            pred_seq_np = pred_seq_denorm.cpu().numpy()[0]  # (T_pred, C, H, W)
            # Get the real starting timestep for this sample
            base_real_timestep = dataset.indices[sample_idx] + dataset.input_length + 1
            for t in range(pred_seq_np.shape[0]):
                self._save_prediction_as_h5(
                    pred_seq_np[t],  # Save all channels (C, H, W)
                    f"{split}_autoregressive",  # Distinguish from teacher forcing
                    base_real_timestep + t,  # Use real timestep sequence
                    t + 1,  # Timestep 1-based for prediction sequence
                )

        # Move to CPU
        input_seq = input_seq_denorm.cpu().numpy()[0]  # (T, C, H, W)
        pred_seq = pred_seq_denorm.cpu().numpy()[0]  # (T_pred, C, H, W)

        # Combine input sequence and predictions
        pred_full_sequence = np.concatenate([input_seq, pred_seq], axis=0)
        input_len = input_seq.shape[0]

        # Trim ground truth to match prediction sequence length
        ground_truth_frames = ground_truth_frames[: len(pred_full_sequence)]

        # Separate predictions only (exclude input frames)
        predictions_only = pred_seq  # Keep all channels (T_pred, C, H, W)
        ground_truth_pred_only = ground_truth_frames[input_len:]  # Skip input frames, keep all channels

        return ground_truth_pred_only, predictions_only

    def visualize_teacher_forcing(self, split: str = "train", sample_idx: int = 0, num_future: int = 30):
        """Visualize teacher forcing results for train/val data."""
        print(f"Visualizing {split} data with teacher forcing...")

        ground_truth_frames, predictions = self.evaluate_with_teacher_forcing(split, sample_idx, num_future)

        # Create figure - display all predictions (up to reasonable limit)
        num_display = min(len(predictions), 30)  # Show up to 12 steps instead of 6
        fig, axes = plt.subplots(3, num_display, figsize=(3 * num_display, 9))
        if num_display == 1:
            axes = axes.reshape(3, 1)

        # Calculate and print quantitative metrics
        print(f"\n{split.upper()} Teacher Forcing Results:")
        print("Step | MSE     | MAE     | Rel Error")
        print("-" * 35)

        # Calculate per-timestep colorbar ranges for teacher forcing visualization
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

    def visualize_autoregressive(self, split: str = "train", sample_idx: int = 0, num_future: int = 30):
        """Visualize autoregressive (TFR=0.0) results for train/val data."""
        print(f"Visualizing {split} data with autoregressive prediction (TFR=0.0)...")

        ground_truth_frames, predictions = self.evaluate_with_autoregressive(split, sample_idx, num_future)

        # Create figure - display all predictions (up to reasonable limit)
        num_display = min(len(predictions), 30)  # Show up to 12 steps instead of 6
        fig, axes = plt.subplots(3, num_display, figsize=(3 * num_display, 9))
        if num_display == 1:
            axes = axes.reshape(3, 1)

        # Calculate and print quantitative metrics
        print(f"\n{split.upper()} Autoregressive (TFR=0.0) Results (Per-channel normalization):")
        print("Step | MSE     | MAE     | RMS-Rel Error")
        print("-" * 40)

        # Calculate per-timestep colorbar ranges for autoregressive visualization
        ar_timestep_ranges = {}
        ar_timestep_error_ranges = {}

        for t in range(num_display):
            data = ground_truth_frames[t]  # (C, H, W)
            pred_data = predictions[t]  # (C, H, W)

            # Compute velocity magnitude for visualization
            if data.shape[0] >= 3 and pred_data.shape[0] >= 3:
                data_magnitude = self._compute_velocity_magnitude(data)  # (H, W)
                pred_magnitude = self._compute_velocity_magnitude(pred_data)  # (H, W)
            else:
                # Fallback to first channel
                data_magnitude = data[0] if data.ndim == 3 else data
                pred_magnitude = pred_data[0] if pred_data.ndim == 3 else pred_data

            error = np.abs(data_magnitude - pred_magnitude)

            # Calculate range for this timestep (combining truth and prediction magnitude)
            timestep_vmin = min(data_magnitude.min(), pred_magnitude.min())
            timestep_vmax = max(data_magnitude.max(), pred_magnitude.max())
            ar_timestep_ranges[t] = (timestep_vmin, timestep_vmax)

            # Calculate error range for this timestep
            ar_timestep_error_ranges[t] = (0, error.max() if error.max() > 0 else 0.1)

        for t in range(num_display):
            data = ground_truth_frames[t]  # (C, H, W)
            pred_data = predictions[t]  # (C, H, W)

            # Compute velocity magnitude for visualization
            if data.shape[0] >= 3 and pred_data.shape[0] >= 3:
                data_magnitude = self._compute_velocity_magnitude(data)  # (H, W)
                pred_magnitude = self._compute_velocity_magnitude(pred_data)  # (H, W)
            else:
                # Fallback to first channel
                data_magnitude = data[0] if data.ndim == 3 else data
                pred_magnitude = pred_data[0] if pred_data.ndim == 3 else pred_data

            error = np.abs(data_magnitude - pred_magnitude)

            # Calculate metrics using smart relative error for per-channel analysis
            # smart_errors = self._compute_smart_relative_error(
            #     torch.from_numpy(pred_data), torch.from_numpy(data), ["u", "v", "w"]
            # )

            # For magnitude, calculate traditional metrics
            mse = np.mean(error**2)
            mae = np.mean(error)
            # Use RMS-normalized relative error for magnitude (more stable)
            magnitude_rms = np.sqrt(np.mean(data_magnitude**2))
            rel_error = mae / (magnitude_rms + 1e-8)
            print(f"t+{t + 1:2d} | {mse:.5f} | {mae:.5f} | {rel_error:.5f}")

            # Get timestep-specific ranges
            ar_vmin, ar_vmax = ar_timestep_ranges[t]
            ar_error_vmin, ar_error_vmax = ar_timestep_error_ranges[t]

            # Ground truth (using timestep-specific colorbar range)
            im1 = axes[0, t].imshow(data_magnitude, cmap="viridis", aspect="auto", vmin=ar_vmin, vmax=ar_vmax)
            axes[0, t].set_title(f"True t+{t + 1} (Magnitude)")
            axes[0, t].axis("off")
            plt.colorbar(im1, ax=axes[0, t], fraction=0.046, pad=0.04)

            # Prediction (using same timestep-specific colorbar range)
            im2 = axes[1, t].imshow(pred_magnitude, cmap="viridis", aspect="auto", vmin=ar_vmin, vmax=ar_vmax)
            axes[1, t].set_title(f"Pred t+{t + 1} (AR Magnitude)")
            axes[1, t].axis("off")
            plt.colorbar(im2, ax=axes[1, t], fraction=0.046, pad=0.04)

            # Error (using timestep-specific error range)
            im3 = axes[2, t].imshow(error, cmap="Reds", aspect="auto", vmin=ar_error_vmin, vmax=ar_error_vmax)
            axes[2, t].set_title(f"Error t+{t + 1} (MAE: {mae:.4f})")
            axes[2, t].axis("off")
            plt.colorbar(im3, ax=axes[2, t], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save figure
        save_path = self.output_dir / f"{split}_autoregressive_prediction.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Autoregressive visualization saved to {save_path}")

        # Log image to wandb
        self._log_image_to_wandb(
            f"{split}/autoregressive_prediction",
            save_path,
            f"Autoregressive prediction (TFR=0.0) for {split} data over {num_future} timesteps",
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

            # Update colorbars to reflect the new ranges (use mappable instead of colorbar)
            tf_cb1.mappable.set_clim(vmin=tf_frame_vmin, vmax=tf_frame_vmax)
            tf_cb2.mappable.set_clim(vmin=tf_frame_vmin, vmax=tf_frame_vmax)  # Same range as tf_cb1
            tf_cb3.mappable.set_clim(vmin=tf_frame_error_vmin, vmax=tf_frame_error_vmax)

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
        metrics = self.evaluate_model(num_samples=50, debug_comparison=True)

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

        # NEW: Evaluate training data (autoregressive - TFR=0.0)
        print("\n" + "=" * 60)
        print("TRAINING DATA EVALUATION (Autoregressive - TFR=0.0)")
        print("=" * 60)
        self.visualize_autoregressive(split="train", sample_idx=0, num_future=30)

        # NEW: Evaluate validation data (autoregressive - TFR=0.0)
        print("\n" + "=" * 60)
        print("VALIDATION DATA EVALUATION (Autoregressive - TFR=0.0)")
        print("=" * 60)
        self.visualize_autoregressive(split="val", sample_idx=0, num_future=30)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print("- Test (autoregressive): sample_0_prediction.png, sample_0_evolution.mp4")
        print("- Train (teacher forcing): train_teacher_forcing_prediction.png, train_teacher_forcing_evolution.mp4")
        print("- Val (teacher forcing): val_teacher_forcing_prediction.png, val_teacher_forcing_evolution.mp4")
        print("- Train (autoregressive): train_autoregressive_prediction.png")
        print("- Val (autoregressive): val_autoregressive_prediction.png")

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
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save prediction results as H5 files in logs directory"
    )

    # Parse known args to separate our checkpoint path from Hydra overrides
    args, hydra_overrides = parser.parse_known_args()

    # Use command line argument or default path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Default to the hardcoded path if no argument provided
        checkpoint_path = (
            "/home/sh/CB/icon-thewell-dev/logs/flow_swin_2d/runs/2025-09-09_11-35-33-106383/checkpoints/step_29000.ckpt"
        )

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Initialize Hydra with the remaining overrides
    with hydra.initialize(version_base="1.3", config_path="configs"):
        cfg = hydra.compose(config_name="train_flow_swin_2d", overrides=hydra_overrides)

        evaluator = SimpleModelEvaluator(checkpoint_path, cfg.model, save_predictions=args.save_predictions)
        try:
            evaluator.run_evaluation()
        finally:
            evaluator.close_wandb()  # Ensure wandb is properly closed


if __name__ == "__main__":
    main()
