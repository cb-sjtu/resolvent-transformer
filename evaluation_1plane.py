#!/usr/bin/env python3
"""
Evaluation script for 1-plane 3-channel Flow Swin Transformer implementation.
Loads the best model checkpoint and generates comprehensive visualizations for single plane.
Only processes u,v,w velocity fields (pressure channel removed).
"""

import os
import sys
import warnings
from pathlib import Path

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

from src.datasets.flow_sequence_2d.flow_sequence_1plane import FlowSequence1PlaneDataset  # noqa: E402


class OnePlaneModelEvaluator:
    """Evaluator for the 1-plane 3-channel Flow Swin Transformer model (u,v,w only)."""

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
                        project="turbulence_swin_1plane",
                        id=training_run_id,
                        resume="allow",
                        tags=["evaluation", "flow", "swin", "1plane", "3channel", "uvw", "yslice54"],
                    )
                    print("Successfully resumed training wandb run for evaluation logging")
                except Exception as e:
                    print(f"Could not resume training run: {e}")
                    # Fallback: create a new linked run
                    self.wandb_run = wandb.init(
                        project="turbulence_swin_1plane",
                        name=f"evaluation_{log_dir_name}",
                        tags=["evaluation", "flow", "swin", "1plane", "3channel", "uvw", "yslice54"],
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
                    project="turbulence_swin_1plane",
                    name=f"evaluation_{log_dir_name}",
                    tags=["evaluation", "flow", "swin", "1plane", "3channel", "uvw", "yslice54"],
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
        """Setup the 1-plane datasets for evaluation."""
        print("Setting up 1-plane datasets...")

        # Dataset configuration matching training
        data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"
        field_names = ["u", "v", "w"]  # Only velocity fields (removed pressure)
        file_pattern = "*u-v-w-p_scale2-3-1_yslice54*.h5"
        resolution_scale = (2, 3, 1)
        y_slice = 54  # y_slice54
        norm_stats_file = "norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json"

        # Create datasets for all splits
        train_dataset = FlowSequence1PlaneDataset(
            data_dir=data_dir,
            input_length=5,
            field_names=field_names,
            file_pattern=file_pattern,
            resolution_scale=resolution_scale,
            y_slice=y_slice,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="train",
            enable_normalization=True,
            norm_stats=norm_stats_file,
        )

        val_dataset = FlowSequence1PlaneDataset(
            data_dir=data_dir,
            input_length=5,
            field_names=field_names,
            file_pattern=file_pattern,
            resolution_scale=resolution_scale,
            y_slice=y_slice,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="val",
            enable_normalization=True,
            norm_stats=norm_stats_file,
        )

        test_dataset = FlowSequence1PlaneDataset(
            data_dir=data_dir,
            input_length=5,
            field_names=field_names,
            file_pattern=file_pattern,
            resolution_scale=resolution_scale,
            y_slice=y_slice,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            split="test",
            enable_normalization=True,
            norm_stats=norm_stats_file,
        )

        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"Channel info: {train_dataset.get_channel_info()['num_channels']} total channels")

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
        """Generate autoregressive sequence predictions."""
        self.model.eval()
        with torch.no_grad():
            predictions = []
            current_input = input_seq.clone()

            for _ in range(num_predictions):
                # Predict next frame
                next_pred = self.model(current_input)  # (B, C, H, W)
                predictions.append(next_pred)

                # Update input sequence for next prediction
                # Remove first frame and add prediction
                current_input = torch.cat([current_input[:, 1:], next_pred.unsqueeze(1)], dim=1)

            # Stack predictions: (B, T_pred, C, H, W)
            pred_seq = torch.stack(predictions, dim=1)

        return pred_seq

    def visualize_1plane_prediction(self, sample_idx: int = 0, num_future: int = 20):
        """Visualize 1-plane 3-channel prediction with comprehensive comparison (u,v,w only)."""
        print(f"Visualizing 1-plane sample {sample_idx} with {num_future} future steps...")

        # Get sample and generate predictions
        sample = self.test_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)

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
        input_seq_np = input_seq_denorm.cpu().numpy()[0]  # (T, C, H, W)
        pred_seq_np = pred_seq_denorm.cpu().numpy()[0]  # (T_pred, C, H, W)

        # Get channel info
        channel_info = self.test_dataset.get_channel_info()
        field_names = channel_info["field_names"]  # ["u", "v", "w"]
        y_slice = channel_info["y_slice"]  # 54

        # Create visualizations for each channel (3 total: u, v, w)
        self._create_channel_visualizations(
            input_seq_np, pred_seq_np, ground_truth_frames, field_names, y_slice, sample_idx, num_future
        )

        print(f"Visualization complete for sample {sample_idx}")

    def _create_channel_visualizations(
        self, input_seq, pred_seq, ground_truth_frames, field_names, y_slice, sample_idx, num_future
    ):
        """Create separate visualization for each channel with 20 steps."""
        display_steps = min(num_future, 20)

        for field_idx, field_name in enumerate(field_names):
            channel_idx = field_idx

            # Create figure for this channel: 3 rows (GT, Pred, Error) × timesteps
            fig, axes = plt.subplots(3, display_steps, figsize=(2 * display_steps, 8))
            if display_steps == 1:
                axes = axes.reshape(3, 1)

            print(f"\nChannel: {field_name.upper()} (y={y_slice})")
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

            plt.suptitle(f"{field_name.upper()} (y={y_slice}) - 20 Steps", fontsize=12)
            plt.tight_layout()

            # Save individual channel visualization
            output_path = self.output_dir / f"channel_{field_name}_sample_{sample_idx}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved channel visualization: {output_path}")

            # Log to wandb if available
            if self.wandb_run:
                self.wandb_run.log({f"channel_{field_name}_sample_{sample_idx}": wandb.Image(str(output_path))})

            plt.close()  # Close to save memory

    def run_comprehensive_evaluation(self, num_samples: int = 1, num_future: int = 20):
        """Run comprehensive evaluation."""
        print(f"Running comprehensive evaluation on {num_samples} samples with {num_future} future steps...")

        # Test data evaluation (autoregressive)
        print("\n" + "=" * 60)
        print("TEST DATA EVALUATION (Autoregressive)")
        print("=" * 60)
        for i in range(num_samples):
            if i < len(self.test_dataset):
                print(f"\n=== Evaluating Sample {i} (Autoregressive) ===")
                self.visualize_1plane_prediction(sample_idx=i, num_future=num_future)

        print(f"\nEvaluation complete! Results saved to: {self.output_dir}")
        print("\nGenerated visualizations:")
        print("- Individual channel images (3 files per sample: u, v, w)")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 1-plane Flow Swin Transformer")
    parser.add_argument("checkpoint_path", type=str, nargs="?", help="Path to model checkpoint")
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
            "/home/sh/CB/icon-thewell-dev/logs/flow_swin_1plane/runs/2025-11-01_00-00-00-000000/checkpoints/last.ckpt"
        )

    # Load model config (simplified for direct usage)
    from omegaconf import OmegaConf

    # Create a basic model config for 1-plane model (3 channels: 1 plane × 3 fields)
    model_cfg = OmegaConf.create(
        {
            "input_shape": [128, 128],
            "sequence_length": 5,
            "prediction_horizon": 1,
            "num_channels": 3,
            "patch_size": [4, 4],
            "embed_dim": 384,
            "depths": [2, 2, 4, 6, 4, 2, 2],
            "num_heads": 12,
            "window_size": [8, 8],
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "drop_rate": 0.1,
            "attn_drop_rate": 0.1,
            "drop_path_rate": 0.1,
        }
    )

    # Create evaluator and run evaluation
    evaluator = OnePlaneModelEvaluator(
        checkpoint_path=checkpoint_path, model_cfg=model_cfg, save_predictions=args.save_predictions
    )

    evaluator.run_comprehensive_evaluation(num_samples=args.num_samples, num_future=args.num_future)


if __name__ == "__main__":
    main()
