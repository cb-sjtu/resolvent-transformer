#!/usr/bin/env python3
"""
Visualization module for flow evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import compute_velocity_magnitude, ensure_numpy_array, log_image_to_wandb


class FlowVisualizer:
    """Handle all visualization tasks for flow evaluation."""

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Base directory for saving visualizations
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True, parents=True)

    def plot_single_frame_comparison(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_idx: int = 0,
        timestep: int = 0,
        channel_names: list[str] = None,
        save_plot: bool = True,
    ) -> Path | None:
        """
        Plot prediction vs ground truth for a single frame.

        Args:
            pred: Predicted tensor (C, H, W)
            target: Target tensor (C, H, W)
            sample_idx: Sample index for filename
            timestep: Timestep for filename
            channel_names: Names of channels
            save_plot: Whether to save the plot

        Returns:
            Path to saved plot if save_plot=True
        """
        if channel_names is None:
            channel_names = ["u", "v", "w"]

        pred_np = ensure_numpy_array(pred)
        target_np = ensure_numpy_array(target)

        # Create figure with 5 rows: u, v, w, magnitude, error_magnitude
        num_channels = min(pred_np.shape[0], len(channel_names))
        fig, axes = plt.subplots(5, 2, figsize=(10, 15))
        fig.suptitle(
            f"Prediction vs Ground Truth - Sample {sample_idx}, Step {timestep}",
            fontsize=14,
        )

        # Plot u, v, w channels
        for c in range(min(3, num_channels)):
            channel_name = channel_names[c]
            pred_channel = pred_np[c]
            target_channel = target_np[c]

            # Calculate shared colorbar range
            vmin = min(pred_channel.min(), target_channel.min())
            vmax = max(pred_channel.max(), target_channel.max())

            # Ground truth
            im1 = axes[c, 0].imshow(
                target_channel, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax
            )
            axes[c, 0].set_title(f"{channel_name.upper()} - Ground Truth")
            axes[c, 0].axis("off")
            plt.colorbar(im1, ax=axes[c, 0], fraction=0.046, pad=0.04)

            # Prediction
            im2 = axes[c, 1].imshow(
                pred_channel, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax
            )
            axes[c, 1].set_title(f"{channel_name.upper()} - Prediction")
            axes[c, 1].axis("off")
            plt.colorbar(im2, ax=axes[c, 1], fraction=0.046, pad=0.04)

        # Plot magnitude if we have enough channels
        if pred_np.shape[0] >= 3 and target_np.shape[0] >= 3:
            pred_mag = compute_velocity_magnitude(pred_np)
            target_mag = compute_velocity_magnitude(target_np)

            mag_vmin = min(pred_mag.min(), target_mag.min())
            mag_vmax = max(pred_mag.max(), target_mag.max())

            # Ground truth magnitude
            im3 = axes[3, 0].imshow(
                target_mag, cmap="plasma", aspect="auto", vmin=mag_vmin, vmax=mag_vmax
            )
            axes[3, 0].set_title("Magnitude - Ground Truth")
            axes[3, 0].axis("off")
            plt.colorbar(im3, ax=axes[3, 0], fraction=0.046, pad=0.04)

            # Predicted magnitude
            im4 = axes[3, 1].imshow(
                pred_mag, cmap="plasma", aspect="auto", vmin=mag_vmin, vmax=mag_vmax
            )
            axes[3, 1].set_title("Magnitude - Prediction")
            axes[3, 1].axis("off")
            plt.colorbar(im4, ax=axes[3, 1], fraction=0.046, pad=0.04)

            # Magnitude error
            mag_error = np.abs(target_mag - pred_mag)
            im5 = axes[4, 0].imshow(
                mag_error, cmap="Reds", aspect="auto", vmin=0, vmax=mag_error.max()
            )
            axes[4, 0].set_title("Magnitude Error")
            axes[4, 0].axis("off")
            plt.colorbar(im5, ax=axes[4, 0], fraction=0.046, pad=0.04)

            # Error statistics
            mae = np.mean(mag_error)
            rmse = np.sqrt(np.mean(mag_error**2))
            axes[4, 1].text(
                0.1, 0.7, f"MAE: {mae:.5f}", transform=axes[4, 1].transAxes, fontsize=12
            )
            axes[4, 1].text(
                0.1,
                0.5,
                f"RMSE: {rmse:.5f}",
                transform=axes[4, 1].transAxes,
                fontsize=12,
            )
            axes[4, 1].text(
                0.1,
                0.3,
                f"Max Error: {mag_error.max():.5f}",
                transform=axes[4, 1].transAxes,
                fontsize=12,
            )
            axes[4, 1].set_title("Error Statistics")
            axes[4, 1].axis("off")
        else:
            # Hide magnitude rows if not enough channels
            for i in [3, 4]:
                axes[i, 0].axis("off")
                axes[i, 1].axis("off")

        plt.tight_layout()

        if save_plot:
            plot_path = (
                self.plots_dir / f"comparison_sample_{sample_idx}_step_{timestep}.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return plot_path
        else:
            plt.show()
            return None

    def plot_multi_step_prediction(
        self,
        predictions: list[torch.Tensor],
        ground_truth: list[torch.Tensor],
        sample_idx: int = 0,
        max_steps: int = 10,
        channel_names: list[str] = None,
    ) -> Path:
        """
        Plot multi-step prediction evolution.

        Args:
            predictions: List of prediction tensors (C, H, W)
            ground_truth: List of ground truth tensors (C, H, W)
            sample_idx: Sample index for filename
            max_steps: Maximum number of steps to plot
            channel_names: Names of channels

        Returns:
            Path to saved plot
        """
        if channel_names is None:
            channel_names = ["u", "v", "w"]

        num_steps = min(len(predictions), len(ground_truth), max_steps)

        # Create figure with magnitude comparison
        fig, axes = plt.subplots(3, num_steps, figsize=(3 * num_steps, 9))
        if num_steps == 1:
            axes = axes.reshape(3, 1)

        fig.suptitle(
            f"Multi-step Prediction Evolution - Sample {sample_idx}", fontsize=16
        )

        # Compute global ranges for consistent coloring
        all_pred_mag = []
        all_truth_mag = []

        for i in range(num_steps):
            pred_np = ensure_numpy_array(predictions[i])
            truth_np = ensure_numpy_array(ground_truth[i])

            if pred_np.shape[0] >= 3 and truth_np.shape[0] >= 3:
                pred_mag = compute_velocity_magnitude(pred_np)
                truth_mag = compute_velocity_magnitude(truth_np)
                all_pred_mag.append(pred_mag)
                all_truth_mag.append(truth_mag)

        if all_pred_mag and all_truth_mag:
            global_vmin = min(np.min(all_pred_mag), np.min(all_truth_mag))
            global_vmax = max(np.max(all_pred_mag), np.max(all_truth_mag))
        else:
            global_vmin, global_vmax = 0, 1

        for i in range(num_steps):
            pred_np = ensure_numpy_array(predictions[i])
            truth_np = ensure_numpy_array(ground_truth[i])

            if pred_np.shape[0] >= 3 and truth_np.shape[0] >= 3:
                pred_mag = compute_velocity_magnitude(pred_np)
                truth_mag = compute_velocity_magnitude(truth_np)

                # Ground truth
                im1 = axes[0, i].imshow(
                    truth_mag,
                    cmap="plasma",
                    aspect="auto",
                    vmin=global_vmin,
                    vmax=global_vmax,
                )
                axes[0, i].set_title(f"Truth t+{i + 1}")
                axes[0, i].axis("off")

                # Prediction
                im2 = axes[1, i].imshow(
                    pred_mag,
                    cmap="plasma",
                    aspect="auto",
                    vmin=global_vmin,
                    vmax=global_vmax,
                )
                axes[1, i].set_title(f"Pred t+{i + 1}")
                axes[1, i].axis("off")

                # Error
                error = np.abs(truth_mag - pred_mag)
                im3 = axes[2, i].imshow(
                    error, cmap="Reds", aspect="auto", vmin=0, vmax=error.max()
                )
                mae = np.mean(error)
                axes[2, i].set_title(f"Error (MAE: {mae:.4f})")
                axes[2, i].axis("off")

                if i == num_steps - 1:  # Add colorbars to last column
                    plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
                    plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
                    plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)

        plt.tight_layout()

        plot_path = self.plots_dir / f"multi_step_sample_{sample_idx}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return plot_path

    def plot_error_evolution(
        self, errors: list[float], error_type: str = "MAE", save_plot: bool = True
    ) -> Path | None:
        """
        Plot error evolution over time steps.

        Args:
            errors: List of error values
            error_type: Type of error being plotted
            save_plot: Whether to save the plot

        Returns:
            Path to saved plot if save_plot=True
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        timesteps = list(range(1, len(errors) + 1))
        ax.plot(timesteps, errors, "o-", linewidth=2, markersize=6)

        ax.set_xlabel("Time Step")
        ax.set_ylabel(f"{error_type}")
        ax.set_title(f"{error_type} Evolution Over Time")
        ax.grid(True, alpha=0.3)

        # Add trend line if more than 2 points
        if len(errors) > 2:
            z = np.polyfit(timesteps, errors, 1)
            p = np.poly1d(z)
            ax.plot(
                timesteps,
                p(timesteps),
                "--",
                alpha=0.7,
                label=f"Trend (slope: {z[0]:.6f})",
            )
            ax.legend()

        plt.tight_layout()

        if save_plot:
            plot_path = self.plots_dir / f"error_evolution_{error_type.lower()}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return plot_path
        else:
            plt.show()
            return None

    def log_plot_to_wandb(
        self, wandb_logger, plot_path: Path, key: str, caption: str = None
    ):
        """Log plot to wandb if available."""
        log_image_to_wandb(wandb_logger, key, plot_path, caption)
