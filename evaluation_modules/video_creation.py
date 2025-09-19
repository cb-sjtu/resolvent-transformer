#!/usr/bin/env python3
"""
Video creation module for flow evaluation.
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .utils import compute_velocity_magnitude, ensure_numpy_array


class VideoCreator:
    """Handle video creation for flow evaluation."""

    def __init__(self, output_dir: Path):
        """
        Initialize video creator.

        Args:
            output_dir: Base directory for saving videos
        """
        self.output_dir = Path(output_dir)
        self.videos_dir = self.output_dir / "videos"
        self.videos_dir.mkdir(exist_ok=True, parents=True)

    def create_prediction_video(self, evaluator, sample_idx: int = 0, num_future: int = 30) -> Path:
        """
        Create video showing autoregressive prediction vs ground truth.

        Args:
            evaluator: The evaluator instance (to access model and datasets)
            sample_idx: Sample index to visualize
            num_future: Number of future steps to predict

        Returns:
            Path to saved video
        """
        print(f"Creating prediction video for sample {sample_idx}...")

        # Get sample data
        dataset = evaluator.test_dataset
        sample = dataset[sample_idx]

        input_seq = sample["input_seq"]  # (1, input_length, C, H, W)
        ground_truth_seq = sample.get("target_seq", None)

        # Run autoregressive prediction (with denormalization)
        predictions = self._run_autoregressive_prediction(evaluator.model, input_seq, num_future, evaluator.dataset)

        # Prepare ground truth frames if available (denormalized for proper visualization)
        if ground_truth_seq is not None:
            ground_truth_frames = []
            for i in range(min(num_future, ground_truth_seq.shape[1])):
                gt_frame = ground_truth_seq[0, i]  # (C, H, W)
                if evaluator.dataset is not None:
                    gt_frame = evaluator.dataset.denormalize(gt_frame.unsqueeze(0))[0]  # Denormalize
                ground_truth_frames.append(gt_frame)
        else:
            # Generate synthetic ground truth or use model predictions
            ground_truth_frames = predictions  # Fallback

        # Create the video
        video_path = self._create_comparison_video(predictions, ground_truth_frames, sample_idx, "autoregressive")

        return video_path

    def _run_autoregressive_prediction(self, model, input_seq, num_steps: int, dataset=None):
        """Run autoregressive prediction for num_steps."""
        import torch

        model.eval()
        predictions = []

        current_seq = input_seq.clone()  # (1, input_length, C, H, W)

        with torch.no_grad():
            for _step in range(num_steps):
                # Predict next frame
                next_pred = model(current_seq)  # (1, C, H, W) or (1, 1, C, H, W)

                # Handle different output shapes
                if len(next_pred.shape) == 5:  # (B, T, C, H, W)
                    next_pred = next_pred[:, -1]  # Take last timestep: (B, C, H, W)
                elif len(next_pred.shape) == 4:  # (B, C, H, W)
                    pass  # Already correct shape
                else:
                    raise ValueError(f"Unexpected prediction shape: {next_pred.shape}")

                # Store prediction (denormalized for proper visualization)
                pred_frame = next_pred[0]  # Remove batch dimension: (C, H, W)
                if dataset is not None:
                    pred_frame = dataset.denormalize(pred_frame.unsqueeze(0))[0]  # Denormalize
                predictions.append(pred_frame)

                # Update current sequence for next prediction
                # Add time dimension back: (B, C, H, W) -> (B, 1, C, H, W)
                next_pred_with_time = next_pred.unsqueeze(1)

                # Slide window: remove first frame, add new prediction
                current_seq = torch.cat(
                    [
                        current_seq[:, 1:],  # Remove first frame
                        next_pred_with_time,  # Add new prediction
                    ],
                    dim=1,
                )

        return predictions

    def _create_comparison_video(
        self, predictions: list, ground_truth: list, sample_idx: int, mode: str = "comparison"
    ) -> Path:
        """
        Create comparison video between predictions and ground truth.

        Args:
            predictions: List of prediction tensors
            ground_truth: List of ground truth tensors
            sample_idx: Sample index for filename
            mode: Video mode for filename

        Returns:
            Path to saved video
        """
        num_frames = min(len(predictions), len(ground_truth))

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"Flow Prediction vs Ground Truth - Sample {sample_idx}", fontsize=14)

        # Initialize with first frame
        pred_0 = ensure_numpy_array(predictions[0])
        truth_0 = ensure_numpy_array(ground_truth[0])

        # Compute magnitudes for first frame
        if pred_0.shape[0] >= 3 and truth_0.shape[0] >= 3:
            pred_mag_0 = compute_velocity_magnitude(pred_0)
            truth_mag_0 = compute_velocity_magnitude(truth_0)
        else:
            pred_mag_0 = pred_0[0]  # Fallback to first channel
            truth_mag_0 = truth_0[0]

        # Calculate global ranges
        all_pred_mag = []
        all_truth_mag = []

        for i in range(num_frames):
            pred_i = ensure_numpy_array(predictions[i])
            truth_i = ensure_numpy_array(ground_truth[i])

            if pred_i.shape[0] >= 3 and truth_i.shape[0] >= 3:
                pred_mag_i = compute_velocity_magnitude(pred_i)
                truth_mag_i = compute_velocity_magnitude(truth_i)
                all_pred_mag.append(pred_mag_i)
                all_truth_mag.append(truth_mag_i)

        if all_pred_mag and all_truth_mag:
            global_vmin = min(np.min(all_pred_mag), np.min(all_truth_mag))
            global_vmax = max(np.max(all_pred_mag), np.max(all_truth_mag))
        else:
            global_vmin, global_vmax = pred_mag_0.min(), pred_mag_0.max()

        # Initialize plots
        im1 = axes[0, 0].imshow(truth_mag_0, cmap="plasma", aspect="auto", vmin=global_vmin, vmax=global_vmax)
        axes[0, 0].set_title("Ground Truth")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im2 = axes[0, 1].imshow(pred_mag_0, cmap="plasma", aspect="auto", vmin=global_vmin, vmax=global_vmax)
        axes[0, 1].set_title("Prediction")
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Error plot
        error_0 = np.abs(truth_mag_0 - pred_mag_0)
        im3 = axes[1, 0].imshow(error_0, cmap="Reds", aspect="auto", vmin=0, vmax=error_0.max())
        axes[1, 0].set_title("Absolute Error")
        axes[1, 0].axis("off")
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Text info
        axes[1, 1].axis("off")
        time_text = axes[1, 1].text(0.1, 0.8, "", transform=axes[1, 1].transAxes, fontsize=12)
        error_text = axes[1, 1].text(0.1, 0.6, "", transform=axes[1, 1].transAxes, fontsize=12)

        def animate(frame):
            """Animation function."""
            pred_frame = ensure_numpy_array(predictions[frame])
            truth_frame = ensure_numpy_array(ground_truth[frame])

            # Compute magnitudes
            if pred_frame.shape[0] >= 3 and truth_frame.shape[0] >= 3:
                pred_mag = compute_velocity_magnitude(pred_frame)
                truth_mag = compute_velocity_magnitude(truth_frame)
            else:
                pred_mag = pred_frame[0]
                truth_mag = truth_frame[0]

            error = np.abs(truth_mag - pred_mag)

            # Update images
            im1.set_data(truth_mag)
            im2.set_data(pred_mag)
            im3.set_data(error)
            im3.set_clim(vmin=0, vmax=error.max())

            # Update text
            mae = np.mean(error)
            rmse = np.sqrt(np.mean(error**2))
            time_text.set_text(f"Frame: {frame + 1}/{num_frames}")
            error_text.set_text(f"MAE: {mae:.5f}\nRMSE: {rmse:.5f}")

            return [im1, im2, im3, time_text, error_text]

        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=200, blit=True, repeat=True)

        # Save video
        video_path = self.videos_dir / f"{mode}_sample_{sample_idx}.mp4"

        try:
            writer = animation.FFMpegWriter(fps=5, metadata=dict(artist="Flow Evaluation"), bitrate=1800)
            anim.save(video_path, writer=writer)
            print(f"Video saved: {video_path}")
        except Exception as e:
            print(f"Failed to save MP4: {e}")
            # Fallback to GIF
            gif_path = self.videos_dir / f"{mode}_sample_{sample_idx}.gif"
            try:
                anim.save(gif_path, writer="pillow", fps=5)
                video_path = gif_path
                print(f"GIF saved: {video_path}")
            except Exception as e2:
                print(f"Failed to save GIF: {e2}")
                video_path = None

        plt.close(fig)
        return video_path
