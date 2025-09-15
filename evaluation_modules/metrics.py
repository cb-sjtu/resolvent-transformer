#!/usr/bin/env python3
"""
Metrics calculation module for flow evaluation.
"""

import numpy as np
import torch

from .utils import compute_smart_relative_error, ensure_torch_tensor


class FlowMetrics:
    """Calculate various metrics for flow prediction evaluation."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def compute_basic_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute basic metrics (MSE, MAE, RMSE).

        Args:
            pred: Predicted tensor
            target: Target tensor

        Returns:
            Dictionary with computed metrics
        """
        pred = ensure_torch_tensor(pred)
        target = ensure_torch_tensor(target)

        # Compute absolute error
        abs_error = torch.abs(pred - target)
        squared_error = (pred - target) ** 2

        metrics = {
            "mse": torch.mean(squared_error).item(),
            "mae": torch.mean(abs_error).item(),
            "rmse": torch.sqrt(torch.mean(squared_error)).item(),
        }

        return metrics

    def compute_relative_errors(
        self, pred: torch.Tensor, target: torch.Tensor, channel_names: list[str] | None = None
    ) -> dict:
        """
        Compute various types of relative errors.

        Args:
            pred: Predicted tensor (C, H, W) or (..., C, H, W)
            target: Target tensor (same shape as pred)
            channel_names: Names of channels

        Returns:
            Dictionary with relative error metrics
        """
        return compute_smart_relative_error(pred, target, channel_names)

    def compute_magnitude_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute metrics for velocity magnitude.

        Args:
            pred: Predicted velocity tensor (C>=3, H, W)
            target: Target velocity tensor (C>=3, H, W)

        Returns:
            Dictionary with magnitude-specific metrics
        """
        pred = ensure_torch_tensor(pred)
        target = ensure_torch_tensor(target)

        # Extract numpy arrays for magnitude computation
        pred_np = pred.cpu().numpy() if pred.requires_grad else pred.numpy()
        target_np = target.cpu().numpy() if target.requires_grad else target.numpy()

        try:
            # Compute magnitudes
            from .utils import compute_velocity_magnitude

            pred_mag = compute_velocity_magnitude(pred_np)
            target_mag = compute_velocity_magnitude(target_np)

            # Convert back to tensors for metric computation
            pred_mag_tensor = torch.from_numpy(pred_mag)
            target_mag_tensor = torch.from_numpy(target_mag)

            # Compute basic metrics for magnitude
            mag_metrics = self.compute_basic_metrics(pred_mag_tensor, target_mag_tensor)

            # Add magnitude-specific relative error
            mag_abs_error = np.abs(pred_mag - target_mag)
            target_mag_rms = np.sqrt(np.mean(target_mag**2))
            mag_rel_error = np.mean(mag_abs_error) / (target_mag_rms + 1e-8)

            mag_metrics["magnitude_relative_error"] = mag_rel_error

            return {"magnitude": mag_metrics}

        except Exception as e:
            print(f"Warning: Could not compute magnitude metrics: {e}")
            return {"magnitude": {"error": str(e)}}

    def compute_channel_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, channel_names: list[str] | None = None
    ) -> dict:
        """
        Compute metrics for each channel separately.

        Args:
            pred: Predicted tensor (C, H, W)
            target: Target tensor (C, H, W)
            channel_names: Names of channels

        Returns:
            Dictionary with per-channel metrics
        """
        pred = ensure_torch_tensor(pred)
        target = ensure_torch_tensor(target)

        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(pred.shape[0])]

        channel_metrics = {}

        for i, channel_name in enumerate(channel_names[: pred.shape[0]]):
            pred_channel = pred[i : i + 1]  # Keep dimension for consistency
            target_channel = target[i : i + 1]

            # Basic metrics
            basic_metrics = self.compute_basic_metrics(pred_channel, target_channel)

            # Relative error metrics
            rel_metrics = self.compute_relative_errors(pred_channel, target_channel, [channel_name])

            # Combine metrics
            channel_metrics[channel_name] = {**basic_metrics, **rel_metrics.get(channel_name, {})}

        return channel_metrics

    def compute_comprehensive_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, channel_names: list[str] | None = None
    ) -> dict:
        """
        Compute all available metrics for the prediction.

        Args:
            pred: Predicted tensor (C, H, W)
            target: Target tensor (C, H, W)
            channel_names: Names of channels (default: ["u", "v", "w"])

        Returns:
            Comprehensive metrics dictionary
        """
        if channel_names is None:
            channel_names = ["u", "v", "w"]

        metrics = {}

        # Overall metrics (all channels combined)
        metrics["overall"] = self.compute_basic_metrics(pred, target)

        # Per-channel metrics
        metrics["channels"] = self.compute_channel_metrics(pred, target, channel_names)

        # Relative error analysis
        metrics["relative_errors"] = self.compute_relative_errors(pred, target, channel_names)

        # Magnitude metrics (if we have at least 3 channels)
        if pred.shape[0] >= 3:
            magnitude_metrics = self.compute_magnitude_metrics(pred, target)
            metrics.update(magnitude_metrics)

        return metrics

    def format_metrics_summary(self, metrics: dict, precision: int = 5) -> str:
        """
        Format metrics into a readable summary string.

        Args:
            metrics: Metrics dictionary
            precision: Number of decimal places

        Returns:
            Formatted string summary
        """
        summary_parts = []

        # Overall metrics
        if "overall" in metrics:
            overall = metrics["overall"]
            summary_parts.append(
                f"Overall - MSE: {overall.get('mse', 0):.{precision}f}, MAE: {overall.get('mae', 0):.{precision}f}"
            )

        # Channel summaries
        if "channels" in metrics:
            for channel, channel_metrics in metrics["channels"].items():
                mse = channel_metrics.get("mse", 0)
                mae = channel_metrics.get("mae", 0)
                summary_parts.append(f"{channel.upper()} - MSE: {mse:.{precision}f}, MAE: {mae:.{precision}f}")

        # Magnitude metrics
        if "magnitude" in metrics and isinstance(metrics["magnitude"], dict):
            mag_metrics = metrics["magnitude"]
            if "mse" in mag_metrics:
                summary_parts.append(
                    f"Magnitude - MSE: {mag_metrics['mse']:.{precision}f}, MAE: {mag_metrics['mae']:.{precision}f}"
                )

        return " | ".join(summary_parts)

    def log_metrics_to_wandb(self, metrics: dict, prefix: str = "", step: int | None = None):
        """
        Log metrics to wandb if available.

        Args:
            metrics: Metrics dictionary
            prefix: Prefix for metric names
            step: Step number for logging
        """
        try:
            import wandb

            if wandb.run is not None:
                # Flatten metrics dictionary for wandb logging
                flat_metrics = {}

                def flatten_dict(d, parent_key=""):
                    for k, v in d.items():
                        new_key = f"{parent_key}/{k}" if parent_key else k
                        if prefix:
                            new_key = f"{prefix}/{new_key}"

                        if isinstance(v, dict):
                            flat_metrics.update(flatten_dict(v, new_key))
                        elif isinstance(v, (int, float)):
                            flat_metrics[new_key] = v

                flatten_dict(metrics)

                if step is not None:
                    wandb.log(flat_metrics, step=step)
                else:
                    wandb.log(flat_metrics)

        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Failed to log metrics to wandb: {e}")
