#!/usr/bin/env python3
"""
Utility functions for flow evaluation.
"""

from pathlib import Path

import numpy as np
import torch


def compute_velocity_magnitude(velocity_data):
    """Compute velocity magnitude from u, v, w components.

    Args:
        velocity_data: Velocity data with shape (C, H, W) where C >= 3 for u, v, w

    Returns:
        Velocity magnitude with shape (H, W)
    """
    if isinstance(velocity_data, torch.Tensor):
        velocity_data = velocity_data.cpu().numpy()

    if velocity_data.ndim != 3 or velocity_data.shape[0] < 3:
        raise ValueError(
            f"Expected 3D array with at least 3 channels, got shape {velocity_data.shape}"
        )

    u, v, w = velocity_data[0], velocity_data[1], velocity_data[2]
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    return magnitude


def ensure_torch_tensor(data):
    """Ensure data is a torch tensor."""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return data


def ensure_numpy_array(data):
    """Ensure data is a numpy array."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return data


def create_output_directory(base_dir: str, sub_dir: str = None) -> Path:
    """Create output directory and return Path object."""
    output_dir = Path(base_dir)
    if sub_dir:
        output_dir = output_dir / sub_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def format_metrics_string(metrics_dict: dict, precision: int = 5) -> str:
    """Format metrics dictionary into a readable string."""
    formatted_parts = []
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            # Nested dictionary
            nested_parts = []
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, int | float):
                    nested_parts.append(f"{nested_key}: {nested_value:.{precision}f}")
                else:
                    nested_parts.append(f"{nested_key}: {nested_value}")
            formatted_parts.append(f"{key}: {{{', '.join(nested_parts)}}}")
        elif isinstance(value, int | float):
            formatted_parts.append(f"{key}: {value:.{precision}f}")
        else:
            formatted_parts.append(f"{key}: {value}")

    return " | ".join(formatted_parts)


def get_default_monitor_points(shape: tuple = (256, 256)) -> list:
    """
    Get default monitoring points distributed across the domain.

    Args:
        shape: (H, W) shape of the domain

    Returns:
        List of (z_index, x_index) tuples
    """
    H, W = shape

    # Create a 3x3 grid plus one center point
    points = []

    # Grid points (avoiding exact edges)
    z_positions = [H // 4, H // 2, 3 * H // 4]
    x_positions = [W // 4, W // 2, 3 * W // 4]

    for z in z_positions:
        for x in x_positions:
            points.append((z, x))

    # Add one additional point
    points.append((H // 8, W // 8))

    return points[:10]  # Limit to 10 points


def log_image_to_wandb(wandb_logger, key: str, image_path: Path, caption: str = None):
    """Log image to wandb if available."""
    try:
        import wandb

        if wandb_logger is not None:
            wandb_logger.log({key: wandb.Image(str(image_path), caption=caption)})
    except ImportError:
        pass


def save_tensor_as_numpy(tensor: torch.Tensor, filepath: Path):
    """Save tensor as numpy array."""
    numpy_data = ensure_numpy_array(tensor)
    np.save(filepath, numpy_data)


def load_numpy_as_tensor(filepath: Path) -> torch.Tensor:
    """Load numpy array as tensor."""
    numpy_data = np.load(filepath)
    return torch.from_numpy(numpy_data)


def compute_smart_relative_error(pred, target, channel_names=None):
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
    pred = ensure_torch_tensor(pred)
    target = ensure_torch_tensor(target)

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
