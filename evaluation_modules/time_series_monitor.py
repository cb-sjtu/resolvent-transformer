#!/usr/bin/env python3
"""
Time Series Point Monitor

This module provides functionality to monitor specific points in the flow field
over time and generate time series plots for u, v, w components and velocity magnitude.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import compute_velocity_magnitude


class TimeSeriesMonitor:
    """Monitor specific points in flow field over time."""

    def __init__(self, monitor_points: list[tuple[int, int]] | None = None):
        """
        Initialize time series monitor.

        Args:
            monitor_points: List of (z_index, x_index) tuples for monitoring points.
                          If None, will use default 10 points distributed across domain.
        """
        self.monitor_points = monitor_points or self._get_default_monitor_points()
        self.time_series_data = {}
        self.reset_data()

    def _get_default_monitor_points(self) -> list[tuple[int, int]]:
        """Get default monitoring points distributed across the domain."""
        # Default points for 256x256 domain (can be adjusted)
        default_points = [
            (64, 64),  # Lower-left region
            (64, 128),  # Lower-center
            (64, 192),  # Lower-right region
            (128, 64),  # Mid-left
            (128, 128),  # Center
            (128, 192),  # Mid-right
            (192, 64),  # Upper-left region
            (192, 128),  # Upper-center
            (192, 192),  # Upper-right region
            (96, 96),  # Additional point
        ]
        return default_points

    def reset_data(self):
        """Reset all time series data."""
        self.time_series_data = {
            "train": {"ar": {}, "tf": {}},
            "val": {"ar": {}, "tf": {}},
            "test": {"ar": {}, "tf": {}},
        }

        for split in self.time_series_data:
            for mode in self.time_series_data[split]:
                self.time_series_data[split][mode] = {
                    "u_pred": {i: [] for i in range(len(self.monitor_points))},
                    "v_pred": {i: [] for i in range(len(self.monitor_points))},
                    "w_pred": {i: [] for i in range(len(self.monitor_points))},
                    "mag_pred": {i: [] for i in range(len(self.monitor_points))},
                    "u_gt": {i: [] for i in range(len(self.monitor_points))},
                    "v_gt": {i: [] for i in range(len(self.monitor_points))},
                    "w_gt": {i: [] for i in range(len(self.monitor_points))},
                    "mag_gt": {i: [] for i in range(len(self.monitor_points))},
                    "timesteps": [],
                }

    def set_monitor_points(self, points: list[tuple[int, int]]):
        """
        Update monitoring points.

        Args:
            points: List of (z_index, x_index) tuples
        """
        self.monitor_points = points
        self.reset_data()
        print(f"Updated monitoring points: {self.monitor_points}")

    def extract_point_values(self, data: torch.Tensor, timestep: int = None) -> dict:
        """
        Extract values at monitoring points from flow data.

        Args:
            data: Flow data tensor with shape (C, H, W) where C includes u, v, w
            timestep: Current timestep (for logging)

        Returns:
            Dictionary with extracted values for each point and component
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Ensure data has at least 3 channels (u, v, w)
        if data.shape[0] < 3:
            raise ValueError(f"Data must have at least 3 channels (u, v, w), got {data.shape[0]}")

        point_values = {"u": [], "v": [], "w": [], "mag": []}

        # Compute velocity magnitude
        magnitude = compute_velocity_magnitude(data)

        # Extract values at each monitoring point
        for i, (z_idx, x_idx) in enumerate(self.monitor_points):
            # Ensure indices are within bounds
            z_idx = min(z_idx, data.shape[1] - 1)
            x_idx = min(x_idx, data.shape[2] - 1)

            u_val = data[0, z_idx, x_idx]
            v_val = data[1, z_idx, x_idx]
            w_val = data[2, z_idx, x_idx]
            mag_val = magnitude[z_idx, x_idx]

            point_values["u"].append(float(u_val))
            point_values["v"].append(float(v_val))
            point_values["w"].append(float(w_val))
            point_values["mag"].append(float(mag_val))

        return point_values

    def record_timestep(
        self, pred_data: torch.Tensor, split: str, mode: str, timestep: int, gt_data: torch.Tensor = None
    ):
        """
        Record prediction and ground truth data for current timestep.

        Args:
            pred_data: Prediction flow data tensor (C, H, W)
            split: 'train', 'val', or 'test'
            mode: 'ar' (autoregressive) or 'tf' (teacher forcing)
            timestep: Current timestep number
            gt_data: Ground truth flow data tensor (C, H, W), optional
        """
        # Extract prediction values
        pred_values = self.extract_point_values(pred_data, timestep)

        # Record timestep if not already recorded for this mode
        if timestep not in self.time_series_data[split][mode]["timesteps"]:
            self.time_series_data[split][mode]["timesteps"].append(timestep)

        # Record prediction values for each point
        for component in ["u", "v", "w", "mag"]:
            for i, value in enumerate(pred_values[component]):
                self.time_series_data[split][mode][f"{component}_pred"][i].append(value)

        # Record ground truth values if available
        if gt_data is not None:
            gt_values = self.extract_point_values(gt_data, timestep)
            for component in ["u", "v", "w", "mag"]:
                for i, value in enumerate(gt_values[component]):
                    self.time_series_data[split][mode][f"{component}_gt"][i].append(value)
        else:
            # Fill with None if no ground truth available
            for component in ["u", "v", "w", "mag"]:
                for i in range(len(self.monitor_points)):
                    self.time_series_data[split][mode][f"{component}_gt"][i].append(None)

    def plot_point_time_series(
        self, point_idx: int, output_dir: Path, split: str = "test", show_both_modes: bool = True
    ):
        """
        Plot time series for a specific monitoring point.

        Args:
            point_idx: Index of monitoring point (0 to len(monitor_points)-1)
            output_dir: Directory to save plots
            split: Which data split to plot
            show_both_modes: Whether to show both AR and TF modes
        """
        if point_idx >= len(self.monitor_points):
            raise ValueError(f"Point index {point_idx} out of range")

        z_idx, x_idx = self.monitor_points[point_idx]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Time Series at Point ({z_idx}, {x_idx}) - {split.upper()} Data", fontsize=16)

        components = ["u", "v", "w", "mag"]
        colors = {"ar": "blue", "tf": "red"}
        labels = {"ar": "Autoregressive", "tf": "Teacher Forcing"}

        for idx, component in enumerate(components):
            ax = axes[idx // 2, idx % 2]

            modes_to_plot = ["ar", "tf"] if show_both_modes else ["ar"]

            for mode in modes_to_plot:
                if split in self.time_series_data and mode in self.time_series_data[split]:
                    timesteps = self.time_series_data[split][mode]["timesteps"]
                    pred_key = f"{component}_pred"
                    gt_key = f"{component}_gt"

                    # Plot predictions
                    if (
                        pred_key in self.time_series_data[split][mode]
                        and point_idx in self.time_series_data[split][mode][pred_key]
                    ):
                        pred_values = self.time_series_data[split][mode][pred_key][point_idx]
                        if timesteps and pred_values and len(timesteps) == len(pred_values):
                            ax.plot(
                                timesteps,
                                pred_values,
                                color=colors[mode],
                                alpha=0.8,
                                linewidth=2,
                                label=f"{labels[mode]} (Pred)",
                                marker="o",
                                markersize=4,
                            )

                    # Plot ground truth if available
                    if (
                        gt_key in self.time_series_data[split][mode]
                        and point_idx in self.time_series_data[split][mode][gt_key]
                    ):
                        gt_values = self.time_series_data[split][mode][gt_key][point_idx]
                        # Filter out None values
                        valid_gt = [(t, v) for t, v in zip(timesteps, gt_values, strict=False) if v is not None]
                        if valid_gt:
                            gt_timesteps, gt_vals = zip(*valid_gt, strict=False)
                            ax.plot(
                                gt_timesteps,
                                gt_vals,
                                color="green",
                                alpha=0.6,
                                linewidth=3,
                                label="Ground Truth",
                                linestyle="--",
                                marker="s",
                                markersize=3,
                            )

            ax.set_xlabel("Timestep")
            ax.set_ylabel(f"{component.upper()} {'Magnitude' if component == 'mag' else 'Velocity'}")
            ax.set_title(f"{component.upper()} Component")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"time_series_point_{point_idx}_z{z_idx}_x{x_idx}_{split}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Time series plot saved: {plot_path}")
        return plot_path

    def plot_all_points_component(self, component: str, output_dir: Path, split: str = "test", mode: str = "ar"):
        """
        Plot time series for all monitoring points for a specific component.

        Args:
            component: 'u', 'v', 'w', or 'mag'
            output_dir: Directory to save plots
            split: Which data split to plot
            mode: 'ar' or 'tf'
        """
        if component not in ["u", "v", "w", "mag"]:
            raise ValueError(f"Component must be one of ['u', 'v', 'w', 'mag'], got {component}")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Color map for different points
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.monitor_points)))

        for i, (z_idx, x_idx) in enumerate(self.monitor_points):
            if split in self.time_series_data and mode in self.time_series_data[split]:
                timesteps = self.time_series_data[split][mode]["timesteps"]
                pred_key = f"{component}_pred"
                gt_key = f"{component}_gt"

                # Plot prediction line
                if pred_key in self.time_series_data[split][mode] and i in self.time_series_data[split][mode][pred_key]:
                    pred_values = self.time_series_data[split][mode][pred_key][i]
                    if timesteps and pred_values and len(timesteps) == len(pred_values):
                        ax.plot(
                            timesteps,
                            pred_values,
                            color=colors[i],
                            alpha=0.8,
                            linewidth=2,
                            label=f"Point {i}: ({z_idx}, {x_idx}) Pred",
                            marker="o",
                            markersize=3,
                        )

                # Plot ground truth line if available
                if gt_key in self.time_series_data[split][mode] and i in self.time_series_data[split][mode][gt_key]:
                    gt_values = self.time_series_data[split][mode][gt_key][i]
                    valid_gt = [(t, v) for t, v in zip(timesteps, gt_values, strict=False) if v is not None]
                    if valid_gt:
                        gt_timesteps, gt_vals = zip(*valid_gt, strict=False)
                        ax.plot(
                            gt_timesteps,
                            gt_vals,
                            color=colors[i],
                            alpha=0.6,
                            linewidth=2,
                            label=f"Point {i}: ({z_idx}, {x_idx}) GT",
                            linestyle="--",
                            marker="s",
                            markersize=3,
                        )

        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"{component.upper()} {'Magnitude' if component == 'mag' else 'Velocity'}")
        ax.set_title(f"{component.upper()} Component - All Monitor Points ({split.upper()}, {mode.upper()})")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"time_series_all_points_{component}_{split}_{mode}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"All points {component} time series plot saved: {plot_path}")
        return plot_path

    def generate_all_plots(self, output_dir: Path, split: str = "test"):
        """
        Generate all time series plots.

        Args:
            output_dir: Directory to save all plots
            split: Which data split to plot
        """
        output_dir = Path(output_dir)
        ts_dir = output_dir / "time_series_plots"
        ts_dir.mkdir(exist_ok=True, parents=True)

        print(f"Generating time series plots for {len(self.monitor_points)} monitoring points...")

        # Plot individual points
        for i in range(len(self.monitor_points)):
            self.plot_point_time_series(i, ts_dir, split)

        # Plot all points for each component
        for component in ["u", "v", "w", "mag"]:
            for mode in ["ar", "tf"]:
                self.plot_all_points_component(component, ts_dir, split, mode)

        print(f"All time series plots saved to: {ts_dir}")
        return ts_dir

    def save_data_csv(self, output_dir: Path, split: str = "test"):
        """
        Save time series data to CSV files.

        Args:
            output_dir: Directory to save CSV files
            split: Which data split to save
        """
        import pandas as pd

        output_dir = Path(output_dir)
        csv_dir = output_dir / "time_series_data"
        csv_dir.mkdir(exist_ok=True, parents=True)

        for mode in ["ar", "tf"]:
            if split in self.time_series_data and mode in self.time_series_data[split]:
                data_dict = {"timestep": self.time_series_data[split][mode]["timesteps"]}

                # Add data for each point and component (prediction and ground truth)
                num_timesteps = len(data_dict["timestep"])
                for component in ["u", "v", "w", "mag"]:
                    for i, (z_idx, x_idx) in enumerate(self.monitor_points):
                        # Prediction data
                        pred_col_name = f"{component}_pred_point{i}_z{z_idx}_x{x_idx}"
                        pred_key = f"{component}_pred"
                        if (
                            pred_key in self.time_series_data[split][mode]
                            and i in self.time_series_data[split][mode][pred_key]
                        ):
                            point_data = self.time_series_data[split][mode][pred_key][i]
                            # Ensure all arrays have the same length
                            if len(point_data) < num_timesteps:
                                point_data = (
                                    point_data + [point_data[-1]] * (num_timesteps - len(point_data))
                                    if point_data
                                    else [0.0] * num_timesteps
                                )
                            elif len(point_data) > num_timesteps:
                                point_data = point_data[:num_timesteps]
                            data_dict[pred_col_name] = point_data
                        else:
                            data_dict[pred_col_name] = [0.0] * num_timesteps

                        # Ground truth data
                        gt_col_name = f"{component}_gt_point{i}_z{z_idx}_x{x_idx}"
                        gt_key = f"{component}_gt"
                        if (
                            gt_key in self.time_series_data[split][mode]
                            and i in self.time_series_data[split][mode][gt_key]
                        ):
                            gt_data = self.time_series_data[split][mode][gt_key][i]
                            # Ensure all arrays have the same length
                            if len(gt_data) < num_timesteps:
                                gt_data = gt_data + [gt_data[-1] if gt_data else None] * (num_timesteps - len(gt_data))
                            elif len(gt_data) > num_timesteps:
                                gt_data = gt_data[:num_timesteps]
                            data_dict[gt_col_name] = gt_data
                        else:
                            data_dict[gt_col_name] = [None] * num_timesteps

                # Create DataFrame and save
                if data_dict["timestep"]:  # Only save if we have data
                    try:
                        df = pd.DataFrame(data_dict)
                        csv_path = csv_dir / f"time_series_{split}_{mode}.csv"
                        df.to_csv(csv_path, index=False)
                        print(f"Time series data saved: {csv_path}")
                    except ValueError as e:
                        print(f"Warning: Failed to save CSV for {split}_{mode}: {e}")
                        # Debug information
                        for key, values in data_dict.items():
                            print(f"  {key}: {len(values)} values")
                            if key == "timestep":
                                print(f"    {values[:5]}...")  # First 5 timesteps

        return csv_dir
