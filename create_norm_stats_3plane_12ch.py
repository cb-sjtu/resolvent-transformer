#!/usr/bin/env python3
"""
Create normalization statistics for 12-channel (3-plane × 4-field) flow data.
This script computes per-channel statistics for proper normalization.
Each channel gets its own mean/std for precise normalization.
"""

import glob
import json
import os

import h5py
import numpy as np


def compute_12_channel_normalization_stats(data_dir, field_names, scale, y_slices, output_file):
    """Compute normalization statistics for 12-channel 3-plane data.

    Args:
        data_dir: Directory containing the data files
        field_names: List of field names ['u', 'v', 'w', 'p']
        scale: Resolution scale tuple (z, x, y)
        y_slices: List of 3 y-slices [29, 54, 75]
        output_file: Output JSON file name
    """

    print("Computing normalization stats for 3-plane 12-channel data")
    print(f"Fields: {field_names}")
    print(f"Y-slices: {y_slices}")
    print(f"Resolution scale: {scale}")

    # Use multi-channel files (u-v-w-p combined)
    field_pattern = "-".join(field_names)  # "u-v-w-p"
    pattern = f"{field_pattern}_scale{scale[0]}-{scale[1]}-{scale[2]}_yslice*.h5"
    file_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"Error: No multi-channel files found with pattern: {pattern}")
        return None

    print(f"Found {len(files)} multi-channel files")

    # Initialize storage for each of the 12 channels
    # Channel order: [plane0_u, plane0_v, plane0_w, plane0_p,
    #                 plane1_u, plane1_v, plane1_w, plane1_p,
    #                 plane2_u, plane2_v, plane2_w, plane2_p]
    num_channels = len(field_names) * len(y_slices)  # 4 * 3 = 12
    channel_values = [[] for _ in range(num_channels)]
    total_count = 0

    # Process files - ensure we get files from each y_slice
    files_per_slice = {}
    for y_slice in y_slices:
        files_per_slice[y_slice] = [f for f in files if f"yslice{y_slice}" in f][:1000]  # 100 files per slice

    processed_files = []
    for y_slice in y_slices:
        processed_files.extend(files_per_slice[y_slice])

    for i, fpath in enumerate(processed_files):
        try:
            with h5py.File(fpath, "r") as f:
                # Load multi-channel data: shape (C, H, W) where C=4 for u,v,w,p
                data_multi_channel = f["data"][()]  # Shape: (4, H, W)

                # Verify shape
                if len(data_multi_channel.shape) != 3 or data_multi_channel.shape[0] != len(field_names):
                    print(f"Warning: Expected shape (4, H, W), got {data_multi_channel.shape} in {fpath}")
                    continue

                # This is 2D data for one y-plane. We need to determine which y-plane this file corresponds to.
                # From filename: extract yslice info
                import re

                match = re.search(r"yslice(\d+)", fpath)
                if not match:
                    print(f"Warning: Could not extract y_slice from filename {fpath}")
                    continue

                file_y_slice = int(match.group(1))

                # Find which plane this corresponds to
                try:
                    plane_idx = y_slices.index(file_y_slice)
                except ValueError:
                    # This file doesn't correspond to one of our selected y_slices
                    continue

                # Extract data for each channel in this plane
                for field_idx, _field in enumerate(field_names):
                    channel_idx = plane_idx * len(field_names) + field_idx

                    # Get 2D data for this field
                    data_2d = data_multi_channel[field_idx]  # Shape: (H, W)
                    channel_values[channel_idx].append(data_2d.flatten())

                total_count += data_multi_channel.size

                if i % 20 == 0:
                    print(f"  Processed {i + 1} files")

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    if not any(channel_values):
        print("No valid data found!")
        return None

    # Compute statistics for each channel
    print("\nComputing per-channel statistics...")

    channel_stats = []
    channel_names = []

    # Build channel names and compute stats
    for plane_idx, y_slice in enumerate(y_slices):
        for field_idx, field in enumerate(field_names):
            channel_idx = plane_idx * len(field_names) + field_idx
            channel_name = f"plane{plane_idx}_{field}_y{y_slice}"
            channel_names.append(channel_name)

            if channel_values[channel_idx]:
                # Combine all values for this channel across all files
                channel_data = np.concatenate(channel_values[channel_idx])

                stats = {
                    "mean": float(np.mean(channel_data)),
                    "std": float(np.std(channel_data)),
                    "min": float(np.min(channel_data)),
                    "max": float(np.max(channel_data)),
                    "count": int(len(channel_data)),
                }

                print(f"  {channel_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
                channel_stats.append(stats)
            else:
                print(f"  {channel_name}: No data!")
                channel_stats.append(None)

    # Create per-channel stats dict for compatibility with existing code
    per_channel_stats = {}
    for i, (_channel_name, stats) in enumerate(zip(channel_names, channel_stats, strict=False)):
        if stats is not None:
            per_channel_stats[f"channel_{i:02d}"] = stats

    # Compute global stats for reference (not used in normalization)
    all_values = []
    for channel_data_list in channel_values:
        if channel_data_list:
            all_values.extend([data for batch in channel_data_list for data in batch.flatten()])

    if all_values:
        global_mean = float(np.mean(all_values))
        global_std = float(np.std(all_values))
        print(f"\nGlobal stats (reference only): mean={global_mean:.6f}, std={global_std:.6f}")
    else:
        global_mean = 0.0
        global_std = 1.0

    # Save complete statistics
    stats = {
        "mean": global_mean,  # For compatibility, not used
        "std": global_std,  # For compatibility, not used
        "per_channel_stats": per_channel_stats,
        "channel_names": channel_names,
        "num_channels": num_channels,
        "num_planes": len(y_slices),
        "field_names": field_names,
        "y_slices": y_slices,
        "resolution_scale": scale,
        "train_files": len(files),
        "total_samples": total_count,
        "normalization_type": "per_channel_12ch",
        "description": f"Per-channel normalization for 12 channels (3 planes × {len(field_names)} fields)",
    }

    output_path = os.path.join(data_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n12-channel normalization stats saved to: {output_path}")
    return stats


if __name__ == "__main__":
    data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"
    field_names = ["u", "v", "w", "p"]
    scale = [4, 6, 1]  # (z, x, y) downsampling used in 3-plane data
    y_slices = [29, 54, 75]  # The 3 y-slices available
    output_file = "norm_stats_12ch_3plane_u-v-w-p_scale4-6-1.json"

    stats = compute_12_channel_normalization_stats(
        data_dir=data_dir, field_names=field_names, scale=scale, y_slices=y_slices, output_file=output_file
    )

    if stats:
        print("\n=== SUCCESS ===")
        print("12-channel normalization stats created!")
        print(f"Use this file in your training: {output_file}")
        print(f"Total channels: {stats['num_channels']}")
        print(f"Channel names: {stats['channel_names'][:6]}... (showing first 6)")
    else:
        print("Failed to create normalization stats")
