#!/usr/bin/env python3
"""
Create normalization statistics for multi-channel (u,v,w) flow data.
This script computes combined statistics across all channels for proper normalization.
"""

import glob
import json
import os

import h5py
import numpy as np


def compute_multi_channel_normalization_stats(data_dir, field_names, scale, y_slice, output_file):
    """Compute normalization statistics for multi-channel data.

    Args:
        data_dir: Directory containing the data files
        field_names: List of field names ['u', 'v', 'w']
        scale: Resolution scale tuple (z, x, y)
        y_slice: Y slice number
        output_file: Output JSON file name
    """

    print(f"Computing normalization stats for channels: {field_names}")
    print(f"Resolution scale: {scale}, Y slice: {y_slice}")

    # Use multi-channel files (u-v-w combined)
    field_pattern = "-".join(field_names)  # "u-v-w"
    pattern = f"{field_pattern}_scale{scale[0]}-{scale[1]}-{scale[2]}_yslice{y_slice}_*.h5"
    file_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"Error: No multi-channel files found with pattern: {pattern}")
        return None

    print(f"Found {len(files)} multi-channel files")

    all_values = []
    per_channel_values = [[] for _ in field_names]
    total_count = 0

    # Process multi-channel files
    for i, fpath in enumerate(files[:1200]):  # Process more files for better stats
        try:
            with h5py.File(fpath, "r") as f:
                data = f["data"][()]  # Shape: (3, H, W) for u,v,w

                # Verify shape
                if len(data.shape) != 3 or data.shape[0] != len(field_names):
                    print(f"Warning: Unexpected data shape {data.shape} in {fpath}")
                    continue

                # Collect all values (across all channels)
                all_values.append(data.flatten())
                total_count += data.size

                # Collect per-channel values
                for c in range(data.shape[0]):
                    per_channel_values[c].append(data[c].flatten())

                if i % 20 == 0:
                    print(f"  Processed {i + 1} files, shape: {data.shape}")

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    if not all_values:
        print("No valid data found!")
        return None

    # Combine all values across all files and channels
    all_values = np.concatenate(all_values)

    # Combine per-channel values
    per_channel_data = []
    for c in range(len(field_names)):
        if per_channel_values[c]:
            channel_data = np.concatenate(per_channel_values[c])
            per_channel_data.append(channel_data)
            print(f"  {field_names[c]} stats: mean={np.mean(channel_data):.6f}, std={np.std(channel_data):.6f}")
        else:
            per_channel_data.append(None)

    # Compute global statistics
    print("\nComputing global statistics across all channels...")

    # Compute global stats
    mean = float(np.mean(all_values))
    std = float(np.std(all_values))

    print("\nGlobal statistics across all channels:")
    print(f"  Mean: {mean:.8f}")
    print(f"  Std: {std:.8f}")
    print(f"  Min: {np.min(all_values):.8f}")
    print(f"  Max: {np.max(all_values):.8f}")
    print(f"  Total samples: {total_count}")

    # Compute per-channel statistics for reference
    per_channel_stats = {}
    for i, field_name in enumerate(field_names):
        if i < len(per_channel_data) and per_channel_data[i] is not None:
            channel_data = per_channel_data[i]
            per_channel_stats[field_name] = {
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "min": float(np.min(channel_data)),
                "max": float(np.max(channel_data)),
            }

    print("\nPer-channel statistics:")
    for field_name, stats in per_channel_stats.items():
        print(f"  {field_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    # Save global stats (used for normalization)
    stats = {
        "mean": mean,
        "std": std,
        "count": total_count,
        "train_files": len(files),
        "field_names": field_names,
        "resolution_scale": scale,
        "y_slice": y_slice,
        "per_channel_stats": per_channel_stats,
        "normalization_type": "global_across_channels",
        "description": f"Global normalization statistics computed across all {len(field_names)} channels",
    }

    output_path = os.path.join(data_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNormalization stats saved to: {output_path}")
    return stats


if __name__ == "__main__":
    data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"
    field_names = ["u", "v", "w"]
    scale = [2, 3, 1]  # (z, x, y) downsampling used in training
    y_slice = 54
    output_file = "norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json"

    stats = compute_multi_channel_normalization_stats(
        data_dir=data_dir, field_names=field_names, scale=scale, y_slice=y_slice, output_file=output_file
    )

    if stats:
        print("\n=== SUCCESS ===")
        print("Multi-channel normalization stats created!")
        print(f"Use this file in your training/evaluation: {output_file}")
        print(f"Global mean: {stats['mean']:.8f}")
        print(f"Global std: {stats['std']:.8f}")
    else:
        print("Failed to create normalization stats")
