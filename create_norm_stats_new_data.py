#!/usr/bin/env python3
"""
Create normalization statistics for new 10,000-frame dataset.
"""

import glob
import json
import os

import h5py
import numpy as np


def compute_multi_channel_normalization_stats(data_dir, field_names, scale, output_file):
    """Compute normalization statistics for multi-channel data."""

    print(f"Computing normalization stats for channels: {field_names}")
    print(f"Resolution scale: {scale}")

    # Use multi-channel files (u-v-w combined)
    field_pattern = "-".join(field_names)  # "u-v-w"
    pattern = f"{field_pattern}_scale{scale[0]}-{scale[1]}_ylayer2_ts*.h5"
    file_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"Error: No multi-channel files found with pattern: {pattern}")
        return None

    print(f"Found {len(files)} multi-channel files")

    all_values = []
    per_channel_values = [[] for _ in field_names]
    total_count = 0

    # Process first 1200 files for stats (from training set)
    for i, fpath in enumerate(files[:1200]):
        try:
            with h5py.File(fpath, "r") as f:
                data = f["data"][()]  # Shape: (3, H, W) for u,v,w

                if len(data.shape) != 3 or data.shape[0] != len(field_names):
                    print(f"Warning: Unexpected data shape {data.shape} in {fpath}")
                    continue

                # Collect all values (across all channels)
                all_values.append(data.flatten())
                total_count += data.size

                # Collect per-channel values
                for c in range(data.shape[0]):
                    per_channel_values[c].append(data[c].flatten())

                if i % 200 == 0:
                    print(f"  Processed {i + 1} files, shape: {data.shape}")

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    if not all_values:
        print("No valid data found!")
        return None

    # Combine all values
    all_values = np.concatenate(all_values)

    # Compute per-channel stats
    per_channel_stats = {}
    for i, field_name in enumerate(field_names):
        if per_channel_values[i]:
            channel_data = np.concatenate(per_channel_values[i])
            per_channel_stats[f"channel_{i:02d}"] = {
                "field_name": field_name,
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "min": float(np.min(channel_data)),
                "max": float(np.max(channel_data)),
            }
            print(
                f"  {field_name}: mean={per_channel_stats[f'channel_{i:02d}']['mean']:.6f}, "
                f"std={per_channel_stats[f'channel_{i:02d}']['std']:.6f}"
            )

    # Compute global statistics
    mean = float(np.mean(all_values))
    std = float(np.std(all_values))

    print("\nGlobal statistics:")
    print(f"  Mean: {mean:.8f}")
    print(f"  Std: {std:.8f}")
    print(f"  Min: {np.min(all_values):.8f}")
    print(f"  Max: {np.max(all_values):.8f}")

    # Save stats
    stats = {
        "mean": mean,
        "std": std,
        "count": total_count,
        "train_files": len(files),
        "field_names": field_names,
        "resolution_scale": scale,
        "per_channel_stats": per_channel_stats,
        "normalization_type": "per_channel",
        "description": "Per-channel normalization for new 10k-frame dataset",
    }

    output_path = os.path.join(data_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNormalization stats saved to: {output_path}")
    return stats


if __name__ == "__main__":
    data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow/new"
    field_names = ["u", "v", "w"]
    scale = [2, 3]  # (z, x) downsampling
    output_file = "norm_stats_3ch_1plane_u-v-w_scale2-3_ylayer2.json"

    stats = compute_multi_channel_normalization_stats(
        data_dir=data_dir, field_names=field_names, scale=scale, output_file=output_file
    )

    if stats:
        print("\n=== SUCCESS ===")
        print(f"Global mean: {stats['mean']:.8f}")
        print(f"Global std: {stats['std']:.8f}")
