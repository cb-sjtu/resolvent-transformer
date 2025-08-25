#!/usr/bin/env python3

import glob
import json
import os

import h5py
import numpy as np


def compute_normalization_stats(data_dir, pattern, output_file):
    """Compute normalization statistics for the actual data files."""

    # Find all matching files
    file_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(files)} files matching pattern")
    print(f"First few files: {files[:3]}")

    # Compute statistics
    all_values = []
    total_count = 0

    for i, fpath in enumerate(files[:100]):  # Process first 100 files for stats
        try:
            with h5py.File(fpath, "r") as f:
                data = f["data"][()]
                all_values.append(data.flatten())
                total_count += data.size

            if i % 20 == 0:
                print(f"Processed {i + 1} files, shape: {data.shape}")

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    if not all_values:
        print("No valid data found!")
        return

    # Concatenate all values
    all_values = np.concatenate(all_values)

    # Compute stats
    mean = float(np.mean(all_values))
    std = float(np.std(all_values))

    print("Computed statistics:")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
    print(f"  Min: {np.min(all_values)}")
    print(f"  Max: {np.max(all_values)}")
    print(f"  Total samples: {total_count}")

    # Save to file
    stats = {
        "mean": mean,
        "std": std,
        "count": total_count,
        "train_files": len(files),
        "field_name": "u",
        "resolution_scale": [1, 4, 4],  # Based on file pattern
        "y_slice": 48,  # Based on file pattern
    }

    output_path = os.path.join(data_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Normalization stats saved to: {output_path}")
    return stats


if __name__ == "__main__":
    data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"
    pattern = "u_scale1-4-4_yslice48_*.h5"
    output_file = "norm_stats_u_scale1-4-4_yslice48.json"

    compute_normalization_stats(data_dir, pattern, output_file)
