"""
Compute global normalization statistics for flow_2d dataset.
This script calculates mean and std from the training set only.
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np
from tqdm import tqdm


def compute_global_stats(
    data_dir: str,
    field_name: str = "u",
    resolution_scale: tuple = (2, 3, 1),
    y_slice: int = 5,
    train_ratio: float = 0.7,
):
    """
    Compute global mean and std from training data only.

    Args:
        data_dir: Directory containing preprocessed HDF5 files
        field_name: Field name (e.g., "u")
        resolution_scale: Resolution scale tuple
        y_slice: Y slice used in preprocessing
        train_ratio: Ratio of data used for training

    Returns:
        dict: {"mean": float, "std": float}
    """
    # Get all preprocessed files
    pattern = f"{field_name}_scale{resolution_scale[0]}-{resolution_scale[1]}-{resolution_scale[2]}_yslice{y_slice}_*.h5"
    file_list = sorted(glob.glob(os.path.join(data_dir, pattern)))

    if not file_list:
        raise ValueError(f"No files found with pattern: {pattern}")

    print(f"Found {len(file_list)} files")

    # Calculate training files only
    train_files = int(len(file_list) * train_ratio)
    train_file_list = file_list[:train_files]

    print(f"Using {len(train_file_list)} files for statistics computation")

    # Compute statistics using Welford's online algorithm for numerical stability
    count = 0
    mean = 0.0
    M2 = 0.0

    for filepath in tqdm(train_file_list, desc="Computing stats"):
        try:
            with h5py.File(filepath, "r") as f:
                data = f["data"][()]  # Load the 2D data

                # Flatten the data to 1D for global statistics
                flat_data = data.flatten()

                # Update running statistics using Welford's algorithm
                for value in flat_data:
                    count += 1
                    delta = value - mean
                    mean += delta / count
                    delta2 = value - mean
                    M2 += delta * delta2

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    if count == 0:
        raise ValueError("No valid data found")

    # Calculate final statistics
    variance = M2 / count
    std = np.sqrt(variance)

    stats = {
        "mean": float(mean),
        "std": float(std),
        "count": count,
        "train_files": len(train_file_list),
        "field_name": field_name,
        "resolution_scale": resolution_scale,
        "y_slice": y_slice,
    }

    print("\nGlobal Statistics:")
    print(f"Mean: {stats['mean']:.6f}")
    print(f"Std:  {stats['std']:.6f}")
    print(f"Count: {stats['count']} pixels")
    print(f"Files: {stats['train_files']} training files")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for flow_2d dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sh/CB/icon-thewell-dev/data/preprocessed_flow",
        help="Directory containing preprocessed HDF5 files",
    )
    parser.add_argument("--field_name", type=str, default="u", help="Field name")
    parser.add_argument(
        "--resolution_scale",
        type=int,
        nargs=3,
        default=[2, 3, 1],
        help="Resolution scale as three integers",
    )
    parser.add_argument("--y_slice", type=int, default=5, help="Y slice value")
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Training data ratio"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated)",
    )

    args = parser.parse_args()

    # Compute statistics
    stats = compute_global_stats(
        data_dir=args.data_dir,
        field_name=args.field_name,
        resolution_scale=tuple(args.resolution_scale),
        y_slice=args.y_slice,
        train_ratio=args.train_ratio,
    )

    # Generate output filename if not provided
    if args.output is None:
        scale_str = f"{args.resolution_scale[0]}-{args.resolution_scale[1]}-{args.resolution_scale[2]}"
        args.output = (
            f"norm_stats_{args.field_name}_scale{scale_str}_yslice{args.y_slice}.json"
        )

    # Save to JSON file
    output_path = os.path.join(args.data_dir, args.output)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to: {output_path}")


if __name__ == "__main__":
    main()
