#!/usr/bin/env python3
"""Test script to examine the structure of the original HDF5 files."""

import sys

import h5py
import numpy as np


def examine_h5_file(filepath):
    """Examine the structure of an HDF5 file."""
    print(f"Examining file: {filepath}")
    print("=" * 60)

    try:
        with h5py.File(filepath, "r") as f:
            print("File structure:")

            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name}")
                    print(f"    Shape: {obj.shape}")
                    print(f"    Dtype: {obj.dtype}")
                    print(
                        f"    Size: {obj.size * obj.dtype.itemsize / 1024 / 1024:.1f} MB"
                    )
                elif isinstance(obj, h5py.Group):
                    print(f"  Group: {name}")

            f.visititems(print_structure)

            # Check if it has the expected structure
            if "data" in f:
                data_group = f["data"]
                print(f"\nData group contents: {list(data_group.keys())}")

                # Check common field names
                for field in ["u", "v", "w", "p"]:
                    if field in data_group:
                        field_data = data_group[field]
                        print(
                            f"  Field '{field}': shape {field_data.shape}, dtype {field_data.dtype}"
                        )

                        # Show data range for first field
                        if field == "u":
                            sample_data = field_data[
                                ::10, ::10, ::10
                            ]  # Sample for speed
                            print(
                                f"    Data range (sampled): {sample_data.min():.6f} to {sample_data.max():.6f}"
                            )

    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/home/sh/CB/RE550_test/t00001.h5"

    examine_h5_file(filepath)

    # Test the preprocessing parameters
    print("\nTesting preprocessing parameters:")
    print("Field: u")
    print("Resolution scale: (1, 4, 4)")
    print("Y-slice: middle (will be determined from data)")

    try:
        with h5py.File(filepath, "r") as f:
            if "data" in f and "u" in f["data"]:
                original_data = f["data"]["u"]
                original_shape = original_data.shape
                print(f"Original shape: {original_shape}")

                # Simulate the preprocessing
                y_slice = original_shape[2] // 2  # Middle y-slice
                print(f"Middle y-slice: {y_slice}")

                # Calculate final shape after downsampling and slicing
                downsampled_shape = (
                    original_shape[0] // 1,  # z: no downsampling
                    original_shape[1] // 4,  # y: 4x downsampling
                    original_shape[2] // 4,  # x: 4x downsampling
                )
                print(f"After downsampling: {downsampled_shape}")

                final_2d_shape = (
                    downsampled_shape[0],
                    downsampled_shape[2],
                )  # (z, x) after y-slice
                print(f"Final 2D shape: {final_2d_shape}")

                # Estimate file size reduction
                original_size = np.prod(original_shape) * 4  # float32 = 4 bytes
                final_size = np.prod(final_2d_shape) * 4
                reduction_factor = final_size / original_size
                print(
                    f"Size reduction: {reduction_factor:.6f}x ({100 * (1 - reduction_factor):.1f}% smaller)"
                )
                print(
                    f"Estimated output size per file: {final_size / 1024 / 1024:.1f} MB"
                )

    except Exception as e:
        print(f"Error in preprocessing test: {e}")
