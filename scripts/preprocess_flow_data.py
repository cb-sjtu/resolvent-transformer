#!/usr/bin/env python3
"""
Data preprocessing script to extract smaller files from large HDF5 flow field data.
This extracts specific field, resolution scale, and y-slice to create much smaller files for faster training.
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm


def extract_flow_data(
    input_dir: str,
    output_dir: str,
    field_name: str = "u",
    resolution_scale: tuple[int, int, int] = (1, 4, 4),
    y_slice: int = None,
    file_pattern: str = "*.h5",
    overwrite: bool = False,
    compress: bool = True,
):
    """
    Extract and downsample flow field data from large HDF5 files.

    Args:
        input_dir: Directory containing original large HDF5 files
        output_dir: Directory to save extracted smaller files
        field_name: Field to extract ('u', 'v', 'w', 'p')
        resolution_scale: Downsampling factors for (z, y, x) dimensions
        y_slice: Which y-slice to extract (None for middle slice)
        file_pattern: Pattern to match input files
        overwrite: Whether to overwrite existing output files
        compress: Whether to use HDF5 compression
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of input files
    input_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    if not input_files:
        raise ValueError(f"No files found matching pattern {file_pattern} in {input_dir}")

    print(f"Found {len(input_files)} files to process")
    print(f"Field: {field_name}")
    print(f"Resolution scale: {resolution_scale}")
    print(f"Y-slice: {y_slice} (None means middle slice)")
    print(f"Output directory: {output_dir}")
    print(f"Compression: {compress}")

    # Determine y_slice from first file if not specified
    if y_slice is None:
        with h5py.File(input_files[0], "r") as f:
            original_shape = f["data"][field_name].shape
            # Calculate y_slice AFTER downsampling
            downsampled_y_size = original_shape[2] // resolution_scale[2]
            y_slice = downsampled_y_size // 2  # Middle slice in downsampled y-direction
            print(f"Original shape: {original_shape}")
            print(
                f"After downsampling: ({original_shape[0] // resolution_scale[0]},"
                f" {original_shape[1] // resolution_scale[1]}, {downsampled_y_size})"
            )
            print(f"Using middle y-slice: {y_slice} (out of {downsampled_y_size})")

    # Process each file
    total_saved_size = 0
    total_original_size = 0

    for input_file in tqdm(input_files, desc="Processing files"):
        # Generate output filename
        input_filename = os.path.basename(input_file)
        output_filename = f"{field_name}_scale{resolution_scale[0]}-"
        f"{resolution_scale[1]}-{resolution_scale[2]}_"
        f"yslice{y_slice}_{input_filename}"
        output_path = os.path.join(output_dir, output_filename)

        # Skip if file exists and not overwriting
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {output_filename} (already exists)")
            continue

        try:
            # Load and process data
            with h5py.File(input_file, "r") as f_in:
                # Get original data
                original_data = f_in["data"][field_name][()]
                original_shape = original_data.shape

                # Apply downsampling
                downsampled_data = original_data[:: resolution_scale[0], :: resolution_scale[1], :: resolution_scale[2]]

                # Extract 2D slice (y-slice, result is (z, x))
                data_2d = downsampled_data[:, :, y_slice]

                # Get file sizes for comparison
                original_size = os.path.getsize(input_file)
                total_original_size += original_size

            # Save extracted data
            compression_opts = {"compression": "gzip", "compression_opts": 9} if compress else {}

            with h5py.File(output_path, "w") as f_out:
                # Save the 2D extracted data
                f_out.create_dataset("data", data=data_2d, dtype=np.float32, **compression_opts)

                # Save metadata
                f_out.attrs["original_field"] = field_name
                f_out.attrs["original_shape"] = original_shape
                f_out.attrs["resolution_scale"] = resolution_scale
                f_out.attrs["y_slice"] = y_slice
                f_out.attrs["extracted_shape"] = data_2d.shape
                f_out.attrs["source_file"] = input_filename

            # Track saved size
            saved_size = os.path.getsize(output_path)
            total_saved_size += saved_size

            # Print progress info
            reduction_ratio = saved_size / original_size if original_size > 0 else 0
            print(
                f"  {input_filename}: {original_shape} -> {data_2d.shape}, "
                f"{original_size / 1024 / 1024:.1f}MB -> {saved_size / 1024 / 1024:.1f}MB "
                f"({reduction_ratio:.3f}x)"
            )

        except Exception as e:
            print(f"ERROR processing {input_file}: {e}")
            continue

    # Print summary
    print("\nProcessing complete!")
    print(f"Total files processed: {len(input_files)}")
    print(f"Original total size: {total_original_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Extracted total size: {total_saved_size / 1024 / 1024 / 1024:.2f} GB")

    if total_original_size > 0:
        total_reduction = total_saved_size / total_original_size
        print(f"Total size reduction: {total_reduction:.4f}x ({100 * (1 - total_reduction):.1f}% smaller)")


def create_fast_dataset_class(output_dir: str, field_name: str, resolution_scale: tuple, y_slice: int) -> str:
    """Create a fast dataset class for the preprocessed data."""

    dataset_code = f'''import glob
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FastFlowSequence2DDataset(Dataset):
    """Fast dataset for preprocessed 2D flow field data."""

    def __init__(
        self,
        data_dir: str = "{output_dir}",
        input_length: int = 5,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "train",
    ):
        """
        Args:
            data_dir: Directory containing preprocessed HDF5 files
            input_length: Number of previous timesteps for prediction
            train_ratio: Ratio of data for training
            valid_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            split: Dataset split ('train', 'val', 'test')
        """
        assert split in ["train", "val", "test"]
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

        self.data_dir = data_dir
        self.input_length = input_length
        self.split = split

        # Metadata from preprocessing
        self.field_name = "{field_name}"
        self.resolution_scale = {resolution_scale}
        self.y_slice = {y_slice}

        # Get all preprocessed files
        pattern = f"{field_name}_scale{resolution_scale[0]}-"
        f"{resolution_scale[1]}-{resolution_scale[2]}_yslice{y_slice}_*.h5"
        self.file_list = sorted(glob.glob(os.path.join(data_dir, pattern)))
        self.num_frames = len(self.file_list)
        self.num_samples = self.num_frames - self.input_length

        if self.num_samples <= 0:
            raise ValueError(f"Not enough frames. Need at least {{input_length + 1}}, got {{self.num_frames}}")

        # Split data indices
        train_samples = int(self.num_samples * train_ratio)
        valid_samples = int(self.num_samples * valid_ratio)

        if split == "train":
            self.indices = list(range(0, train_samples))
        elif split == "val":
            self.indices = list(range(train_samples, train_samples + valid_samples))
        else:  # test
            self.indices = list(range(train_samples + valid_samples, self.num_samples))

        print(f"FastFlowSequence2DDataset {{split}}: {{len(self.indices)}} samples from {{self.num_frames}} files")

        # Get data shape from first file
        if self.file_list:
            with h5py.File(self.file_list[0], "r") as f:
                self.data_shape = f["data"].shape
                print(f"Data shape per frame: {{self.data_shape}}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - "description": sample description
                - "data": input sequence data
                - "label": target frame
        """
        base_idx = self.indices[idx]
        description = f"dataset: FastFlowSequence2DDataset, idx: {{idx}}, field: {{self.field_name}}"

        # Load input sequence - much faster now!
        frames = []
        for i in range(self.input_length + 1):
            fpath = self.file_list[base_idx + i]
            with h5py.File(fpath, "r") as f:
                data_2d = f["data"][()]  # Already preprocessed 2D data
                frames.append(data_2d)

        # Convert to tensors
        input_seq = np.stack(frames[:-1], axis=0)  # (input_length, H, W)
        target = frames[-1]  # (H, W)

        # Add channel dimension
        input_seq = input_seq[:, None, :, :]  # (input_length, 1, H, W)
        target = target[None, :, :]  # (1, H, W)

        # Add batch dimension like original dataset
        input_seq = torch.from_numpy(input_seq).float().unsqueeze(0)  # (1, input_length, 1, H, W)
        target = torch.from_numpy(target).float().unsqueeze(0)  # (1, 1, H, W)

        # Structure like original dataset
        data = {{"input_seq": input_seq}}
        label = target

        return {{
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": data,
            "label": label
        }}


if __name__ == "__main__":
    # Test the fast dataset
    dataset = FastFlowSequence2DDataset(split="train")
    print(f"Dataset length: {{len(dataset)}}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input sequence shape: {{sample['data']['input_seq'].shape}}")
        print(f"Target shape: {{sample['label'].shape}}")
'''

    return dataset_code


def main():
    parser = argparse.ArgumentParser(description="Preprocess flow field data for faster training")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with large HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for extracted files")
    parser.add_argument(
        "--field", type=str, default="u", choices=["u", "v", "w", "p"], help="Field to extract (default: u)"
    )
    parser.add_argument(
        "--resolution_scale",
        type=int,
        nargs=3,
        default=[1, 4, 4],
        help="Downsampling factors for z, y, x dimensions (default: 1 4 4)",
    )
    parser.add_argument("--y_slice", type=int, default=None, help="Y-slice index to extract (default: middle slice)")
    parser.add_argument("--pattern", type=str, default="*.h5", help="File pattern to match (default: *.h5)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--no_compress", action="store_true", help="Disable HDF5 compression")
    parser.add_argument("--create_dataset", action="store_true", help="Create fast dataset class file")

    args = parser.parse_args()

    try:
        # Extract data
        extract_flow_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            field_name=args.field,
            resolution_scale=tuple(args.resolution_scale),
            y_slice=args.y_slice,
            file_pattern=args.pattern,
            overwrite=args.overwrite,
            compress=not args.no_compress,
        )

        # Create fast dataset class if requested
        if args.create_dataset:
            dataset_code = create_fast_dataset_class(
                args.output_dir, args.field, tuple(args.resolution_scale), args.y_slice or "middle"
            )

            dataset_file = os.path.join(args.output_dir, "fast_flow_dataset.py")
            with open(dataset_file, "w") as f:
                f.write(dataset_code)
            print(f"Created fast dataset class: {dataset_file}")

        print("\\nPreprocessing completed successfully!")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
