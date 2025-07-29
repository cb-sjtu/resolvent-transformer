import glob
import os

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class FlowSequence2DDataset(Dataset):
    """Dataset for 2D temporal sequence prediction of flow fields from HDF5 files."""

    def __init__(
        self,
        data_dir: str,
        input_length: int = 5,
        field_name: str = "u",
        file_pattern: str = "*.h5",
        resolution_scale: tuple[int, int, int] = (1, 4, 4),  # (z, y, x) downsampling
        y_slice: int = None,  # Which y-slice to extract for 2D (if None, use middle)
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "train",
    ):
        """
        Args:
            data_dir: Directory containing HDF5 files
            input_length: Number of previous timesteps to use for prediction
            field_name: Which field to predict ('u', 'v', 'w', or 'p')
            file_pattern: Pattern for HDF5 files
            resolution_scale: Downsampling factor for (z, y, x) dimensions
            y_slice: Which y-slice to extract (if None, uses middle slice)
            train_ratio: Ratio of data for training
            valid_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            split: Dataset split ('train', 'val', 'test')
        """
        assert split in ["train", "val", "test"]
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

        self.data_dir = data_dir
        self.input_length = input_length
        self.field_name = field_name
        self.resolution_scale = resolution_scale
        self.y_slice = y_slice
        self.split = split

        # Get all files
        self.file_list = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        self.num_frames = len(self.file_list)
        self.num_samples = self.num_frames - self.input_length

        if self.num_samples <= 0:
            raise ValueError(f"Not enough frames. Need at least {input_length + 1}, got {self.num_frames}")

        # Split data indices
        train_samples = int(self.num_samples * train_ratio)
        valid_samples = int(self.num_samples * valid_ratio)

        if split == "train":
            self.indices = list(range(0, train_samples))
        elif split == "val":
            self.indices = list(range(train_samples, train_samples + valid_samples))
        else:  # test
            self.indices = list(range(train_samples + valid_samples, self.num_samples))

        print(f"Created {split} dataset with {len(self.indices)} samples from {self.num_frames} files")

        # Get data shape from first file
        self._get_data_shape()

    def _get_data_shape(self):
        """Get the shape of 2D data after processing."""
        with h5py.File(self.file_list[0], "r") as f:
            data = f["data"][self.field_name][()]

            # Apply downsampling
            data = data[:: self.resolution_scale[0], :: self.resolution_scale[1], :: self.resolution_scale[2]]

            # Extract 2D slice (take slice in y-direction, so result is (z, x))
            if self.y_slice is None:
                self.y_slice = data.shape[1] // 2  # Use middle slice in y-direction

            data_2d = data[:, self.y_slice, :]  # Shape: (z, x)
            self.data_shape = data_2d.shape

            print(f"Original 3D shape: {f['data'][self.field_name].shape}")
            print(f"After downsampling: {data.shape}")
            print(f"2D slice shape: {self.data_shape}, using y_slice={self.y_slice}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_sequence: (input_length, 1, H, W) - input sequence
            target: (1, H, W) - target frame
        """
        base_idx = self.indices[idx]

        # Load input sequence
        frames = []
        for i in range(self.input_length + 1):
            fpath = self.file_list[base_idx + i]

            with h5py.File(fpath, "r") as f:
                data = f["data"][self.field_name][()]

                # Apply downsampling
                data = data[:: self.resolution_scale[0], :: self.resolution_scale[1], :: self.resolution_scale[2]]

                # Extract 2D slice (y-slice, result is (z, x))
                data_2d = data[:, self.y_slice, :]
                frames.append(data_2d)

        # Convert to tensors
        input_seq = np.stack(frames[:-1], axis=0)  # (input_length, H, W)
        target = frames[-1]  # (H, W)

        # Add channel dimension
        input_seq = input_seq[:, None, :, :]  # (input_length, 1, H, W)
        target = target[None, :, :]  # (1, H, W)

        return torch.from_numpy(input_seq).float(), torch.from_numpy(target).float()


class TurbulenceDataset2D(Dataset):
    """Alternative dataset class matching your reference structure."""

    def __init__(self, config_path, split="train"):
        assert split in ["train", "val", "test"]

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.input_length = cfg["input_length"]
        self.scale = cfg["resolution_scale"]
        self.data_dir = cfg["data_dir"]
        self.train_ratio = cfg.get("train_ratio", 0.7)
        self.valid_ratio = cfg.get("valid_ratio", 0.15)
        self.test_ratio = cfg.get("test_ratio", 0.15)
        self.y_slice = cfg.get("y_slice", None)  # Which y-slice to extract
        self.field_name = cfg.get("field_name", "u")  # Which field to predict

        self.file_list = sorted(glob.glob(os.path.join(self.data_dir, "*.h5")))
        self.num_frames = len(self.file_list)
        self.num_samples = self.num_frames - self.input_length

        # Split indices based on ratios
        train_samples = int(self.num_samples * self.train_ratio)
        valid_samples = int(self.num_samples * self.valid_ratio)

        if split == "train":
            self.indices = list(range(0, train_samples))
        elif split == "val":
            self.indices = list(range(train_samples, train_samples + valid_samples))
        else:  # test
            self.indices = list(range(train_samples + valid_samples, self.num_samples))

        # Determine y_slice if not specified
        if self.y_slice is None:
            with h5py.File(self.file_list[0], "r") as f:
                y_dim = f["data"][self.field_name].shape[1]
                self.y_slice = y_dim // 2  # Use middle slice in y-direction

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        frames = []

        for i in range(self.input_length + 1):
            fpath = self.file_list[base_idx + i]
            with h5py.File(fpath, "r") as f:
                # Load only the specified field
                field_data = f["data"][self.field_name][()]

                # Apply downsampling
                field_data = field_data[:: self.scale[0], :: self.scale[1], :: self.scale[2]]

                # Extract 2D slice (y-slice, result is (z, x))
                field_2d = field_data[:, self.y_slice, :]

                # Add channel dimension
                tensor = field_2d[None, :, :]  # (1, H, W)
                frames.append(tensor)

        input_seq = np.stack(frames[:-1], axis=0)  # (input_length, 1, H, W)
        target = frames[-1]  # (1, H, W)

        return torch.from_numpy(input_seq).float(), torch.from_numpy(target).float()


if __name__ == "__main__":
    # Test the dataset
    dataset = FlowSequence2DDataset(
        data_dir="/media/sh/Seagate Basic/RE550/",
        input_length=5,
        field_name="u",
        resolution_scale=(1, 4, 4),  # No z downsampling, 4x downsample in y,x
        split="train",
    )

    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        input_seq, target = dataset[0]
        print(f"Input sequence shape: {input_seq.shape}")
        print(f"Target shape: {target.shape}")
