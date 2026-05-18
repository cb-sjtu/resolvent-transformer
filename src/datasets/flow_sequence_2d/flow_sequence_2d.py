import glob
import os

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class FlowSequence2DDataset(Dataset):
    """Dataset for 2D temporal sequence prediction of flow fields from HDF5 files.

    Enhanced to support multi-channel (u,v,w) turbulence data.
    """

    def __init__(
        self,
        data_dir: str,
        input_length: int = 5,
        field_names: list[str] = None,  # Multi-channel support
        file_pattern: str = "*.h5",
        resolution_scale: tuple[int, int, int] = (1, 4, 4),  # (z, y, x) downsampling
        y_slice: int = None,  # Which y-slice to extract for 2D (if None, use middle)
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "train",
        # Backward compatibility
        field_name: str = None,  # Deprecated, use field_names instead
    ):
        """
        Args:
            data_dir: Directory containing HDF5 files
            input_length: Number of previous timesteps to use for prediction
            field_names: List of fields to predict (['u', 'v', 'w'] for multi-channel)
            file_pattern: Pattern for HDF5 files
            resolution_scale: Downsampling factor for (z, y, x) dimensions
            y_slice: Which y-slice to extract (if None, uses middle slice)
            train_ratio: Ratio of data for training
            valid_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            split: Dataset split ('train', 'val', 'test')
            field_name: Deprecated, use field_names instead
        """
        assert split in ["train", "val", "test"]
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

        if field_names is None:
            field_names = ["u", "v", "w"]

        self.data_dir = data_dir
        self.input_length = input_length

        # Handle backward compatibility
        if field_name is not None and field_names == ["u", "v", "w"]:
            # If old field_name is provided and field_names is default, use single field
            self.field_names = [field_name]
        else:
            self.field_names = (
                field_names if isinstance(field_names, list) else [field_names]
            )

        self.num_channels = len(self.field_names)
        self.resolution_scale = resolution_scale
        self.y_slice = y_slice
        self.split = split

        # Get all files
        self.file_list = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        self.num_frames = len(self.file_list)
        self.num_samples = self.num_frames - self.input_length

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough frames. Need at least {input_length + 1}, got {self.num_frames}"
            )

        # Split data indices
        train_samples = int(self.num_samples * train_ratio)
        valid_samples = int(self.num_samples * valid_ratio)

        if split == "train":
            self.indices = list(range(0, train_samples))
        elif split == "val":
            self.indices = list(range(train_samples, train_samples + valid_samples))
        else:  # test
            self.indices = list(range(train_samples + valid_samples, self.num_samples))

        print(
            f"Created {split} dataset with {len(self.indices)} samples from {self.num_frames} files"
        )

        # Get data shape from first file
        self._get_data_shape()

    def _get_data_shape(self):
        """Get the shape of 2D data after processing."""
        with h5py.File(self.file_list[0], "r") as f:
            # Use first field to determine spatial dimensions
            first_field = self.field_names[0]
            data = f["data"][first_field][()]

            # Apply downsampling
            data = data[
                :: self.resolution_scale[0],
                :: self.resolution_scale[1],
                :: self.resolution_scale[2],
            ]

            # Extract 2D slice (take slice in y-direction, so result is (z, x))
            if self.y_slice is None:
                self.y_slice = (
                    data.shape[1] // 2
                )  # Use middle slice in y-direction (assuming shape is (z,y,x))

            data_2d = data[:, self.y_slice, :]  # Shape: (z, x)
            self.data_shape = data_2d.shape

            print(f"Fields: {self.field_names} ({self.num_channels} channels)")
            print(f"Original 3D shape: {f['data'][first_field].shape}")
            print(f"After downsampling: {data.shape}")
            print(f"2D slice shape: {self.data_shape}, using y_slice={self.y_slice}")

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
        description = f"dataset: {self.__class__.__name__}, idx: {idx}, fields: {self.field_names}"

        # Load input sequence for all channels
        frames = []
        for i in range(self.input_length + 1):
            fpath = self.file_list[base_idx + i]

            # Load all channels for this timestep
            channel_data = []
            with h5py.File(fpath, "r") as f:
                for field_name in self.field_names:
                    data = f["data"][field_name][()]

                    # Apply downsampling
                    data = data[
                        :: self.resolution_scale[0],
                        :: self.resolution_scale[1],
                        :: self.resolution_scale[2],
                    ]

                    # Extract 2D slice (y-slice, result is (z, x))
                    data_2d = data[:, self.y_slice, :]
                    channel_data.append(data_2d)

            # Stack channels: (C, H, W)
            multi_channel_frame = np.stack(channel_data, axis=0)
            frames.append(multi_channel_frame)

        # Convert to tensors
        input_seq = np.stack(frames[:-1], axis=0)  # (input_length, C, H, W)
        target = frames[-1]  # (C, H, W)

        # Add leading batch dimension of 1 (like TheWell dataset)
        input_seq = (
            torch.from_numpy(input_seq).float().unsqueeze(0)
        )  # (1, input_length, C, H, W)
        target = torch.from_numpy(target).float().unsqueeze(0)  # (1, C, H, W)

        # Structure like TheWell dataset
        data = {"input_seq": input_seq}  # Could add more fields here if needed
        label = target

        return {
            "description": np.array([description], dtype="<U100"),
            "data": data,
            "label": label,
        }


class TurbulenceDataset2D(Dataset):
    """Alternative dataset class with multi-channel support."""

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

        # Support both single field and multi-field configurations
        if "field_names" in cfg:
            self.field_names = cfg["field_names"]  # Multi-channel: ["u", "v", "w"]
        else:
            self.field_names = [cfg.get("field_name", "u")]  # Single channel fallback

        self.num_channels = len(self.field_names)

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
                first_field = self.field_names[0]
                y_dim = f["data"][first_field].shape[1]
                self.y_slice = y_dim // 2  # Use middle slice in y-direction

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        frames = []

        for i in range(self.input_length + 1):
            fpath = self.file_list[base_idx + i]

            # Load all channels for this timestep
            channel_data = []
            with h5py.File(fpath, "r") as f:
                for field_name in self.field_names:
                    field_data = f["data"][field_name][()]

                    # Apply downsampling
                    field_data = field_data[
                        :: self.scale[0], :: self.scale[1], :: self.scale[2]
                    ]

                    # Extract 2D slice (y-slice, result is (z, x))
                    field_2d = field_data[:, self.y_slice, :]
                    channel_data.append(field_2d)

            # Stack channels: (C, H, W)
            multi_channel_frame = np.stack(channel_data, axis=0)
            frames.append(multi_channel_frame)

        input_seq = np.stack(frames[:-1], axis=0)  # (input_length, C, H, W)
        target = frames[-1]  # (C, H, W)

        return torch.from_numpy(input_seq).float(), torch.from_numpy(target).float()


if __name__ == "__main__":
    print("Testing Multi-Channel FlowSequence2DDataset...")

    # Test multi-channel dataset (u, v, w)
    dataset_multi = FlowSequence2DDataset(
        data_dir="/media/sh/Seagate Basic/RE550/",
        input_length=5,
        field_names=["u", "v", "w"],  # Multi-channel
        resolution_scale=(1, 4, 4),  # No z downsampling, 4x downsample in y,x
        split="train",
    )

    print(f"Multi-channel dataset length: {len(dataset_multi)}")
    if len(dataset_multi) > 0:
        sample = dataset_multi[0]
        input_seq = sample["data"]["input_seq"]
        target = sample["label"]
        print(f"Multi-channel input sequence shape: {input_seq.shape}")
        print("Expected: (1, input_length=3, C=3, H, W)")
        print(f"Multi-channel target shape: {target.shape}")
        print("Expected: (1, C=3, H, W)")

    print("\nTesting single-channel backward compatibility...")
    # Test backward compatibility with single channel
    dataset_single = FlowSequence2DDataset(
        data_dir="/media/sh/Seagate Basic/RE550/",
        input_length=3,
        field_name="u",  # Use deprecated field_name for backward compatibility
        resolution_scale=(1, 4, 4),
        split="train",
    )

    print(f"Single-channel dataset length: {len(dataset_single)}")
    if len(dataset_single) > 0:
        sample = dataset_single[0]
        input_seq = sample["data"]["input_seq"]
        target = sample["label"]
        print(f"Single-channel input sequence shape: {input_seq.shape}")
        print("Expected: (1, input_length=3, C=1, H, W)")
        print(f"Single-channel target shape: {target.shape}")
        print("Expected: (1, C=1, H, W)")

    print("\n✓ All dataset tests passed!")
