import glob
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FastFlowSequence2DDataset(Dataset):
    """Fast dataset for preprocessed 2D flow field data."""

    def __init__(
        self,
        data_dir: str = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow",
        input_length: int = 5,
        max_k_steps: int = 1,
        field_names: list[str] = None,  # Multi-channel support
        resolution_scale: tuple = (2, 3, 1),
        y_slice: int = 192,
        file_pattern: str = "*.h5",  # Not used but kept for compatibility
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "train",
        norm_stats: dict = None,  # Normalization statistics
        enable_normalization: bool = True,  # Whether to apply normalization
        # Backward compatibility
        field_name: str = None,  # Deprecated, use field_names instead
    ):
        """
        Args:
            data_dir: Directory containing preprocessed HDF5 files
            input_length: Number of previous timesteps for prediction
            max_k_steps: Maximum number of future steps for k-step rollout training
            train_ratio: Ratio of data for training
            valid_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            split: Dataset split ('train', 'val', 'test')
            norm_stats: Dictionary with 'mean' and 'std' for normalization
            enable_normalization: Whether to apply normalization
        """
        assert split in ["train", "val", "test"]
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

        self.data_dir = data_dir
        self.input_length = input_length
        self.max_k_steps = max_k_steps
        self.split = split
        self.enable_normalization = enable_normalization

        # Initialize field_names default
        if field_names is None:
            field_names = ["u", "v", "w"]

        # Handle backward compatibility
        if field_name is not None and field_names == ["u", "v", "w"]:
            # If old field_name is provided and field_names is default, use single field
            self.field_names = (
                list(field_name)
                if hasattr(field_name, "__iter__") and not isinstance(field_name, str)
                else [field_name]
            )
        else:
            self.field_names = (
                list(field_names)
                if hasattr(field_names, "__iter__") and not isinstance(field_names, str)
                else [field_names]
            )

        self.num_channels = len(self.field_names)
        self.resolution_scale = resolution_scale
        self.y_slice = y_slice

        # Setup normalization
        self._setup_normalization(norm_stats)

        # Get all preprocessed files
        field_names_str = "-".join([str(name) for name in self.field_names])
        pattern = (
            f"{field_names_str}_scale{resolution_scale[0]}-"
            f"{resolution_scale[1]}-{resolution_scale[2]}_yslice{y_slice}_*.h5"
        )
        self.file_list = sorted(glob.glob(os.path.join(data_dir, pattern)))
        print(f"Looking for pattern: {pattern}")
        print(f"Found {len(self.file_list)} files")
        self.num_frames = len(self.file_list)
        # For k-step rollout, we need input_length frames + max_k_steps target frames
        self.num_samples = self.num_frames - self.input_length - self.max_k_steps + 1

        if self.num_samples <= 0:
            raise ValueError(f"Not enough frames. Need at least {input_length + max_k_steps}, got {self.num_frames}")

        # Split data indices
        train_samples = int(self.num_samples * train_ratio)
        valid_samples = int(self.num_samples * valid_ratio)

        if split == "train":
            self.indices = list(range(0, train_samples))
        elif split == "val":
            self.indices = list(range(train_samples, train_samples + valid_samples))
        else:  # test
            self.indices = list(range(train_samples + valid_samples, self.num_samples))

        print(f"FastFlowSequence2DDataset {split}: {len(self.indices)} samples from {self.num_frames} files")

        # Get data shape from first file
        if self.file_list:
            with h5py.File(self.file_list[0], "r") as f:
                self.data_shape = f["data"].shape
                # Try to read metadata
                try:
                    stored_field_names = list(f.attrs.get("field_names", self.field_names))
                    stored_num_channels = f.attrs.get("num_channels", self.num_channels)
                    print(f"Data shape per frame: {self.data_shape}")
                    print(f"Stored field names: {stored_field_names}")
                    print(f"Number of channels: {stored_num_channels}")
                except Exception:
                    print(f"Data shape per frame: {self.data_shape}")
                    print(f"Field names: {self.field_names}")
                    print(f"Number of channels: {self.num_channels}")

    def _setup_normalization(self, norm_stats):
        """Setup normalization parameters."""
        if not self.enable_normalization or norm_stats is None:
            self.mean = None
            self.std = None
            print(f"Normalization disabled for {self.split} split")
            return

        # Load from dict or file path
        if isinstance(norm_stats, str):
            # Load from JSON file
            norm_stats_path = norm_stats
            if not os.path.isabs(norm_stats_path):
                norm_stats_path = os.path.join(self.data_dir, norm_stats_path)

            try:
                with open(norm_stats_path) as f:
                    stats = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load normalization stats from {norm_stats_path}: {e}")
                print("Normalization will be disabled.")
                self.mean = None
                self.std = None
                return
        elif isinstance(norm_stats, dict):
            stats = norm_stats
        else:
            raise ValueError(f"norm_stats must be dict or file path, got {type(norm_stats)}")

        # Extract mean and std - support both global and per-channel normalization
        try:
            # Check if we have per-channel statistics
            if "per_channel_stats" in stats and len(stats["per_channel_stats"]) > 0:
                # Per-channel normalization
                per_channel_stats = stats["per_channel_stats"]
                self.mean = []
                self.std = []

                for field_name in self.field_names:
                    if field_name in per_channel_stats:
                        self.mean.append(float(per_channel_stats[field_name]["mean"]))
                        self.std.append(float(per_channel_stats[field_name]["std"]))
                    else:
                        # Fallback to global stats if channel not found
                        print(f"Warning: No stats found for channel {field_name}, using global stats")
                        self.mean.append(float(stats["mean"]))
                        self.std.append(float(stats["std"]))

                # Convert to tensors for efficient computation
                import torch

                self.mean = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)  # (C, 1, 1)
                self.std = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)  # (C, 1, 1)
                self.per_channel_norm = True

                print(f"Per-channel normalization enabled for {self.split} split:")
                for i, field_name in enumerate(self.field_names):
                    if i < len(self.mean):
                        print(f"  {field_name}: mean={self.mean[i, 0, 0]:.6f}, std={self.std[i, 0, 0]:.6f}")

            else:
                # Global normalization (backward compatibility)
                self.mean = float(stats["mean"])
                self.std = float(stats["std"])
                self.per_channel_norm = False
                print(f"Global normalization enabled for {self.split} split: mean={self.mean:.6f}, std={self.std:.6f}")

            # Validate std is not zero
            if self.per_channel_norm:
                for i in range(len(self.std)):
                    if abs(self.std[i, 0, 0]) < 1e-8:
                        print(f"Warning: Standard deviation for channel {self.field_names[i]} is very small")
            else:
                if abs(self.std) < 1e-8:
                    print("Warning: Standard deviation is very small, normalization might be unstable")

        except (KeyError, TypeError, ValueError) as e:
            print(f"Warning: Invalid normalization stats format: {e}")
            print("Normalization will be disabled.")
            self.mean = None
            self.std = None

    def normalize(self, data):
        """Apply normalization to data."""
        if self.mean is None or self.std is None:
            return data

        if hasattr(self, "per_channel_norm") and self.per_channel_norm:
            # Per-channel normalization
            # data shape: (..., C, H, W) where C is channels
            # mean/std shape: (C, 1, 1)

            # Ensure tensors are on the same device
            if hasattr(data, "device"):
                mean = self.mean.to(data.device)
                std = self.std.to(data.device)
            else:
                mean = self.mean
                std = self.std

            return (data - mean) / std
        else:
            # Global normalization (backward compatibility)
            return (data - self.mean) / self.std

    def denormalize(self, data):
        """Apply denormalization to data (for inference)."""
        if self.mean is None or self.std is None:
            return data

        if hasattr(self, "per_channel_norm") and self.per_channel_norm:
            # Per-channel denormalization
            # data shape: (..., C, H, W) where C is channels
            # mean/std shape: (C, 1, 1)

            # Ensure tensors are on the same device
            if hasattr(data, "device"):
                mean = self.mean.to(data.device)
                std = self.std.to(data.device)
            else:
                mean = self.mean
                std = self.std

            return data * std + mean
        else:
            # Global denormalization (backward compatibility)
            return data * self.std + self.mean

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - "description": sample description
                - "data": input sequence data
                - "label": target sequence (k frames)
        """
        base_idx = self.indices[idx]
        description = (
            f"dataset: FastFlowSequence2DDataset, idx: {idx}, "
            f"fields: {self.field_names}, max_k_steps: {self.max_k_steps}"
        )

        # Load input sequence + k target frames
        frames = []
        for i in range(self.input_length + self.max_k_steps):
            fpath = self.file_list[base_idx + i]
            with h5py.File(fpath, "r") as f:
                data_multi_channel = f["data"][()]  # Already preprocessed multi-channel 2D data (C, H, W)
                frames.append(data_multi_channel)

        # Convert to tensors
        input_seq = np.stack(frames[: self.input_length], axis=0)  # (input_length, C, H, W)
        target_seq = np.stack(frames[self.input_length :], axis=0)  # (max_k_steps, C, H, W)

        # Convert to tensors
        input_seq = torch.from_numpy(input_seq).float()  # (input_length, C, H, W)
        target_seq = torch.from_numpy(target_seq).float()  # (max_k_steps, C, H, W)

        # Apply normalization
        input_seq = self.normalize(input_seq)
        # Note: target_seq is NOT normalized here because:
        # 1. Loss computation should be in normalized space
        # 2. Model outputs normalized predictions
        # 3. Denormalization happens during inference/evaluation
        target_seq = self.normalize(target_seq)

        # Add batch dimension like original dataset
        input_seq = input_seq.unsqueeze(0)  # (1, input_length, C, H, W)
        target_seq = target_seq.unsqueeze(0)  # (1, max_k_steps, C, H, W)

        # Structure like original dataset
        data = {"input_seq": input_seq}
        label = target_seq

        return {"description": np.array([description], dtype="<U100"), "data": data, "label": label}


if __name__ == "__main__":
    # Test the fast dataset with multi-channel data
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"

    print("Testing FastFlowSequence2DDataset with multi-channel data...")
    print(f"Data directory: {data_dir}")

    try:
        dataset = FastFlowSequence2DDataset(
            data_dir=data_dir, split="train", max_k_steps=4, field_names=["u", "v", "w"]
        )
        print(f"Dataset length: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Input sequence shape: {sample['data']['input_seq'].shape}")
            print(f"Target sequence shape: {sample['label'].shape}")
            print(f"Expected input shape: (1, input_length={dataset.input_length}, C={dataset.num_channels}, H, W)")
            print(f"Expected target shape: (1, max_k_steps={dataset.max_k_steps}, C={dataset.num_channels}, H, W)")
            print("✓ Multi-channel dataset test passed!")
        else:
            print("Warning: Dataset is empty - no preprocessed files found")
            print("Run the preprocessing script first to generate data")
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback

        traceback.print_exc()
