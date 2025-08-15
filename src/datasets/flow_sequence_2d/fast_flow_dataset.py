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
        field_name: str = "u",
        resolution_scale: tuple = (2, 3, 1),
        y_slice: int = 192,
        file_pattern: str = "*.h5",  # Not used but kept for compatibility
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "train",
        norm_stats: dict = None,  # Normalization statistics
        enable_normalization: bool = True,  # Whether to apply normalization
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

        # Metadata from preprocessing
        self.field_name = field_name
        self.resolution_scale = resolution_scale
        self.y_slice = y_slice

        # Setup normalization
        self._setup_normalization(norm_stats)

        # Get all preprocessed files
        pattern = (
            f"{field_name}_scale{resolution_scale[0]}-{resolution_scale[1]}-{resolution_scale[2]}_yslice{y_slice}_*.h5"
        )
        self.file_list = sorted(glob.glob(os.path.join(data_dir, pattern)))
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
                print(f"Data shape per frame: {self.data_shape}")

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

        # Extract mean and std
        try:
            self.mean = float(stats["mean"])
            self.std = float(stats["std"])
            print(f"Normalization enabled for {self.split} split: mean={self.mean:.6f}, std={self.std:.6f}")

            # Validate std is not zero
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
        return (data - self.mean) / self.std

    def denormalize(self, data):
        """Apply denormalization to data (for inference)."""
        if self.mean is None or self.std is None:
            return data
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
            f"dataset: FastFlowSequence2DDataset, idx: {idx}, field: {self.field_name}, max_k_steps: {self.max_k_steps}"
        )

        # Load input sequence + k target frames
        frames = []
        for i in range(self.input_length + self.max_k_steps):
            fpath = self.file_list[base_idx + i]
            with h5py.File(fpath, "r") as f:
                data_2d = f["data"][()]  # Already preprocessed 2D data
                frames.append(data_2d)

        # Convert to tensors
        input_seq = np.stack(frames[: self.input_length], axis=0)  # (input_length, H, W)
        target_seq = np.stack(frames[self.input_length :], axis=0)  # (max_k_steps, H, W)

        # Add channel dimension
        input_seq = input_seq[:, None, :, :]  # (input_length, 1, H, W)
        target_seq = target_seq[:, None, :, :]  # (max_k_steps, 1, H, W)

        # Convert to tensors
        input_seq = torch.from_numpy(input_seq).float()  # (input_length, 1, H, W)
        target_seq = torch.from_numpy(target_seq).float()  # (max_k_steps, 1, H, W)

        # Apply normalization
        input_seq = self.normalize(input_seq)
        # Note: target_seq is NOT normalized here because:
        # 1. Loss computation should be in normalized space
        # 2. Model outputs normalized predictions
        # 3. Denormalization happens during inference/evaluation
        target_seq = self.normalize(target_seq)

        # Add batch dimension like original dataset
        input_seq = input_seq.unsqueeze(0)  # (1, input_length, 1, H, W)
        target_seq = target_seq.unsqueeze(0)  # (1, max_k_steps, 1, H, W)

        # Structure like original dataset
        data = {"input_seq": input_seq}
        label = target_seq

        return {"description": np.array([description], dtype="<U100"), "data": data, "label": label}


if __name__ == "__main__":
    # Test the fast dataset
    dataset = FastFlowSequence2DDataset(split="train", max_k_steps=4)
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input sequence shape: {sample['data']['input_seq'].shape}")
        print(f"Target sequence shape: {sample['label'].shape}")
        print(f"Expected input shape: (1, input_length={dataset.input_length}, 1, H, W)")
        print(f"Expected target shape: (1, max_k_steps={dataset.max_k_steps}, 1, H, W)")
