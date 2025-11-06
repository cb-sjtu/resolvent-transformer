"""
Dataset for single-plane (y_slice54) flow sequence prediction.
支持加载1个y平面（yslice54），包含uvw三个通道的数据。
这是从3plane版本简化而来，只处理单个y平面。
"""

import glob
import json
import os
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FlowSequence1PlaneDataset(Dataset):
    """Dataset for single-plane temporal sequence prediction of flow fields.

    Loads 1 y-plane (yslice54) with 3 channels (u,v,w), for a total of 3 channels.
    """

    def __init__(
        self,
        data_dir: str,
        input_length: int = 5,
        max_k_steps: int = 1,  # Number of future steps to load as ground truth
        field_names: list[str] = None,  # ["u", "v", "w"]
        file_pattern: str = "*u-v-w-p_scale4-6-1_yslice54*.h5",  # 匹配yslice54的文件
        resolution_scale: tuple[int, int, int] = (4, 6, 1),  # (z, y, x) downsampling
        y_slice: int = 54,  # y平面的索引
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "train",
        enable_normalization: bool = True,
        norm_stats=None,  # Normalization statistics (file path or dict)
        time_stride: int = 1,  # Time stride between frames (1=consecutive, 2=skip one frame)
        prediction_step_size: int = 1,  # Number of steps for each prediction (for discontinuity filtering)
    ):
        """
        Args:
            data_dir: Directory containing HDF5 files
            input_length: Number of previous timesteps to use for prediction
            max_k_steps: Number of future steps to load as ground truth (for evaluation)
            field_names: List of fields to predict (default: ['u', 'v', 'w'])
            file_pattern: Pattern for HDF5 files (should match yslice54 uvwp files)
            resolution_scale: Downsampling factor for (z, y, x) dimensions
            y_slice: The y-slice index to load (default: 54)
            train_ratio: Ratio of data for training
            valid_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            split: Dataset split ('train', 'val', 'test')
            enable_normalization: Whether to enable normalization
            norm_stats: Normalization statistics (file path or dict)
            time_stride: Time stride between frames (1=consecutive, 2=every other frame, etc.)
            prediction_step_size: Number of steps for each prediction (default: 1, used for discontinuity filtering)
        """
        assert split in ["train", "val", "test"]
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
        assert time_stride >= 1, "time_stride must be >= 1"

        if field_names is None:
            field_names = ["u", "v", "w"]

        self.data_dir = data_dir
        self.input_length = input_length
        self.max_k_steps = max_k_steps
        self.prediction_step_size = prediction_step_size
        self.field_names = field_names
        self.num_channels = len(self.field_names)  # 3 channels for u,v,w
        self.resolution_scale = resolution_scale
        self.y_slice = y_slice
        self.split = split
        self.enable_normalization = enable_normalization
        self.time_stride = time_stride

        # Get files matching the pattern
        files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))

        if not files:
            raise ValueError(f"No files found with pattern: {file_pattern} in {data_dir}")

        # Extract timesteps from filenames
        self.timesteps = []
        self.file_dict = {}  # Map timestep -> filename

        for f in files:
            # Extract timestep from filename like "..._t00123.h5"
            match = re.search(r"_t(\d+)\.h5$", f)
            if match:
                timestep = int(match.group(1))
                self.timesteps.append(timestep)
                self.file_dict[timestep] = f

        self.timesteps = sorted(self.timesteps)
        self.num_frames = len(self.timesteps)

        # Each sample needs input_length + max_k_steps frames with time_stride spacing
        # Total span: (input_length + max_k_steps - 1) * time_stride + 1
        # Example: input=5, target=1, stride=2 -> need frames at [0,2,4,6,8,10] -> span=11
        total_frames_needed = (self.input_length + self.prediction_step_size - 1) * self.time_stride + 1
        self.num_samples = self.num_frames - total_frames_needed + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough frames. Need at least {total_frames_needed} "
                f"(input_length={self.input_length}, max_k_steps={self.max_k_steps}, "
                f"time_stride={self.time_stride}), got {self.num_frames}"
            )

        print(f"Found {self.num_frames} timesteps for y_slice={y_slice}")
        print(f"Files: {len(files)}")
        print(f"Time stride: {self.time_stride} (frames spaced by {self.time_stride}t)")
        print(f"Total frames needed per sample: {total_frames_needed}")

        # Filter out sequences that span the discontinuity between timestep 1080 and 1081
        # For discontinuity filtering, we use prediction_step_size (typically 1) instead of max_k_steps
        # because the physical discontinuity matters for single-step predictions during training/inference,
        # not for the number of ground truth frames loaded for comparison during evaluation.
        # A sequence starting at timestep T will access frames from T to T
        # + (input_length + prediction_step_size - 1) * time_stride
        # We need to exclude sequences where this range crosses the discontinuity (1080 -> 1081)
        discontinuity_timestep = 1081

        valid_indices = []
        excluded_count = 0

        for i in range(self.num_samples):
            start_timestep = self.timesteps[i]
            # Calculate the last timestep index this sequence will access for prediction
            # A sequence starting at index i will access frames at:
            # i, i+stride, i+2*stride, ..., i+(input_length+prediction_step_size-1)*stride
            # Note: We use prediction_step_size here, NOT max_k_steps
            end_idx = i + (self.input_length + self.prediction_step_size - 1) * self.time_stride
            end_timestep = self.timesteps[end_idx]

            # Exclude if the sequence spans across the discontinuity
            # This happens when start <= 1080 and end >= 1081
            if start_timestep <= 1080 < end_timestep:
                excluded_count += 1
                print(
                    f"Excluding sequence starting at timestep {start_timestep}, "
                    f"ending at {end_timestep} (spans discontinuity at 1080->1081)"
                )
            else:
                valid_indices.append(i)

        print(f"Excluded {excluded_count} sequences due to discontinuity at timestep {discontinuity_timestep}")
        print(f"Valid samples: {len(valid_indices)} (originally {self.num_samples})")

        # Update num_samples to reflect filtered data
        self.num_samples = len(valid_indices)
        self.valid_base_indices = valid_indices

        # Split data indices using the filtered valid_base_indices
        train_samples = int(self.num_samples * train_ratio)
        valid_samples = int(self.num_samples * valid_ratio)

        if split == "train":
            self.indices = [self.valid_base_indices[i] for i in range(0, train_samples)]
        elif split == "val":
            self.indices = [self.valid_base_indices[i] for i in range(train_samples, train_samples + valid_samples)]
        else:  # test
            self.indices = [self.valid_base_indices[i] for i in range(train_samples + valid_samples, self.num_samples)]

        print(f"Created {split} dataset with {len(self.indices)} samples from {self.num_frames} files")

        # Get data shape from first file
        self._get_data_shape()

        # Setup normalization
        self._setup_normalization(norm_stats)

    def _get_data_shape(self):
        """Get the shape of 2D data."""
        first_file = self.file_dict[self.timesteps[0]]

        with h5py.File(first_file, "r") as f:
            # Data is stored as (C, H, W) where C can be 3 (u,v,w) or 4 (u,v,w,p)
            data_multi_channel = f["data"][()]  # Shape: (C, H, W)

            # Check that data shape is 3D and has at least the required number of channels
            if len(data_multi_channel.shape) != 3:
                raise ValueError(f"Expected 3D data shape (C, H, W), got {data_multi_channel.shape}")

            if data_multi_channel.shape[0] < len(self.field_names):
                raise ValueError(
                    f"Data file has {data_multi_channel.shape[0]} channels, "
                    f"but requested {len(self.field_names)} fields: {self.field_names}"
                )

            # Get 2D shape from the data (H, W)
            self.data_shape = data_multi_channel.shape[1:]  # (H, W)

            print(f"Fields: {self.field_names} ({self.num_channels} channels)")
            print(f"Data shape per file: {data_multi_channel.shape}")
            print(f"2D shape: {self.data_shape}")
            print(f"Y-slice: {self.y_slice}")

    def _setup_normalization(self, norm_stats):
        """Setup normalization parameters for single-plane 3-channel data."""
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

        # Extract per-channel statistics
        try:
            if "per_channel_stats" in stats and len(stats["per_channel_stats"]) > 0:
                # Per-channel normalization
                per_channel_stats = stats["per_channel_stats"]
                self.mean = []
                self.std = []

                # Build channel stats in order: [u, v, w]
                for ch_idx in range(self.num_channels):
                    channel_key = f"channel_{ch_idx:02d}"

                    if channel_key in per_channel_stats:
                        self.mean.append(float(per_channel_stats[channel_key]["mean"]))
                        self.std.append(float(per_channel_stats[channel_key]["std"]))
                    else:
                        # Fallback to global stats if channel not found
                        print(f"Warning: No stats found for {channel_key}, using global stats")
                        self.mean.append(float(stats["mean"]))
                        self.std.append(float(stats["std"]))

                # Convert to tensors for efficient computation
                self.mean = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)  # (num_channels, 1, 1)
                self.std = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)  # (num_channels, 1, 1)
                self.per_channel_norm = True

                print(f"{self.num_channels}-channel normalization enabled for {self.split} split:")
                for ch_idx in range(len(self.mean)):
                    print(
                        f"  channel_{ch_idx:02d} ({self.field_names[ch_idx]}): "
                        f"mean={self.mean[ch_idx, 0, 0]:.6f}, std={self.std[ch_idx, 0, 0]:.6f}"
                    )

            else:
                # Global normalization fallback
                self.mean = float(stats["mean"])
                self.std = float(stats["std"])
                self.per_channel_norm = False
                print(f"Global normalization enabled for {self.split} split: mean={self.mean:.6f}, std={self.std:.6f}")

            # Validate std is not zero
            if self.per_channel_norm:
                for i in range(len(self.std)):
                    if abs(self.std[i, 0, 0]) < 1e-8:
                        print(f"Warning: Standard deviation for channel {i:02d} is very small")
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
            # data shape: (..., C, H, W) where C is num_channels
            # mean/std shape: (num_channels, 1, 1)

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
            # data shape: (..., C, H, W) where C is num_channels
            # mean/std shape: (num_channels, 1, 1)

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
                - "data": input sequence data with shape (1, input_length, num_channels, H, W)
                - "label": target sequence with shape (1, max_k_steps, num_channels, H, W)

        Channel organization:
            - Channel 0: u (streamwise velocity)
            - Channel 1: v (wall-normal velocity)
            - Channel 2: w (spanwise velocity)
        """
        base_idx = self.indices[idx]
        description = (
            f"dataset: {self.__class__.__name__}, idx: {idx}, y_slice: {self.y_slice}, fields: {self.field_names}"
        )

        # Load input sequence + target frames with time_stride spacing
        frames = []
        total_frames_to_load = self.input_length + self.max_k_steps

        for i in range(total_frames_to_load):
            # Get the timestep for this frame with time_stride spacing
            # Example: time_stride=2, i=0 -> base_idx+0, i=1 -> base_idx+2, i=2 -> base_idx+4
            frame_offset = base_idx + i * self.time_stride
            timestep = self.timesteps[frame_offset]

            # Get the file path for this timestep
            fpath = self.file_dict[timestep]

            # Load multi-channel data
            with h5py.File(fpath, "r") as f:
                data_multi_channel = f["data"][()]  # Shape: (C, H, W) where C >= 3

                # Extract the requested fields (u, v, w) - indices 0, 1, 2
                channels = []
                for field_idx in range(len(self.field_names)):
                    data_2d = data_multi_channel[field_idx]  # Shape: (H, W)
                    channels.append(data_2d)

            # Stack channels: (num_channels, H, W)
            multi_channel_frame = np.stack(channels, axis=0)
            frames.append(multi_channel_frame)

        # Split into input and target sequences
        input_seq = np.stack(frames[: self.input_length], axis=0)  # (input_length, num_channels, H, W)
        target_frames = frames[self.input_length :]  # List of max_k_steps frames

        # Convert to tensors
        input_seq = torch.from_numpy(input_seq).float()  # (input_length, num_channels, H, W)
        target_seq = torch.from_numpy(np.stack(target_frames, axis=0)).float()  # (max_k_steps, num_channels, H, W)

        # Apply normalization
        input_seq = self.normalize(input_seq)
        target_seq = self.normalize(target_seq)

        # Add leading batch dimension of 1
        input_seq = input_seq.unsqueeze(0)  # (1, input_length, num_channels, H, W)
        target_seq = target_seq.unsqueeze(0)  # (1, max_k_steps, num_channels, H, W)

        # Structure like original dataset
        data = {"input_seq": input_seq}
        label = target_seq

        return {"description": np.array([description], dtype="<U200"), "data": data, "label": label}

    def get_channel_info(self):
        """Return information about channel organization."""
        info = {
            "y_slice": self.y_slice,
            "field_names": self.field_names,
            "num_channels": self.num_channels,
            "time_stride": self.time_stride,  # Add time_stride info
        }

        # Channel mapping
        channel_mapping = []
        for ch_idx, field_name in enumerate(self.field_names):
            channel_mapping.append(f"ch{ch_idx}: y{self.y_slice}_{field_name}")

        info["channel_mapping"] = channel_mapping
        return info

    def get_time_stride(self):
        """Return the time stride used in this dataset."""
        return self.time_stride
