"""
Enhanced dataset for 3D multi-plane flow sequence prediction.
支持同时加载3个y平面，每个平面包含uvwp四个通道的数据。
位置编码在网络层面处理。
包含12通道标准化功能。
"""

import glob
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FlowSequence3PlaneDataset(Dataset):
    """Dataset for 3D multi-plane temporal sequence prediction of flow fields.

    Loads 3 y-planes with 4 channels (u,v,w,p) each, for a total of 12 channels.
    Position encoding is handled at the network level.
    """

    def __init__(
        self,
        data_dir: str,
        input_length: int = 5,
        max_k_steps: int = 1,  # Number of future steps to load as ground truth
        field_names: list[str] = None,  # ["u", "v", "w", "p"]
        file_pattern: str = "*scale4-6-1_yslice*.h5",  # 匹配包含uvwp的文件
        resolution_scale: tuple[int, int, int] = (1, 4, 4),  # (z, y, x) downsampling
        y_slices: list[int] = None,  # 3个y平面的索引，如 [50, 75, 100]
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "train",
        enable_normalization: bool = True,
        norm_stats=None,  # Normalization statistics (file path or dict)
    ):
        """
        Args:
            data_dir: Directory containing HDF5 files
            input_length: Number of previous timesteps to use for prediction
            max_k_steps: Number of future steps to load as ground truth (for evaluation)
            field_names: List of fields to predict (default: ['u', 'v', 'w', 'p'])
            file_pattern: Pattern for HDF5 files (should match uvwp files)
            resolution_scale: Downsampling factor for (z, y, x) dimensions
            y_slices: List of 3 y-slice indices to load (if None, auto-select)
            train_ratio: Ratio of data for training
            valid_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            split: Dataset split ('train', 'val', 'test')
        """
        assert split in ["train", "val", "test"]
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

        if field_names is None:
            field_names = ["u", "v", "w", "p"]

        self.data_dir = data_dir
        self.input_length = input_length
        self.max_k_steps = max_k_steps
        self.field_names = field_names
        self.num_channels_per_plane = len(self.field_names)
        self.resolution_scale = resolution_scale
        self.split = split
        self.enable_normalization = enable_normalization

        # 总通道数 = 3个平面 × 4个物理通道 = 12
        self.num_planes = 3
        self.num_total_channels = self.num_planes * self.num_channels_per_plane

        # Get files for each y-plane separately and find common timesteps
        self.files_by_plane = {}
        self.timesteps = None

        for y_slice in y_slices:
            # Create specific pattern for this y_slice
            pattern_for_slice = f"*u-v-w-p_scale4-6-1_yslice{y_slice}_*.h5"
            files = sorted(glob.glob(os.path.join(data_dir, pattern_for_slice)))

            if not files:
                raise ValueError(
                    f"No files found for y_slice {y_slice} with pattern: {pattern_for_slice}"
                )

            self.files_by_plane[y_slice] = files

            # Extract timesteps from filenames
            timesteps = []
            for f in files:
                # Extract timestep from filename like "..._t00123.h5"
                import re

                match = re.search(r"_t(\d+)\.h5$", f)
                if match:
                    timesteps.append(int(match.group(1)))

            if self.timesteps is None:
                self.timesteps = set(timesteps)
            else:
                # Find intersection of timesteps across all planes
                self.timesteps = self.timesteps.intersection(set(timesteps))

        # Convert to sorted list
        self.timesteps = sorted(list(self.timesteps))
        self.num_frames = len(self.timesteps)
        # Each sample needs input_length + max_k_steps frames
        total_frames_needed = self.input_length + self.max_k_steps
        self.num_samples = self.num_frames - total_frames_needed + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough frames. Need at least {total_frames_needed}, got {self.num_frames}"
            )

        print(f"Found {self.num_frames} common timesteps across all planes")
        print(f"Y-planes: {list(self.files_by_plane.keys())}")
        print(
            f"Files per plane: {[len(files) for files in self.files_by_plane.values()]}"
        )

        # Filter out sequences that span the discontinuity between timestep 1080 and 1081
        # Remove sequences with starting timesteps that would include the discontinuity
        discontinuity_timestep = 1081
        exclude_start = (
            discontinuity_timestep - total_frames_needed + 1
        )  # Adjusted for max_k_steps
        exclude_end = discontinuity_timestep  # 1081

        valid_indices = []
        excluded_count = 0

        for i in range(self.num_samples):
            # Get the starting and ending timestep for this sequence
            start_timestep = self.timesteps[i]
            # end_timestep = self.timesteps[i + total_frames_needed - 1]  # (unused)

            # Exclude sequences that would span the discontinuity
            # This happens if the sequence starts from exclude_start to exclude_end
            if exclude_start <= start_timestep <= exclude_end:
                excluded_count += 1
                print(
                    f"Excluding sequence starting at timestep {start_timestep} (would span discontinuity)"
                )
            else:
                valid_indices.append(i)

        print(
            f"Excluded {excluded_count} sequences due to discontinuity at timestep {discontinuity_timestep}"
        )
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
            self.indices = [
                self.valid_base_indices[i]
                for i in range(train_samples, train_samples + valid_samples)
            ]
        else:  # test
            self.indices = [
                self.valid_base_indices[i]
                for i in range(train_samples + valid_samples, self.num_samples)
            ]

        print(
            f"Created {split} dataset with {len(self.indices)} samples from {self.num_frames} files"
        )

        # Get data shape and y_slices from first file
        self._get_data_shape_and_slices(y_slices)

        # Setup normalization
        self._setup_normalization(norm_stats)

    def _get_data_shape_and_slices(self, y_slices: list[int] | None):
        """Get the shape of 2D data and determine y_slices."""
        # Use the first file from the first plane
        first_plane_files = self.files_by_plane[y_slices[0]]
        with h5py.File(first_plane_files[0], "r") as f:
            # Data is stored as (C, H, W) where C can be 3 (u,v,w) or 4 (u,v,w,p)
            data_multi_channel = f["data"][()]  # Shape: (C, H, W)

            # Check that data shape is 3D and has at least the required number of channels
            if len(data_multi_channel.shape) != 3:
                raise ValueError(
                    f"Expected 3D data shape (C, H, W), got {data_multi_channel.shape}"
                )

            if data_multi_channel.shape[0] < len(self.field_names):
                raise ValueError(
                    f"Data file has {data_multi_channel.shape[0]} channels, "
                    f"but requested {len(self.field_names)} fields: {self.field_names}"
                )

            # The data is already 2D per y-slice, we just need to determine which y_slices are available
            # y_slices are determined by the filename pattern and file availability
            if y_slices is None:
                raise ValueError("y_slices must be provided for 3-plane dataset")
            else:
                assert len(y_slices) == 3, "Must provide exactly 3 y_slices"
                self.y_slices = y_slices

            # Get 2D shape from the data (H, W)
            self.data_shape = data_multi_channel.shape[1:]  # (H, W)

            print(
                f"Fields: {self.field_names} ({self.num_channels_per_plane} channels per plane)"
            )
            print(f"Data shape per file: {data_multi_channel.shape}")
            print(f"Selected y_slices: {self.y_slices}")
            print(f"2D shape per plane: {self.data_shape}")
            print(f"Total physical channels: {self.num_total_channels}")

    def _setup_normalization(self, norm_stats):
        """Setup normalization parameters for multi-channel 3-plane data."""
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
                print(
                    f"Warning: Could not load normalization stats from {norm_stats_path}: {e}"
                )
                print("Normalization will be disabled.")
                self.mean = None
                self.std = None
                return
        elif isinstance(norm_stats, dict):
            stats = norm_stats
        else:
            raise ValueError(
                f"norm_stats must be dict or file path, got {type(norm_stats)}"
            )

        # Extract per-channel statistics
        try:
            if "per_channel_stats" in stats and len(stats["per_channel_stats"]) > 0:
                # Per-channel normalization
                per_channel_stats = stats["per_channel_stats"]
                self.mean = []
                self.std = []

                # Build channel stats in order: [plane0_u, plane0_v, plane0_w, plane1_u, ...]
                for ch_idx in range(self.num_total_channels):
                    channel_key = f"channel_{ch_idx:02d}"

                    if channel_key in per_channel_stats:
                        self.mean.append(float(per_channel_stats[channel_key]["mean"]))
                        self.std.append(float(per_channel_stats[channel_key]["std"]))
                    else:
                        # Fallback to global stats if channel not found
                        print(
                            f"Warning: No stats found for {channel_key}, using global stats"
                        )
                        self.mean.append(float(stats["mean"]))
                        self.std.append(float(stats["std"]))

                # Convert to tensors for efficient computation
                self.mean = torch.tensor(self.mean, dtype=torch.float32).view(
                    -1, 1, 1
                )  # (num_channels, 1, 1)
                self.std = torch.tensor(self.std, dtype=torch.float32).view(
                    -1, 1, 1
                )  # (num_channels, 1, 1)
                self.per_channel_norm = True

                print(
                    f"{self.num_total_channels}-channel normalization enabled for {self.split} split:"
                )
                for ch_idx in range(min(6, len(self.mean))):  # Show first 6 channels
                    print(
                        f"  channel_{ch_idx:02d}: mean={self.mean[ch_idx, 0, 0]:.6f}, std={self.std[ch_idx, 0, 0]:.6f}"
                    )
                if len(self.mean) > 6:
                    print(f"  ... and {len(self.mean) - 6} more channels")

            else:
                # Global normalization fallback
                self.mean = float(stats["mean"])
                self.std = float(stats["std"])
                self.per_channel_norm = False
                print(
                    f"Global normalization enabled for {self.split} split: mean={self.mean:.6f}, std={self.std:.6f}"
                )

            # Validate std is not zero
            if self.per_channel_norm:
                for i in range(len(self.std)):
                    if abs(self.std[i, 0, 0]) < 1e-8:
                        print(
                            f"Warning: Standard deviation for channel {i:02d} is very small"
                        )
            else:
                if abs(self.std) < 1e-8:
                    print(
                        "Warning: Standard deviation is very small, normalization might be unstable"
                    )

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
            # data shape: (..., C, H, W) where C is num_total_channels
            # mean/std shape: (num_total_channels, 1, 1)

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
            # data shape: (..., C, H, W) where C is num_total_channels
            # mean/std shape: (num_total_channels, 1, 1)

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
                - "data": input sequence data with shape (1, input_length, num_total_channels, H, W)
                - "label": target sequence with shape (1, max_k_steps, num_total_channels, H, W)

        Channel organization (example for 3 fields):
            - Channels 0-2: Plane 0 (y_slice[0]) - [u, v, w]
            - Channels 3-5: Plane 1 (y_slice[1]) - [u, v, w]
            - Channels 6-8: Plane 2 (y_slice[2]) - [u, v, w]
        """
        base_idx = self.indices[idx]
        description = f"dataset: {self.__class__.__name__}, idx: {idx}, planes: {self.y_slices}, fields: {self.field_names}"

        # Load input sequence + target frames for all planes and channels
        frames = []
        total_frames_needed = self.input_length + self.max_k_steps

        for i in range(total_frames_needed):
            # Get the timestep for this frame
            timestep = self.timesteps[base_idx + i]

            # Load data for all 3 planes at this timestep
            plane_channels = []

            # 遍历3个平面，每个平面对应不同的文件
            for y_slice in self.y_slices:
                # Find the file for this y_slice and timestep
                target_filename = (
                    f"u-v-w-p_scale4-6-1_yslice{y_slice}_t{timestep:05d}.h5"
                )
                fpath = os.path.join(self.data_dir, target_filename)

                if not os.path.exists(fpath):
                    raise ValueError(f"File not found: {fpath}")

                # Load multi-channel data for this plane
                with h5py.File(fpath, "r") as f:
                    data_multi_channel = f["data"][()]  # Shape: (4, H, W)

                    # Extract each field (u, v, w, p) for this plane
                    for field_idx in range(len(self.field_names)):
                        data_2d = data_multi_channel[field_idx]  # Shape: (H, W)
                        plane_channels.append(data_2d)

            # Stack all channels: (num_total_channels, H, W)
            # Channel order: [plane0_field0, plane0_field1, ...,
            #                 plane1_field0, plane1_field1, ...,
            #                 plane2_field0, plane2_field1, ...]
            multi_channel_frame = np.stack(plane_channels, axis=0)
            frames.append(multi_channel_frame)

        # Split into input and target sequences
        input_seq = np.stack(
            frames[: self.input_length], axis=0
        )  # (input_length, num_total_channels, H, W)
        target_frames = frames[self.input_length :]  # List of max_k_steps frames

        # Convert to tensors
        input_seq = torch.from_numpy(
            input_seq
        ).float()  # (input_length, num_total_channels, H, W)
        target_seq = torch.from_numpy(
            np.stack(target_frames, axis=0)
        ).float()  # (max_k_steps, num_total_channels, H, W)

        # Apply normalization
        input_seq = self.normalize(input_seq)
        target_seq = self.normalize(target_seq)

        # Add leading batch dimension of 1
        input_seq = input_seq.unsqueeze(
            0
        )  # (1, input_length, num_total_channels, H, W)
        target_seq = target_seq.unsqueeze(
            0
        )  # (1, max_k_steps, num_total_channels, H, W)

        # Structure like original dataset
        data = {"input_seq": input_seq}
        label = target_seq

        return {
            "description": np.array([description], dtype="<U200"),
            "data": data,
            "label": label,
        }

    def get_channel_info(self):
        """Return information about channel organization."""
        info = {
            "num_planes": self.num_planes,
            "y_slices": self.y_slices,
            "field_names": self.field_names,
            "num_channels_per_plane": self.num_channels_per_plane,
            "num_total_channels": self.num_total_channels,
            "time_stride": 1,  # 3plane dataset uses consecutive frames (time_stride=1)
        }

        # 详细的通道映射
        channel_mapping = []
        ch_idx = 0

        for plane_idx in range(self.num_planes):
            y_slice = self.y_slices[plane_idx]

            # 物理通道
            for field_name in self.field_names:
                channel_mapping.append(
                    f"ch{ch_idx}: plane{plane_idx}_y{y_slice}_{field_name}"
                )
                ch_idx += 1

        info["channel_mapping"] = channel_mapping
        return info

    def get_time_stride(self):
        """Return the time stride used in this dataset (always 1 for 3plane)."""
        return 1
