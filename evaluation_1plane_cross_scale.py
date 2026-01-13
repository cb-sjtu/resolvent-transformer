#!/usr/bin/env python3
"""
Cross-scale evaluation script for 1-plane Flow Swin Transformer.
Alternates between small-scale (t) and large-scale (10t) models to extend autoregressive predictions.

Prediction strategy:
- Small-scale model: predicts consecutive steps (frames spaced by t)
- Large-scale model: predicts 1 step using every 10th frame (frames spaced by 10t)
- Alternates between the two models to achieve long-term predictions

Example prediction sequence:
Phase 0: Load historical GT at t=-19, -9 from validation set
Warm-up: Small-scale predicts t=6 to t=30, collecting anchors at t=1,11,21
Cycle 1: Small-scale predicts t=31 (candidate)
         Large-scale [-19,-9,1,11,21]→31, then FUSE at t=31
         Continue small-scale to t=40
Cycle 2: Small-scale predicts t=41 (candidate)
         Large-scale [-9,1,11,21,31]→41, then FUSE at t=41
...
"""

import os
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

warnings.filterwarnings("ignore")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Plots will only be saved locally.")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.datasets.flow_sequence_2d.flow_sequence_1plane import FlowSequence1PlaneDataset  # noqa: E402

# ========================================
# 🎯 CONFIGURATION: Modify these values to change settings everywhere
# ========================================
SMALL_SCALE_TIME_STRIDE = 1  # Small-scale model: consecutive frames (1t spacing)
LARGE_SCALE_TIME_STRIDE = 5  # Large-scale model: every 10th frame (10t spacing)


class CrossScaleEvaluator:
    """Evaluator that alternates between small-scale and large-scale models for extended predictions."""

    def __init__(
        self,
        small_scale_checkpoint: str,
        large_scale_checkpoint: str,
        small_scale_cfg: DictConfig,
        large_scale_cfg: DictConfig,
        data_config: dict,
    ):
        """Initialize the cross-scale evaluator.

        Args:
            small_scale_checkpoint: Path to small-scale model checkpoint (t spacing)
            large_scale_checkpoint: Path to large-scale model checkpoint (10t spacing)
            small_scale_cfg: Model configuration for small-scale model
            large_scale_cfg: Model configuration for large-scale model
            data_config: Data configuration dictionary
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load both models
        print("\n" + "=" * 60)
        print("Loading small-scale model (t spacing)...")
        print("=" * 60)
        self.small_scale_model = self._load_model(small_scale_checkpoint, small_scale_cfg)

        print("\n" + "=" * 60)
        print("Loading large-scale model (10t spacing)...")
        print("=" * 60)
        self.large_scale_model = self._load_model(large_scale_checkpoint, large_scale_cfg)

        # Setup datasets
        self.data_config = data_config
        self.small_scale_dataset = self._setup_dataset(time_stride=SMALL_SCALE_TIME_STRIDE)  # t spacing
        self.large_scale_dataset = self._setup_dataset(time_stride=LARGE_SCALE_TIME_STRIDE)  # 10t spacing

        # Create output directory
        self.output_dir = Path("evaluation_results") / "cross_scale_evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {self.output_dir}")

        # Initialize wandb if available
        if WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project="turbulence_cross_scale",
                name="cross_scale_evaluation",
                tags=["evaluation", "cross_scale", "1plane", "uvw"],
                config={
                    "small_scale_checkpoint": small_scale_checkpoint,
                    "large_scale_checkpoint": large_scale_checkpoint,
                    "device": str(self.device),
                },
            )
        else:
            self.wandb_run = None

    def _load_model(self, checkpoint_path: str, model_cfg: DictConfig) -> nn.Module:
        """Load a model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")

        from src.plmodules.flow_swin_2d_lit_module import FlowSwin2DLitModule

        try:
            model = FlowSwin2DLitModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
        except Exception as e:
            print(f"Could not load with checkpoint hyperparameters: {e}")
            print("Creating module with provided config...")
            module_cfg = OmegaConf.create(
                {
                    "model": model_cfg,
                    "loss_fn": "mse",
                    "accelerate": {
                        "fp32_matmul_precision": "high",
                        "dynamo_cache_size_limit": None,
                        "compile": False,
                        "sdpa": ["efficient"],
                    },
                }
            )
            model = FlowSwin2DLitModule(module_cfg)

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # Remove _metadata key if present (can cause loading issues)
                if "_metadata" in state_dict:
                    del state_dict["_metadata"]
                model.load_state_dict(state_dict, strict=False)
                print("Model weights loaded successfully!")

        model.eval()
        model.to(self.device)
        return model

    def _setup_dataset(self, time_stride: int) -> FlowSequence1PlaneDataset:
        """Setup dataset with specified time stride."""
        print(f"\nSetting up dataset with time_stride={time_stride}...")

        dataset = FlowSequence1PlaneDataset(
            data_dir=self.data_config["data_dir"],
            input_length=self.data_config["input_length"],
            field_names=self.data_config["field_names"],
            file_pattern=self.data_config["file_pattern"],
            resolution_scale=self.data_config["resolution_scale"],
            y_slice=self.data_config["y_slice"],
            train_ratio=self.data_config.get("train_ratio", 0.8),
            valid_ratio=self.data_config.get("valid_ratio", 0.05),
            test_ratio=self.data_config.get("test_ratio", 0.15),
            split="test",
            time_stride=time_stride,
            enable_normalization=self.data_config.get("enable_normalization", True),
            norm_stats=self.data_config.get("norm_stats", None),
        )

        print(f"Dataset size: {len(dataset)}")
        return dataset

    def load_consecutive_ground_truth(self, sample_idx: int, num_frames: int, time_stride: int = 1) -> list[np.ndarray]:
        """Load consecutive ground truth frames with specified time stride.

        This method directly loads frames from HDF5 files to ensure continuous temporal sampling,
        bypassing dataset filtering that might create gaps.

        Args:
            sample_idx: Starting sample index in the dataset
            num_frames: Number of consecutive frames to load
            time_stride: Time spacing between frames (1 for small-scale, 10 for large-scale)

        Returns:
            List of denormalized frames, each with shape (C, H, W)
        """
        # Get base index in the timesteps array
        base_idx = self.small_scale_dataset.indices[sample_idx]

        # The label corresponds to the frame after the input sequence
        # For small_scale_dataset with time_stride=1 and input_length=5:
        # - Input frames are at indices: base_idx, base_idx+1, ..., base_idx+4
        # - Label (first GT frame) is at index: base_idx + 5
        dataset_time_stride = self.small_scale_dataset.time_stride
        input_length = self.small_scale_dataset.input_length

        # Starting timestep index in the full timesteps array (first GT frame)
        start_timestep_idx = base_idx + input_length * dataset_time_stride

        print("\nLoading ground truth:")
        print(f"  Sample index: {sample_idx}")
        print(f"  Base index in timesteps array: {base_idx}")
        print(f"  Dataset time_stride: {dataset_time_stride}")
        print(f"  Starting GT timestep index: {start_timestep_idx}")
        print(f"  Requested time_stride for GT: {time_stride}")

        ground_truth_frames = []

        # Load all frames
        for i in range(num_frames):
            # Calculate the timestep index with desired time_stride
            timestep_idx = start_timestep_idx + i * time_stride

            if timestep_idx >= len(self.small_scale_dataset.timesteps):
                print(f"  Warning: timestep_idx {timestep_idx} out of range, stopping at frame {i}")
                break

            # Get the actual timestep number
            timestep = self.small_scale_dataset.timesteps[timestep_idx]

            if i < 5 or i % 20 == 0:  # Print first 5 and every 20th frame
                print(f"  Frame {i}: timestep_idx={timestep_idx}, timestep={timestep}")

            # Load directly from HDF5 file
            fpath = self.small_scale_dataset.file_dict[timestep]

            with h5py.File(fpath, "r") as f:
                data_multi_channel = f["data"][()]  # Shape: (C, H, W)

                # Extract the requested fields
                channels = []
                for field_idx in range(len(self.small_scale_dataset.field_names)):
                    channels.append(data_multi_channel[field_idx])

                frame = np.stack(channels, axis=0)  # (C, H, W)

            # HDF5 data is raw (not normalized), so we need to normalize then denormalize
            # to match the same processing as predictions
            frame_tensor = torch.from_numpy(frame).float()
            frame_normalized = self.small_scale_dataset.normalize(frame_tensor)
            frame_denorm = self.small_scale_dataset.denormalize(frame_normalized.unsqueeze(0).unsqueeze(0))
            ground_truth_frames.append(frame_denorm.cpu().numpy()[0, 0])

        print(f"  Loaded {len(ground_truth_frames)} ground truth frames\n")
        return ground_truth_frames

    def cross_scale_prediction(
        self,
        initial_frames: torch.Tensor,
        num_predictions: int = 100,
        fusion_weight: float = 0.5,
        sample_idx: int = 0,
        fusion_interval: int = 10,
        first_fusion_point: int = 31,
    ) -> tuple[np.ndarray, list[str], list[dict]]:
        """Perform MR-PC (Multi-Resolution Prediction-Correction) cross-scale prediction.

        Strategy:
        1. Warm-up phase: Use small-scale model to reach first_fusion_point-1, build frame cache
        2. First fusion at first_fusion_point using historical data
        3. Main loop: Fusion every fusion_interval steps with large-scale correction

        Args:
            initial_frames: Initial input sequence (B, T=5, C, H, W), frames at t=1,2,3,4,5
            num_predictions: Total number of future steps to predict
            fusion_weight: Weight for fusion, x_fused = (1-α)*x_small + α*x_large
            sample_idx: Sample index for accessing historical ground truth frames
            fusion_interval: Interval between fusion points in timesteps (default: 10)
            first_fusion_point: Time step of the first fusion point (default: 31)

        Returns:
            predictions: Array of all predicted frames (num_predictions, C, H, W)
            model_used: List indicating which model was used ('small', 'large', 'fused')
            fusion_info: List of dicts with fusion details for each fused frame
        """
        print(f"\n{'=' * 60}")
        print("Starting MR-PC cross-scale prediction")
        print(f"Total predictions: {num_predictions}")
        print(f"Fusion weight α: {fusion_weight}")
        print(f"Fusion interval: {fusion_interval}t")
        print(f"First fusion point: t={first_fusion_point}")
        print(f"{'=' * 60}")

        # Calculate derived parameters based on first_fusion_point
        # Warmup predicts from t=6 to t=(first_fusion_point-1)
        # Initial frames are at t=1-5, current_time starts at 5
        # So warmup_steps = (first_fusion_point - 1) - 5 = first_fusion_point - 6
        warmup_steps = first_fusion_point - 6
        historical_start = first_fusion_point - 5 * LARGE_SCALE_TIME_STRIDE
        historical_end = 0
        num_historical_frames = max(0, 1 - historical_start)

        # Calculate anchor times (LARGE_SCALE_TIME_STRIDE intervals before first fusion point, within warmup range)
        anchor_times = []
        for i in [5, 4, 3, 2, 1]:
            offset = i * LARGE_SCALE_TIME_STRIDE
            anchor_t = first_fusion_point - offset
            if 1 <= anchor_t <= first_fusion_point - 1:
                anchor_times.append(anchor_t)

        print("Derived parameters:")
        print(f"  Warmup steps: {warmup_steps} (predict t=6 to t={first_fusion_point - 1})")
        print(f"  Historical frames: t={historical_start} to t={historical_end} ({num_historical_frames} frames)")
        print(f"  Anchor times during warmup: {anchor_times}")

        self.small_scale_model.eval()
        self.large_scale_model.eval()

        predictions = []
        model_used = []
        fusion_info = []

        # Initialize queues
        # S: small-step window (length=5, t-spacing), for small-scale model
        # B: anchor window (length=5, 10t-spacing), for large-scale model (kept for compatibility)
        # all_frames: complete history of all frames
        # frame_cache: dictionary mapping time -> frame tensor for efficient lookup

        # initial_frames: (1, 5, C, H, W) -> frames at t=1,2,3,4,5
        initial_list = [initial_frames[0, i].clone() for i in range(5)]
        all_frames = initial_list.copy()  # Complete history

        S = initial_list.copy()  # [x[1], x[2], x[3], x[4], x[5]]
        B = []  # Will be filled with anchors at t=-19,-9,1,11,21 for first fusion at t=31

        # Frame cache: stores all frames for efficient large-scale input construction
        frame_cache = {}
        for i, frame in enumerate(initial_list):
            frame_cache[i + 1] = frame.clone()  # Store t=1,2,3,4,5

        current_time = 5  # Current time in units of t

        print("\nInitialization:")
        print("  S (small window): frames at t=[1,2,3,4,5]")
        print("  B (anchor queue): will use historical GT for t=-19,-9, predictions for t=1,11,21")
        print("  Frame cache: initialized with t=1 to t=5")
        print(f"  Current time: {current_time}t")

        with torch.no_grad():
            # ========================================
            # Phase 0: Load ALL historical ground truth frames
            # ========================================
            print(f"\n{'=' * 60}")
            print("Phase 0: Loading historical ground truth frames")
            print(f"{'=' * 60}")

            # Load historical frames from historical_start to 0
            # This ensures we have all frames needed for large-scale input construction
            if sample_idx >= num_historical_frames:
                print(f"  Loading historical GT frames from t={historical_start} to t=0...")

                for t in range(historical_start, 1):  # t: historical_start to 0
                    offset = 1 - t  # Calculate offset from current sample
                    hist_sample = self.small_scale_dataset[sample_idx - offset]
                    hist_frame = hist_sample["data"]["input_seq"][0, -1].to(self.device)  # (C, H, W)
                    frame_cache[t] = hist_frame.clone()

                print(f"  ✓ Loaded {len([t for t in frame_cache if t <= 0])} historical frames")

                # For backward compatibility with B
                if historical_start <= -19:
                    B.append(frame_cache[-19].clone())
                if historical_start <= -9:
                    B.append(frame_cache[-9].clone())
            else:
                print(f"  WARNING: Not enough history (sample_idx={sample_idx}, need >={num_historical_frames})")
                print("           Using initial frames as fallback")
                # Fallback: create dummy historical frames
                for t in range(historical_start, 1):
                    frame_cache[t] = initial_list[0].clone()
                if historical_start <= -19:
                    B.append(initial_list[0].clone())
                if historical_start <= -9:
                    B.append(initial_list[2].clone())

            # ========================================
            # Phase 1: Warm-up
            # ========================================
            print(f"\n{'=' * 60}")
            print("Phase 1: Warm-up - Building anchor queue B")
            print(f"{'=' * 60}")

            # Add initial frame at t=1 to B if it's an anchor
            if 1 in anchor_times:
                B.append(initial_list[0].clone())  # t=1
                print("  Anchor at t=1 added to B")

            # warmup_steps already calculated based on first_fusion_point
            for step in range(warmup_steps):
                # Small-scale prediction
                small_input = torch.stack(S, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                next_pred = self.small_scale_model(small_input)[0]  # (C, H, W)

                current_time += 1
                all_frames.append(next_pred.clone())
                predictions.append(next_pred.cpu())
                model_used.append("small")

                # Store in frame cache
                frame_cache[current_time] = next_pred.clone()

                # Update S: slide window
                S = S[1:] + [next_pred]

                # Check if this is an anchor point
                if current_time in anchor_times:
                    B.append(next_pred.clone())
                    print(f"  Step {step + 1}/{warmup_steps}: t={current_time}, Small-scale (★ Anchor added to B)")
                else:
                    print(f"  Step {step + 1}/{warmup_steps}: t={current_time}, Small-scale")

            print("\nWarm-up complete!")
            print(f"  B now contains {len(B)} anchors")
            print(f"  Frame cache now contains {len(frame_cache)} frames (t={historical_start} to t={current_time})")
            print(f"  Current time: {current_time}t (should be {first_fusion_point - 1})")

            # ========================================
            # Phase 2: Main MR-PC loop
            # ========================================
            print(f"\n{'=' * 60}")
            print("Phase 2: MR-PC Main Loop")
            print(f"{'=' * 60}")

            cycle_count = 0
            while len(predictions) < num_predictions:
                cycle_count += 1
                print(f"\n--- Cycle {cycle_count} (starting at t={current_time}) ---")

                # Step 1: Small-scale prediction for first step (will be fused)
                if len(predictions) >= num_predictions:
                    break

                small_input = torch.stack(S, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                small_pred_first = self.small_scale_model(small_input)[0]  # (C, H, W)

                current_time += 1
                all_frames.append(small_pred_first.clone())
                print(f"  Small-scale: t={current_time} (candidate for fusion)")

                # Step 2: Large-scale prediction using frame cache
                # Construct large-scale input: 5 frames with LARGE_SCALE_TIME_STRIDE spacing
                # Input times: [current_time-4*stride, current_time-3*stride, ..., current_time-stride]
                large_input_times = [
                    current_time - 5 * LARGE_SCALE_TIME_STRIDE,
                    current_time - 4 * LARGE_SCALE_TIME_STRIDE,
                    current_time - 3 * LARGE_SCALE_TIME_STRIDE,
                    current_time - 2 * LARGE_SCALE_TIME_STRIDE,
                    current_time - 1 * LARGE_SCALE_TIME_STRIDE,
                ]

                # Get frames from cache
                large_input_frames = []
                for t in large_input_times:
                    if t in frame_cache:
                        large_input_frames.append(frame_cache[t])
                    else:
                        # Fallback: use nearest available frame
                        available_times = sorted([k for k in frame_cache if k <= t])
                        if available_times:
                            fallback_t = available_times[-1]
                            large_input_frames.append(frame_cache[fallback_t])
                            print(f"    WARNING: t={t} not in cache, using t={fallback_t}")
                        else:
                            # Extreme fallback: use first initial frame
                            large_input_frames.append(initial_list[0])
                            print(f"    WARNING: t={t} not in cache, using initial frame")

                large_input = torch.stack(large_input_frames, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                large_pred = self.large_scale_model(large_input)[0]  # (C, H, W)

                print(f"  Large-scale: using frames at t={large_input_times} → t={current_time}")

                # Step 3: Fusion at current time
                x_fused = (1 - fusion_weight) * small_pred_first + fusion_weight * large_pred

                # Record fusion info
                fusion_info.append(
                    {
                        "time": current_time,
                        "cycle": cycle_count,
                        "fusion_weight": fusion_weight,
                        "small_pred": small_pred_first.cpu(),
                        "large_pred": large_pred.cpu(),
                        "fused": x_fused.cpu(),
                    }
                )

                # Add fused result
                predictions.append(x_fused.cpu())
                model_used.append("fused")
                all_frames[-1] = x_fused.clone()  # Replace with fused version

                # Store in frame cache
                frame_cache[current_time] = x_fused.clone()

                print(f"  Fusion: x[{current_time}] = {1 - fusion_weight:.2f}*small + {fusion_weight:.2f}*large")

                # Update S with fused result
                S = S[1:] + [x_fused]

                # Update B with new anchor (for backward compatibility, though not strictly needed)
                if current_time % LARGE_SCALE_TIME_STRIDE == 1:  # Only update B at stride intervals
                    B = B[1:] + [x_fused]
                    updated_anchor_times = [
                        current_time - 4 * LARGE_SCALE_TIME_STRIDE + (i * LARGE_SCALE_TIME_STRIDE) for i in range(5)
                    ]
                    print(f"  B updated: anchors now at t={updated_anchor_times}")

                # Step 4: Continue with small-scale for next (fusion_interval - 1) steps
                for _ in range(fusion_interval - 1):
                    if len(predictions) >= num_predictions:
                        break

                    small_input = torch.stack(S, dim=0).unsqueeze(0)
                    next_pred = self.small_scale_model(small_input)[0]

                    current_time += 1
                    all_frames.append(next_pred.clone())
                    predictions.append(next_pred.cpu())
                    model_used.append("small")

                    # Store in frame cache
                    frame_cache[current_time] = next_pred.clone()

                    S = S[1:] + [next_pred]

                    print(f"  Small-scale: t={current_time}")

        # Stack all predictions
        pred_array = torch.stack(predictions, dim=0).cpu().numpy()  # (num_predictions, C, H, W)

        print(f"\n{'=' * 60}")
        print("MR-PC Prediction Complete!")
        print(f"{'=' * 60}")
        print(f"Total predictions: {len(predictions)}")
        print(f"Small-scale only: {model_used.count('small')}")
        print(f"Fused (small+large): {model_used.count('fused')}")
        print(f"Fusion events: {len(fusion_info)}")

        return pred_array, model_used, fusion_info

    def pure_small_scale_prediction(self, initial_frames: torch.Tensor, num_predictions: int = 100) -> np.ndarray:
        """Pure small-scale autoregressive prediction (baseline)."""
        print(f"\nRunning pure small-scale baseline ({num_predictions} steps)...")

        self.small_scale_model.eval()
        predictions = []

        # Initialize with initial frames
        S = [initial_frames[0, i].clone() for i in range(5)]

        with torch.no_grad():
            for _step in range(num_predictions):
                small_input = torch.stack(S, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                next_pred = self.small_scale_model(small_input)[0]  # (C, H, W)

                predictions.append(next_pred.cpu())
                S = S[1:] + [next_pred]

        pred_array = torch.stack(predictions, dim=0).cpu().numpy()
        print(f"Pure small-scale baseline complete: {len(predictions)} predictions")
        return pred_array

    def pure_large_scale_prediction(
        self, initial_frames: torch.Tensor, num_predictions: int = 100, sample_idx: int = 0
    ) -> np.ndarray:
        """Pure large-scale autoregressive prediction (baseline).

        Uses the same historical frame strategy as MR-PC for fair comparison:
        - Loads ground truth frames at t=-19 and t=-9 from dataset
        - Uses anchors at [-19, -9, 1, 11, 21] to predict t=31
        - Then continues autoregressively with 10t spacing

        Args:
            initial_frames: Initial input sequence (1, 5, C, H, W) at t=1,2,3,4,5
            num_predictions: Number of steps to predict
            sample_idx: Index of current sample (needed to access historical frames)

        Returns:
            Predictions array of shape (num_predictions, C, H, W)
        """
        print(f"\nRunning pure large-scale baseline ({num_predictions} steps)...")

        self.large_scale_model.eval()
        predictions = []

        with torch.no_grad():
            # Phase 0: Load historical ground truth frames at t=-2*stride and t=-stride
            historical_frames = []
            if sample_idx >= 2 * LARGE_SCALE_TIME_STRIDE:
                # Load frame at t=-2*stride (2*stride samples back)
                hist_sample_2stride = self.small_scale_dataset[sample_idx - 2 * LARGE_SCALE_TIME_STRIDE]
                hist_frame_2stride = hist_sample_2stride["data"]["input_seq"][0, -1].to(self.device)  # (C, H, W)
                historical_frames.append(hist_frame_2stride.clone())
                print(
                    f"Loaded historical frame at t=-{2 * LARGE_SCALE_TIME_STRIDE} \
                        from sample {sample_idx - 2 * LARGE_SCALE_TIME_STRIDE}"
                )

                # Load frame at t=-stride (stride samples back)
                hist_sample_stride = self.small_scale_dataset[sample_idx - LARGE_SCALE_TIME_STRIDE]
                hist_frame_stride = hist_sample_stride["data"]["input_seq"][0, -1].to(self.device)  # (C, H, W)
                historical_frames.append(hist_frame_stride.clone())
                print(
                    f"Loaded historical frame at t=-{LARGE_SCALE_TIME_STRIDE} from sample\
                         {sample_idx - LARGE_SCALE_TIME_STRIDE}"
                )
            else:
                # If not enough history, fall back to using early frames from initial_frames
                print(
                    f"Warning: sample_idx={sample_idx} < {2 * LARGE_SCALE_TIME_STRIDE}, cannot load historical frames"
                )
                print("Using first two frames from initial_frames as fallback")
                historical_frames = [initial_frames[0, 0].clone(), initial_frames[0, 1].clone()]

            # Initialize anchor queue B with historical frames
            B = historical_frames.copy()  # [t=-19, t=-9]

            # Add t=1 from initial frames
            B.append(initial_frames[0, 0].clone())

            # Phase 1: Warm-up - predict t=6 to t=30 using small-scale model, collect anchors at t=11, 21
            print("\nPhase 1: Warm-up (small-scale) - predicting t=6 to t=30")
            S = [initial_frames[0, i].clone() for i in range(5)]  # t=1,2,3,4,5
            all_frames = S.copy()
            current_time = 5

            warmup_steps = 25  # Predict from t=6 to t=30
            for _ in range(warmup_steps):
                small_input = torch.stack(S, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                next_pred = self.small_scale_model(small_input)[0]  # (C, H, W)

                current_time += 1
                all_frames.append(next_pred.clone())
                S = S[1:] + [next_pred]

                # Collect anchors at t=11 and t=21
                if current_time in [11, 21]:
                    B.append(next_pred.clone())
                    print(f"  Collected anchor at t={current_time}")

            # B now contains: [t=-19, t=-9, t=1, t=11, t=21]
            print("Anchor queue B ready with 5 frames: t=[-19, -9, 1, 11, 21]")

            # Phase 2: Main loop - use large-scale model autoregressively
            print("\nPhase 2: Large-scale predictions (every 10t)")
            print("=" * 60)

            remaining = num_predictions
            prediction_count = 0

            while remaining > 0:
                # Prepare input for large-scale model
                large_input = torch.stack(B, dim=0).unsqueeze(0)  # (1, 5, C, H, W)

                # Large-scale prediction
                next_pred = self.large_scale_model(large_input)[0]  # (C, H, W)

                # Store prediction
                predictions.append(next_pred.cpu())
                prediction_count += 1

                # Update anchor queue
                B = B[1:] + [next_pred]

                remaining -= 1

                # Log every 10 predictions
                if prediction_count % 10 == 0:
                    print(f"  Generated {prediction_count}/{num_predictions} predictions")

        pred_array = torch.stack(predictions, dim=0).cpu().numpy()
        print(f"\nPure large-scale baseline complete: {len(predictions)} predictions")
        return pred_array

    def visualize_cross_scale_prediction(
        self,
        sample_idx: int = 0,
        num_predictions: int = 100,
        fusion_weight: float = 0.5,
        fusion_interval: int = 10,
        first_fusion_point: int = 31,
    ):
        """Visualize MR-PC cross-scale prediction results with baselines."""
        print(f"\n{'=' * 60}")
        print(f"Evaluating sample {sample_idx}")
        print(f"{'=' * 60}")

        # Get initial sample from small-scale dataset
        sample = self.small_scale_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)  # (1, T, C, H, W)

        # 1. Perform MR-PC cross-scale prediction
        pred_seq_mrpc, model_used, fusion_info = self.cross_scale_prediction(
            input_seq, num_predictions, fusion_weight, sample_idx, fusion_interval, first_fusion_point
        )

        # 2. Perform pure small-scale baseline
        pred_seq_small = self.pure_small_scale_prediction(input_seq, num_predictions)

        # 3. Perform pure large-scale baseline
        pred_seq_large = self.pure_large_scale_prediction(input_seq, num_predictions, sample_idx)

        # Denormalize all predictions
        pred_seq_mrpc_denorm = (
            self.small_scale_dataset.denormalize(torch.from_numpy(pred_seq_mrpc).unsqueeze(0)).cpu().numpy()[0]
        )

        pred_seq_small_denorm = (
            self.small_scale_dataset.denormalize(torch.from_numpy(pred_seq_small).unsqueeze(0)).cpu().numpy()[0]
        )

        pred_seq_large_denorm = (
            self.small_scale_dataset.denormalize(torch.from_numpy(pred_seq_large).unsqueeze(0)).cpu().numpy()[0]
        )

        # Collect ground truth for comparison
        # For small_scale_dataset with time_stride=1:
        #   - dataset[idx] has label at timestep: base_timestep + 5
        #   - dataset[idx+1] has label at timestep: base_timestep + 6
        #   - So consecutive samples give consecutive timesteps
        print("\nLoading ground truth frames...")
        ground_truth_frames = []

        # First, check if we can use the simple method (consecutive dataset samples)
        # This works if the dataset has time_stride=1 and no gaps in the range we need
        dataset_time_stride = self.small_scale_dataset.time_stride
        print(f"  Small-scale dataset time_stride: {dataset_time_stride}")

        if dataset_time_stride == 1:
            # Simple method: consecutive samples have consecutive labels
            print(f"  Using consecutive dataset samples for {num_predictions} frames")
            for i in range(num_predictions):
                if sample_idx + i < len(self.small_scale_dataset):
                    sample_i = self.small_scale_dataset[sample_idx + i]
                    target_i = sample_i["label"]  # (1, 1, C, H, W)
                    target_denorm = self.small_scale_dataset.denormalize(target_i)
                    target_frame = target_denorm.cpu().numpy()[0, 0]  # (C, H, W)
                    ground_truth_frames.append(target_frame)

                    if i < 5:  # Debug: print first 5 frames
                        base_idx = self.small_scale_dataset.indices[sample_idx + i]
                        timestep_idx = base_idx + self.small_scale_dataset.input_length
                        timestep = self.small_scale_dataset.timesteps[timestep_idx]
                        print(f"    GT frame {i}: dataset[{sample_idx + i}] -> timestep {timestep}")

                        # Additional debug: print value at specific point
                        if i == 0:
                            print(f"    [DEBUG] Sample idx: {sample_idx}")
                            print(f"    [DEBUG] First GT timestep: {timestep}")
                            print(
                                f"    [DEBUG] First GT value at point (64,64), channel 0: {target_frame[0, 64, 64]:.6f}"
                            )
                            print(f"    [DEBUG] GT shape: {target_frame.shape}")
                else:
                    print(
                        f"  Warning: sample_idx + {i} = {sample_idx + i} >= \
                            dataset length {len(self.small_scale_dataset)}"
                    )
                    if ground_truth_frames:
                        ground_truth_frames.append(ground_truth_frames[-1])

            print(f"  Loaded {len(ground_truth_frames)} ground truth frames")
        else:
            # For time_stride != 1, use the direct HDF5 loading method
            print(f"  Using direct HDF5 loading (time_stride={dataset_time_stride})")
            ground_truth_frames = self.load_consecutive_ground_truth(
                sample_idx=sample_idx, num_frames=num_predictions, time_stride=1
            )

        # Get channel info
        channel_info = self.small_scale_dataset.get_channel_info()
        field_names = channel_info["field_names"]  # ["u", "v", "w"]

        # Save complete 3-channel spatial-temporal data as npy files
        gt_array = np.stack(ground_truth_frames, axis=0)  # (T, C, H, W)
        np.save(self.output_dir / f"pred_mrpc_sample_{sample_idx}.npy", pred_seq_mrpc_denorm)
        np.save(self.output_dir / f"pred_small_sample_{sample_idx}.npy", pred_seq_small_denorm)
        np.save(self.output_dir / f"pred_large_sample_{sample_idx}.npy", pred_seq_large_denorm)
        np.save(self.output_dir / f"ground_truth_sample_{sample_idx}.npy", gt_array)
        print(f"Saved complete spatial-temporal data for sample {sample_idx}:")
        print(f"  - pred_mrpc_sample_{sample_idx}.npy: {pred_seq_mrpc_denorm.shape}")
        print(f"  - pred_small_sample_{sample_idx}.npy: {pred_seq_small_denorm.shape}")
        print(f"  - pred_large_sample_{sample_idx}.npy: {pred_seq_large_denorm.shape}")
        print(f"  - ground_truth_sample_{sample_idx}.npy: {gt_array.shape}")

        # Save fusion point predictions (small-scale and large-scale before fusion)
        if len(fusion_info) > 0:
            # Get shape info
            T, C, H, W = pred_seq_mrpc_denorm.shape

            # Create sparse arrays for fusion point predictions (initialized with NaN)
            pred_small_at_fusion = np.full((T, C, H, W), np.nan, dtype=np.float32)
            pred_large_at_fusion = np.full((T, C, H, W), np.nan, dtype=np.float32)

            # Lists to store fusion point indices and weights
            fusion_point_indices = []
            fusion_weights = []

            # Extract and denormalize fusion point predictions
            for info in fusion_info:
                current_time = info["time"]  # Absolute time (e.g., 31, 41, 51, ...)
                fusion_weight = info["fusion_weight"]

                # Convert absolute time to prediction sequence index
                # Predictions start at time=6 (after initial frames at t=1-5)
                # So prediction index = current_time - 6
                pred_idx = current_time - 6

                # Skip if index is out of bounds (shouldn't happen but safety check)
                if pred_idx < 0 or pred_idx >= T:
                    print(
                        f"Warning: Fusion point at time {current_time} (pred_idx={pred_idx}) "
                        f"is out of bounds [0, {T}), skipping..."
                    )
                    continue

                # Get predictions (these are CPU tensors in normalized space)
                small_pred_tensor = info["small_pred"]  # (C, H, W)
                large_pred_tensor = info["large_pred"]  # (C, H, W)

                # Denormalize predictions
                # Add batch dimension for denormalization, then remove it
                small_pred_denorm = (
                    self.small_scale_dataset.denormalize(small_pred_tensor.unsqueeze(0)).cpu().numpy()[0]
                )
                large_pred_denorm = (
                    self.small_scale_dataset.denormalize(large_pred_tensor.unsqueeze(0)).cpu().numpy()[0]
                )

                # Fill in the sparse arrays at fusion points
                pred_small_at_fusion[pred_idx] = small_pred_denorm
                pred_large_at_fusion[pred_idx] = large_pred_denorm

                # Record fusion point info (using prediction index, not absolute time)
                fusion_point_indices.append(pred_idx)
                fusion_weights.append(fusion_weight)

            # Save fusion point predictions and metadata
            np.save(self.output_dir / f"pred_small_at_fusion_sample_{sample_idx}.npy", pred_small_at_fusion)
            np.save(self.output_dir / f"pred_large_at_fusion_sample_{sample_idx}.npy", pred_large_at_fusion)
            np.save(
                self.output_dir / f"fusion_points_sample_{sample_idx}.npy",
                np.array(fusion_point_indices, dtype=np.int32),
            )
            np.save(
                self.output_dir / f"fusion_weights_sample_{sample_idx}.npy", np.array(fusion_weights, dtype=np.float32)
            )

            print(f"\nSaved fusion point data for sample {sample_idx}:")
            print(f"  - pred_small_at_fusion_sample_{sample_idx}.npy: {pred_small_at_fusion.shape}")
            print(f"  - pred_large_at_fusion_sample_{sample_idx}.npy: {pred_large_at_fusion.shape}")
            print(f"  - fusion_points_sample_{sample_idx}.npy: {len(fusion_point_indices)} fusion points")
            print(f"  - fusion_weights_sample_{sample_idx}.npy: {len(fusion_weights)} weights")
            print(f"  Fusion points (prediction indices): {fusion_point_indices}")
            # Also print the mapping to absolute time for reference
            if len(fusion_point_indices) > 0:
                abs_times = [idx + 6 for idx in fusion_point_indices]
                print(f"  Corresponding absolute times: {abs_times}")

        # Create visualizations with baselines
        self._create_temporal_evolution_plot(
            pred_seq_mrpc_denorm,
            pred_seq_small_denorm,
            pred_seq_large_denorm,
            ground_truth_frames,
            model_used,
            fusion_info,
            field_names,
            sample_idx,
        )

        self._create_error_analysis(
            pred_seq_mrpc_denorm,
            pred_seq_small_denorm,
            pred_seq_large_denorm,
            ground_truth_frames,
            model_used,
            fusion_info,
            field_names,
            sample_idx,
        )

        self._create_spatial_comparison(
            pred_seq_mrpc_denorm,
            pred_seq_small_denorm,
            pred_seq_large_denorm,
            ground_truth_frames,
            model_used,
            field_names,
            sample_idx,
            num_predictions,
        )

        # Create fusion analysis plot
        if fusion_info:
            self._create_fusion_analysis(pred_seq_mrpc_denorm, fusion_info, field_names, sample_idx)

        # Create energy spectrum comparison
        self._create_energy_spectrum_comparison(
            pred_seq_mrpc_denorm,
            pred_seq_small_denorm,
            ground_truth_frames,
            field_names,
            sample_idx,
        )

    def _create_temporal_evolution_plot(
        self,
        pred_seq_mrpc,
        pred_seq_small,
        pred_seq_large,
        ground_truth_frames,
        model_used,
        fusion_info,
        field_names,
        sample_idx,
    ):
        """Create temporal evolution plots for multiple points comparing MR-PC with baselines."""
        print("\nCreating temporal evolution plots for 9 points with baselines...")

        # Fixed 9 monitoring points (same as evaluation_1plane_new.py)
        # Base 2D positions in the domain (z_index, x_index)
        # Adjusted monitoring points for 256x256 domain (indices 0-255)
        fixed_positions = [
            (40, 40),  # Bottom-left region
            (40, 64),  # Bottom-center
            (40, 100),  # Bottom-right
            (64, 40),  # Center-left
            (64, 64),  # Center-center
            (64, 100),  # Center-right
            (100, 40),  # Top-left
            (100, 64),  # Top-center
            (100, 100),  # Top-right
        ]

        # Create points list with grid indices for labeling
        points = []
        for idx, (z_pos, x_pos) in enumerate(fixed_positions):
            h_idx = idx // 3  # Row index (0, 1, 2)
            w_idx = idx % 3  # Column index (0, 1, 2)
            points.append((z_pos, x_pos, h_idx, w_idx))

        print(f"  Monitoring {len(points)} fixed points")

        num_steps = len(pred_seq_mrpc)
        time_steps = np.arange(1, num_steps + 1)

        # Create separate figure for each point
        for point_idx, (point_h, point_w, h_idx, w_idx) in enumerate(points):
            print(f"  Creating plot for point {point_idx + 1}/{len(points)} at ({point_h}, {point_w})")

            # Create figure with subplots for each field
            fig, axes = plt.subplots(3, 1, figsize=(18, 14))

            for field_idx, field_name in enumerate(field_names):
                ax = axes[field_idx]

                # Extract values at the selected point for all methods
                mrpc_values = pred_seq_mrpc[:, field_idx, point_h, point_w]
                small_values = pred_seq_small[:, field_idx, point_h, point_w]
                large_values = pred_seq_large[:, field_idx, point_h, point_w]

                if len(ground_truth_frames) > 0:
                    gt_values = np.array([gt[field_idx, point_h, point_w] for gt in ground_truth_frames])
                else:
                    gt_values = None

                # Plot ground truth with markers
                if gt_values is not None:
                    ax.plot(
                        time_steps[: len(gt_values)],
                        gt_values,
                        "k-",
                        linewidth=3,
                        label="Ground Truth",
                        alpha=0.8,
                        zorder=5,
                    )
                    # Add markers to show individual GT data points
                    ax.plot(
                        time_steps[: len(gt_values)],
                        gt_values,
                        "ko",
                        markersize=4,
                        alpha=0.6,
                        zorder=5,
                    )

                # Plot baseline predictions
                ax.plot(time_steps, small_values, "b--", linewidth=1.5, label="Pure Small-scale", alpha=0.6, zorder=2)
                ax.plot(
                    time_steps,
                    large_values,
                    "orange",
                    linestyle=":",
                    linewidth=2,
                    label="Pure Large-scale",
                    alpha=0.6,
                    zorder=2,
                )

                # Plot MR-PC predictions with markers at fusion points
                fused_mask = np.array([m == "fused" for m in model_used])

                # MR-PC line
                ax.plot(time_steps, mrpc_values, "g-", linewidth=2, label="MR-PC (ours)", alpha=0.8, zorder=4)

                # Highlight fusion points
                if np.any(fused_mask):
                    ax.scatter(
                        time_steps[fused_mask],
                        mrpc_values[fused_mask],
                        c="red",
                        marker="*",
                        s=200,
                        label="Fusion points",
                        alpha=0.9,
                        edgecolors="darkred",
                        linewidths=1.5,
                        zorder=6,
                    )

                # Mark warm-up phase
                warmup_end = 20
                if warmup_end < num_steps:
                    ax.axvline(
                        x=warmup_end,
                        color="purple",
                        linestyle="-.",
                        linewidth=2,
                        alpha=0.5,
                        label="Warm-up end",
                        zorder=1,
                    )

                ax.set_xlabel("Time Step (t)", fontsize=13)
                ax.set_ylabel(f"{field_name.upper()} Value", fontsize=13)
                ax.set_title(
                    f"{field_name.upper()} Evolution at Point ({point_h}, {point_w})", fontsize=15, fontweight="bold"
                )
                ax.legend(fontsize=11, loc="best", framealpha=0.9)
                ax.grid(True, alpha=0.3)

            # Add position indicator in title
            position_label = f"Grid Position: Row {h_idx + 1}/3, Col {w_idx + 1}/3"
            plt.suptitle(
                f"MR-PC vs Baselines - Temporal Evolution - Sample {sample_idx}\n{position_label}",
                fontsize=17,
                fontweight="bold",
            )
            plt.tight_layout()

            # Save figure with position indicator
            output_path = self.output_dir / f"temporal_evolution_sample_{sample_idx}_point_{point_idx + 1}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"    Saved: {output_path}")

            if self.wandb_run:
                self.wandb_run.log(
                    {f"temporal_evolution_sample_{sample_idx}_point_{point_idx + 1}": wandb.Image(str(output_path))}
                )

            plt.close()

        # Create additional plots without large-scale baseline for clearer comparison
        print("\n  Creating plots without large-scale baseline for clearer comparison...")
        for point_idx, (point_h, point_w, h_idx, w_idx) in enumerate(points):
            print(f"  Creating plot (no large-scale) for point {point_idx + 1}/{len(points)} at ({point_h}, {point_w})")

            # Create figure with subplots for each field
            fig, axes = plt.subplots(3, 1, figsize=(18, 14))

            for field_idx, field_name in enumerate(field_names):
                ax = axes[field_idx]

                # Extract values at the selected point for MR-PC and small-scale only
                mrpc_values = pred_seq_mrpc[:, field_idx, point_h, point_w]
                small_values = pred_seq_small[:, field_idx, point_h, point_w]

                if len(ground_truth_frames) > 0:
                    gt_values = np.array([gt[field_idx, point_h, point_w] for gt in ground_truth_frames])
                else:
                    gt_values = None

                # Plot ground truth with markers
                if gt_values is not None:
                    ax.plot(
                        time_steps[: len(gt_values)],
                        gt_values,
                        "k-",
                        linewidth=3,
                        label="Ground Truth",
                        alpha=0.8,
                        zorder=5,
                    )
                    # Add markers to show individual GT data points
                    ax.plot(
                        time_steps[: len(gt_values)],
                        gt_values,
                        "ko",
                        markersize=4,
                        alpha=0.6,
                        zorder=5,
                    )

                # Plot small-scale baseline
                ax.plot(time_steps, small_values, "b--", linewidth=2, label="Pure Small-scale", alpha=0.7, zorder=2)

                # Plot MR-PC predictions with markers at fusion points
                fused_mask = np.array([m == "fused" for m in model_used])

                # MR-PC line
                ax.plot(time_steps, mrpc_values, "g-", linewidth=2.5, label="MR-PC (ours)", alpha=0.9, zorder=4)

                # Highlight fusion points
                if np.any(fused_mask):
                    ax.scatter(
                        time_steps[fused_mask],
                        mrpc_values[fused_mask],
                        c="red",
                        marker="*",
                        s=200,
                        label="Fusion points",
                        alpha=0.9,
                        edgecolors="darkred",
                        linewidths=1.5,
                        zorder=6,
                    )

                # Mark warm-up phase
                warmup_end = 20
                if warmup_end < num_steps:
                    ax.axvline(
                        x=warmup_end,
                        color="purple",
                        linestyle="-.",
                        linewidth=2,
                        alpha=0.5,
                        label="Warm-up end",
                        zorder=1,
                    )

                ax.set_xlabel("Time Step (t)", fontsize=13)
                ax.set_ylabel(f"{field_name.upper()} Value", fontsize=13)
                ax.set_title(
                    f"{field_name.upper()} Evolution at Point ({point_h}, {point_w})", fontsize=15, fontweight="bold"
                )
                ax.legend(fontsize=11, loc="best", framealpha=0.9)
                ax.grid(True, alpha=0.3)

            # Add position indicator in title
            position_label = f"Grid Position: Row {h_idx + 1}/3, Col {w_idx + 1}/3"
            plt.suptitle(
                f"MR-PC vs Small-Scale - Temporal Evolution - Sample {sample_idx}\n{position_label}\n"
                "(Large-scale omitted for clarity)",
                fontsize=17,
                fontweight="bold",
            )
            plt.tight_layout()

            # Save figure with _no_large suffix
            output_path = self.output_dir / f"temporal_evolution_sample_{sample_idx}_point_{point_idx + 1}_no_large.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"    Saved: {output_path}")

            if self.wandb_run:
                self.wandb_run.log(
                    {
                        f"temporal_evolution_sample_{sample_idx}_point_{point_idx + 1}_no_large": wandb.Image(
                            str(output_path)
                        )
                    }
                )

            plt.close()

        print(f"  Completed temporal evolution plots for all {len(points)} points (with and without large-scale)")

    def _create_error_analysis(
        self,
        pred_seq_mrpc,
        pred_seq_small,
        pred_seq_large,
        ground_truth_frames,
        model_used,
        fusion_info,
        field_names,
        sample_idx,
    ):
        """Create error analysis plots comparing MR-PC with baselines."""
        print("\nCreating error analysis with baselines...")

        num_steps = min(len(pred_seq_mrpc), len(ground_truth_frames))
        time_steps = np.arange(1, num_steps + 1)

        # Calculate errors for each field and method
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # MSE over time - comparing all methods
        ax = axes[0, 0]
        for field_idx, field_name in enumerate(field_names):
            # MR-PC
            mse_mrpc = []
            mse_small = []
            mse_large = []

            for t in range(num_steps):
                gt = ground_truth_frames[t][field_idx]

                mse_mrpc.append(np.mean((pred_seq_mrpc[t, field_idx] - gt) ** 2))
                mse_small.append(np.mean((pred_seq_small[t, field_idx] - gt) ** 2))
                mse_large.append(np.mean((pred_seq_large[t, field_idx] - gt) ** 2))

            # Plot with different line styles
            ax.plot(time_steps, mse_mrpc, "-", linewidth=2.5, label=f"{field_name.upper()} (MR-PC)", alpha=0.8)
            ax.plot(time_steps, mse_small, "--", linewidth=1.5, label=f"{field_name.upper()} (Small)", alpha=0.6)
            ax.plot(time_steps, mse_large, ":", linewidth=2, label=f"{field_name.upper()} (Large)", alpha=0.6)

        # Mark warm-up phase
        warmup_end = 25  # Warm-up now ends at t=30 (25 predictions from t=6 to t=30)
        if warmup_end < num_steps:
            ax.axvline(x=warmup_end, color="purple", linestyle="-.", linewidth=2, alpha=0.5, label="Warm-up end")

        ax.set_xlabel("Time Step (t)", fontsize=12)
        ax.set_ylabel("MSE", fontsize=12)
        ax.set_title("Mean Squared Error Over Time", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, ncol=3, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Average MSE comparison across all fields
        ax = axes[0, 1]
        avg_mse_mrpc = []
        avg_mse_small = []
        avg_mse_large = []

        for t in range(num_steps):
            gt_all = [ground_truth_frames[t][i] for i in range(len(field_names))]
            pred_mrpc_all = [pred_seq_mrpc[t, i] for i in range(len(field_names))]
            pred_small_all = [pred_seq_small[t, i] for i in range(len(field_names))]
            pred_large_all = [pred_seq_large[t, i] for i in range(len(field_names))]

            mse_mrpc_t = np.mean([np.mean((pred_mrpc_all[i] - gt_all[i]) ** 2) for i in range(len(field_names))])
            mse_small_t = np.mean([np.mean((pred_small_all[i] - gt_all[i]) ** 2) for i in range(len(field_names))])
            mse_large_t = np.mean([np.mean((pred_large_all[i] - gt_all[i]) ** 2) for i in range(len(field_names))])

            avg_mse_mrpc.append(mse_mrpc_t)
            avg_mse_small.append(mse_small_t)
            avg_mse_large.append(mse_large_t)

        ax.plot(time_steps, avg_mse_mrpc, "g-", linewidth=3, label="MR-PC (ours)", alpha=0.8)
        ax.plot(time_steps, avg_mse_small, "b--", linewidth=2, label="Pure Small-scale", alpha=0.6)
        ax.plot(time_steps, avg_mse_large, "orange", linestyle=":", linewidth=2.5, label="Pure Large-scale", alpha=0.6)

        if warmup_end < num_steps:
            ax.axvline(x=warmup_end, color="purple", linestyle="-.", linewidth=2, alpha=0.5)

        ax.set_xlabel("Time Step (t)", fontsize=12)
        ax.set_ylabel("Average MSE", fontsize=12)
        ax.set_title("Average MSE Across All Fields", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # RMS Relative Error comparison
        ax = axes[1, 0]
        for field_idx, field_name in enumerate(field_names):
            rms_rel_mrpc = []
            rms_rel_small = []
            rms_rel_large = []

            for t in range(num_steps):
                gt = ground_truth_frames[t][field_idx]
                gt_rms = np.sqrt(np.mean(gt**2))

                mse_mrpc = np.mean((pred_seq_mrpc[t, field_idx] - gt) ** 2)
                mse_small = np.mean((pred_seq_small[t, field_idx] - gt) ** 2)
                mse_large = np.mean((pred_seq_large[t, field_idx] - gt) ** 2)

                rms_rel_mrpc.append(np.sqrt(mse_mrpc) / (gt_rms + 1e-8))
                rms_rel_small.append(np.sqrt(mse_small) / (gt_rms + 1e-8))
                rms_rel_large.append(np.sqrt(mse_large) / (gt_rms + 1e-8))

            ax.plot(time_steps, rms_rel_mrpc, "-", linewidth=2.5, label=f"{field_name.upper()} (MR-PC)", alpha=0.8)
            ax.plot(time_steps, rms_rel_small, "--", linewidth=1.5, label=f"{field_name.upper()} (Small)", alpha=0.6)
            ax.plot(time_steps, rms_rel_large, ":", linewidth=2, label=f"{field_name.upper()} (Large)", alpha=0.6)

        if warmup_end < num_steps:
            ax.axvline(x=warmup_end, color="purple", linestyle="-.", linewidth=2, alpha=0.5)

        ax.set_xlabel("Time Step (t)", fontsize=12)
        ax.set_ylabel("RMS Relative Error", fontsize=12)
        ax.set_title("RMS Relative Error Over Time", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, ncol=3, loc="best")
        ax.grid(True, alpha=0.3)

        # Cumulative MSE comparison
        ax = axes[1, 1]
        cumsum_mse_mrpc = np.cumsum(avg_mse_mrpc)
        cumsum_mse_small = np.cumsum(avg_mse_small)
        cumsum_mse_large = np.cumsum(avg_mse_large)

        ax.plot(time_steps, cumsum_mse_mrpc, "g-", linewidth=3, label="MR-PC (ours)", alpha=0.8)
        ax.plot(time_steps, cumsum_mse_small, "b--", linewidth=2, label="Pure Small-scale", alpha=0.6)
        ax.plot(
            time_steps, cumsum_mse_large, "orange", linestyle=":", linewidth=2.5, label="Pure Large-scale", alpha=0.6
        )

        if warmup_end < num_steps:
            ax.axvline(x=warmup_end, color="purple", linestyle="-.", linewidth=2, alpha=0.5, label="Warm-up end")

        ax.set_xlabel("Time Step (t)", fontsize=12)
        ax.set_ylabel("Cumulative MSE", fontsize=12)
        ax.set_title("Cumulative MSE Over Time", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

        # Add improvement statistics
        final_improvement_vs_small = (cumsum_mse_small[-1] - cumsum_mse_mrpc[-1]) / cumsum_mse_small[-1] * 100
        final_improvement_vs_large = (cumsum_mse_large[-1] - cumsum_mse_mrpc[-1]) / cumsum_mse_large[-1] * 100

        textstr = "MR-PC Improvement:\n"
        textstr += f"vs Small: {final_improvement_vs_small:+.1f}%\n"
        textstr += f"vs Large: {final_improvement_vs_large:+.1f}%"

        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )

        plt.suptitle(f"MR-PC vs Baselines - Error Analysis - Sample {sample_idx}", fontsize=17, fontweight="bold")
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"error_analysis_sample_{sample_idx}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

        if self.wandb_run:
            self.wandb_run.log({f"error_analysis_sample_{sample_idx}": wandb.Image(str(output_path))})

        plt.close()

    def _create_spatial_comparison(
        self,
        pred_seq_mrpc,
        pred_seq_small,
        pred_seq_large,
        ground_truth_frames,
        model_used,
        field_names,
        sample_idx,
        num_predictions,
    ):
        """Create spatial comparison visualizations comparing MR-PC with baselines."""
        print("\nCreating spatial comparison with baselines...")

        # Select time steps to visualize (fewer steps due to more rows)
        display_steps = min(10, num_predictions)
        step_indices = np.linspace(0, len(pred_seq_mrpc) - 1, display_steps, dtype=int)

        for field_idx, field_name in enumerate(field_names):
            # Create figure: 5 rows (GT, MR-PC, Small, Large, Error) × timesteps
            fig, axes = plt.subplots(5, display_steps, figsize=(2.5 * display_steps, 12))
            if display_steps == 1:
                axes = axes.reshape(5, 1)

            # Calculate colorbar range
            all_data = []
            for t_idx in step_indices:
                if t_idx < len(ground_truth_frames):
                    all_data.append(ground_truth_frames[t_idx][field_idx])
                all_data.append(pred_seq_mrpc[t_idx][field_idx])
                all_data.append(pred_seq_small[t_idx][field_idx])
                all_data.append(pred_seq_large[t_idx][field_idx])

            cmap = "RdBu_r"
            vmax = max([abs(data.min()) for data in all_data] + [abs(data.max()) for data in all_data])
            vmin = -vmax

            for col, t_idx in enumerate(step_indices):
                t = t_idx + 1  # 1-indexed for display

                # Ground truth
                if t_idx < len(ground_truth_frames):
                    gt_data = ground_truth_frames[t_idx][field_idx]
                    im0 = axes[0, col].imshow(gt_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                    axes[0, col].set_title(f"t={t}", fontsize=9, fontweight="bold")
                else:
                    axes[0, col].axis("off")

                # MR-PC prediction
                pred_mrpc = pred_seq_mrpc[t_idx][field_idx]
                im1 = axes[1, col].imshow(pred_mrpc, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                model_type = model_used[t_idx]
                if model_type == "fused":
                    for spine in axes[1, col].spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(3)

                # Small-scale prediction
                pred_small = pred_seq_small[t_idx][field_idx]
                im2 = axes[2, col].imshow(pred_small, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

                # Large-scale prediction
                pred_large = pred_seq_large[t_idx][field_idx]
                im3 = axes[3, col].imshow(pred_large, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

                # Error (MR-PC vs GT)
                if t_idx < len(ground_truth_frames):
                    error = np.abs(pred_mrpc - gt_data)
                    im4 = axes[4, col].imshow(error, cmap="Reds", origin="lower")
                    mae = np.mean(error)
                    axes[4, col].set_title(f"MAE:{mae:.3f}", fontsize=8)
                else:
                    axes[4, col].axis("off")

                # Remove ticks
                for row in range(5):
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])

                # Add colorbar for first column
                if col == 0:
                    if t_idx < len(ground_truth_frames):
                        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
                    plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
                    plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)
                    plt.colorbar(im3, ax=axes[3, col], fraction=0.046, pad=0.04)
                    if t_idx < len(ground_truth_frames):
                        plt.colorbar(im4, ax=axes[4, col], fraction=0.046, pad=0.04)

            # Set row labels
            axes[0, 0].set_ylabel("Ground Truth", fontsize=11, fontweight="bold")
            axes[1, 0].set_ylabel("MR-PC", fontsize=11, fontweight="bold")
            axes[2, 0].set_ylabel("Pure Small", fontsize=11)
            axes[3, 0].set_ylabel("Pure Large", fontsize=11)
            axes[4, 0].set_ylabel("Error (MR-PC)", fontsize=11)

            plt.suptitle(
                f"Spatial Comparison: {field_name.upper()} - Sample {sample_idx}\n(Red border = fusion point)",
                fontsize=13,
                fontweight="bold",
            )
            plt.tight_layout()

            # Save figure
            output_path = self.output_dir / f"spatial_comparison_{field_name}_sample_{sample_idx}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_path}")

            if self.wandb_run:
                self.wandb_run.log(
                    {f"spatial_comparison_{field_name}_sample_{sample_idx}": wandb.Image(str(output_path))}
                )

            plt.close()

    def _create_fusion_analysis(self, pred_seq, fusion_info, field_names, sample_idx):
        """Create detailed fusion analysis visualization for multiple points."""
        print("\nCreating fusion analysis for 9 points...")

        if not fusion_info:
            print("No fusion events to analyze")
            return

        # Fixed 9 monitoring points (same as evaluation_1plane_new.py)
        # Adjusted monitoring points for 256x256 domain (indices 0-255)
        fixed_positions = [
            (80, 80),  # Bottom-left region
            (80, 128),  # Bottom-center
            (80, 200),  # Bottom-right
            (128, 80),  # Center-left
            (128, 128),  # Center-center
            (128, 200),  # Center-right
            (200, 80),  # Top-left
            (200, 128),  # Top-center
            (200, 200),  # Top-right
        ]

        # Create points list with grid indices for labeling
        points = []
        for idx, (z_pos, x_pos) in enumerate(fixed_positions):
            h_idx = idx // 3  # Row index (0, 1, 2)
            w_idx = idx % 3  # Column index (0, 1, 2)
            points.append((z_pos, x_pos, h_idx, w_idx))

        num_fusions = len(fusion_info)
        fusion_times = [f["time"] for f in fusion_info]
        fusion_weights = [f["fusion_weight"] for f in fusion_info]

        # Create separate figure for each point
        for point_idx, (point_h, point_w, h_idx, w_idx) in enumerate(points):
            print(f"  Creating fusion analysis for point {point_idx + 1}/{len(points)} at ({point_h}, {point_w})")

            # Create figure with multiple subplots
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

            # Plot 1: Fusion weights and times
            ax1 = fig.add_subplot(gs[0, :])
            ax1.scatter(fusion_times, fusion_weights, c="red", s=100, alpha=0.7, edgecolors="darkred", linewidths=2)
            ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Equal weight")
            ax1.set_xlabel("Time Step (t)", fontsize=12)
            ax1.set_ylabel("Fusion Weight (α)", fontsize=12)
            ax1.set_title(f"Fusion Events (Total: {num_fusions})", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.legend()

            # Plot 2-4: Field comparison at fusion points (small vs large vs fused)
            for field_idx, field_name in enumerate(field_names):
                ax = fig.add_subplot(gs[1 + field_idx // 2, field_idx % 2])

                small_values = []
                large_values = []
                fused_values = []

                for _fusion_idx, f_info in enumerate(fusion_info):
                    small_val = f_info["small_pred"][field_idx, point_h, point_w].item()
                    large_val = f_info["large_pred"][field_idx, point_h, point_w].item()
                    fused_val = f_info["fused"][field_idx, point_h, point_w].item()

                    small_values.append(small_val)
                    large_values.append(large_val)
                    fused_values.append(fused_val)

                x = np.arange(len(fusion_times))
                width = 0.25

                ax.bar(x - width, small_values, width, label="Small-scale", color="blue", alpha=0.7)
                ax.bar(x, large_values, width, label="Large-scale", color="orange", alpha=0.7)
                ax.bar(x + width, fused_values, width, label="Fused", color="red", alpha=0.7)

                ax.set_xlabel("Fusion Event Index", fontsize=11)
                ax.set_ylabel(f"{field_name.upper()} Value", fontsize=11)
                ax.set_title(f"{field_name.upper()} at Point ({point_h}, {point_w})", fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{i + 1}" for i in range(len(fusion_times))], fontsize=9)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis="y")

            # Add position indicator
            position_label = f"Grid Position: Row {h_idx + 1}/3, Col {w_idx + 1}/3"
            plt.suptitle(f"MR-PC Fusion Analysis - Sample {sample_idx}\n{position_label}", fontsize=16)

            # Save figure
            output_path = self.output_dir / f"fusion_analysis_sample_{sample_idx}_point_{point_idx + 1}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"    Saved: {output_path}")

            if self.wandb_run:
                self.wandb_run.log(
                    {f"fusion_analysis_sample_{sample_idx}_point_{point_idx + 1}": wandb.Image(str(output_path))}
                )

            plt.close()

        print(f"  Completed fusion analysis for all {len(points)} points")

    def _compute_energy_spectrum(self, frames, field_names):
        """
        Compute time-averaged 1D energy spectra for velocity fields.

        Args:
            frames: Array of shape (T, C, H, W) containing velocity fields
            field_names: List of field names ["u", "v", "w"]

        Returns:
            Dictionary containing spectrum data for each field
        """
        print("\nComputing energy spectra...")

        T, C, H, W = frames.shape

        # Create wavenumber arrays (normalized to grid spacing = 1.0)
        kx = np.fft.fftfreq(W, d=1.0)  # Streamwise wavenumber
        kz = np.fft.fftfreq(H, d=1.0)  # Spanwise wavenumber

        # Positive wavenumbers for plotting
        kx_pos = kx[kx > 0]
        kz_pos = kz[kz > 0]

        spectra_results = {"fields": {}}

        for field_idx, field_name in enumerate(field_names):
            print(f"  Processing {field_name} spectrum...")

            # Extract channel data for this field
            field_data = frames[:, field_idx, :, :]  # Shape: (T, H, W)

            # Time-averaged energy spectrum
            spectrum_2d_sum = np.zeros((H, W))

            for t in range(T):
                # Remove mean before FFT
                data_slice = field_data[t] - np.mean(field_data[t])

                # Compute 2D FFT with normalization
                fft_2d = np.fft.fft2(data_slice) / (H * W)

                # Compute energy spectrum: E(kx, kz) = 0.5 * |q_hat|^2
                spectrum_2d = 0.5 * np.abs(fft_2d) ** 2
                spectrum_2d_sum += spectrum_2d

            # Time average
            spectrum_2d_avg = spectrum_2d_sum / T

            # Compute 1D spectra by integration
            spectrum_kx = np.sum(spectrum_2d_avg, axis=0)  # Sum over kz
            spectrum_kz = np.sum(spectrum_2d_avg, axis=1)  # Sum over kx

            # Store results
            spectra_results["fields"][field_name] = {
                "spectrum_2d": spectrum_2d_avg,
                "spectrum_kx": spectrum_kx,
                "spectrum_kz": spectrum_kz,
                "kx": kx,
                "kz": kz,
                "kx_pos": kx_pos,
                "kz_pos": kz_pos,
            }

        print("Energy spectra computation completed.")
        return spectra_results

    def _create_energy_spectrum_comparison(
        self,
        pred_seq_mrpc,
        pred_seq_small,
        ground_truth_frames,
        field_names,
        sample_idx,
    ):
        """
        Create energy spectrum comparison plots between MR-PC and small-scale predictions.

        Args:
            pred_seq_mrpc: MR-PC predictions (T, C, H, W)
            pred_seq_small: Small-scale predictions (T, C, H, W)
            ground_truth_frames: List of ground truth frames
            field_names: List of field names
            sample_idx: Sample index
        """
        print("\nCreating energy spectrum comparison...")

        # Compute spectra for MR-PC predictions
        spectra_mrpc = self._compute_energy_spectrum(pred_seq_mrpc, field_names)

        # Compute spectra for small-scale predictions
        spectra_small = self._compute_energy_spectrum(pred_seq_small, field_names)

        # Compute spectra for ground truth if available
        if len(ground_truth_frames) > 0:
            # Stack ground truth frames
            num_gt = min(len(ground_truth_frames), len(pred_seq_mrpc))
            gt_array = np.stack([ground_truth_frames[i] for i in range(num_gt)], axis=0)
            spectra_gt = self._compute_energy_spectrum(gt_array, field_names)
            has_gt = True
        else:
            has_gt = False

        # Create comparison plots for each field
        for field_name in field_names:
            print(f"  Creating spectrum comparison plot for {field_name}...")

            # Extract spectrum data
            mrpc_data = spectra_mrpc["fields"][field_name]
            small_data = spectra_small["fields"][field_name]

            kx_pos = mrpc_data["kx_pos"]
            kz_pos = mrpc_data["kz_pos"]

            # Get positive wavenumber masks
            kx = mrpc_data["kx"]
            kz = mrpc_data["kz"]
            kx_pos_mask = kx > 0
            kz_pos_mask = kz > 0

            spectrum_kx_mrpc = mrpc_data["spectrum_kx"][kx_pos_mask]
            spectrum_kz_mrpc = mrpc_data["spectrum_kz"][kz_pos_mask]

            spectrum_kx_small = small_data["spectrum_kx"][kx_pos_mask]
            spectrum_kz_small = small_data["spectrum_kz"][kz_pos_mask]

            if has_gt:
                gt_data = spectra_gt["fields"][field_name]
                spectrum_kx_gt = gt_data["spectrum_kx"][kx_pos_mask]
                spectrum_kz_gt = gt_data["spectrum_kz"][kz_pos_mask]

            # Create figure with 2 subplots (1D streamwise and spanwise spectra)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Subplot 1: Streamwise spectrum E(kx)
            ax = axes[0]
            if has_gt:
                ax.loglog(kx_pos, spectrum_kx_gt, "k-", linewidth=3, label="Ground Truth", alpha=0.8, zorder=5)
            ax.loglog(kx_pos, spectrum_kx_mrpc, "g-", linewidth=2.5, label="MR-PC (ours)", alpha=0.8, zorder=4)
            ax.loglog(kx_pos, spectrum_kx_small, "b--", linewidth=2, label="Pure Small-scale", alpha=0.7, zorder=3)

            # Add reference lines (Kolmogorov -5/3 slope)
            if len(kx_pos) > 10:
                k_ref = kx_pos[len(kx_pos) // 3 : 2 * len(kx_pos) // 3]
                E_ref = spectrum_kx_mrpc[len(kx_pos) // 3] * (k_ref / kx_pos[len(kx_pos) // 3]) ** (-5 / 3)
                ax.loglog(k_ref, E_ref, "gray", linestyle=":", linewidth=1.5, label="k⁻⁵/³", alpha=0.5, zorder=1)

            ax.set_xlabel("Streamwise Wavenumber kx", fontsize=13)
            ax.set_ylabel("Energy Spectrum E(kx)", fontsize=13)
            ax.set_title(f"{field_name.upper()} - Streamwise Spectrum", fontsize=14, fontweight="bold")
            ax.legend(fontsize=11, loc="best", framealpha=0.9)
            ax.grid(True, alpha=0.3, which="both")

            # Subplot 2: Spanwise spectrum E(kz)
            ax = axes[1]
            if has_gt:
                ax.loglog(kz_pos, spectrum_kz_gt, "k-", linewidth=3, label="Ground Truth", alpha=0.8, zorder=5)
            ax.loglog(kz_pos, spectrum_kz_mrpc, "g-", linewidth=2.5, label="MR-PC (ours)", alpha=0.8, zorder=4)
            ax.loglog(kz_pos, spectrum_kz_small, "b--", linewidth=2, label="Pure Small-scale", alpha=0.7, zorder=3)

            # Add reference lines (Kolmogorov -5/3 slope)
            if len(kz_pos) > 10:
                k_ref = kz_pos[len(kz_pos) // 3 : 2 * len(kz_pos) // 3]
                E_ref = spectrum_kz_mrpc[len(kz_pos) // 3] * (k_ref / kz_pos[len(kz_pos) // 3]) ** (-5 / 3)
                ax.loglog(k_ref, E_ref, "gray", linestyle=":", linewidth=1.5, label="k⁻⁵/³", alpha=0.5, zorder=1)

            ax.set_xlabel("Spanwise Wavenumber kz", fontsize=13)
            ax.set_ylabel("Energy Spectrum E(kz)", fontsize=13)
            ax.set_title(f"{field_name.upper()} - Spanwise Spectrum", fontsize=14, fontweight="bold")
            ax.legend(fontsize=11, loc="best", framealpha=0.9)
            ax.grid(True, alpha=0.3, which="both")

            # Overall title
            fig.suptitle(
                f"MR-PC vs Small-Scale - Energy Spectrum Comparison - {field_name.upper()} - Sample {sample_idx}",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()

            # Save figure
            output_path = self.output_dir / f"energy_spectrum_comparison_{field_name}_sample_{sample_idx}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"    Saved: {output_path}")

            if self.wandb_run:
                self.wandb_run.log(
                    {f"energy_spectrum_comparison_{field_name}_sample_{sample_idx}": wandb.Image(str(output_path))}
                )

            plt.close()

            # Save numerical data
            np.save(self.output_dir / f"spectrum_mrpc_{field_name}_kx_sample_{sample_idx}.npy", spectrum_kx_mrpc)
            np.save(self.output_dir / f"spectrum_mrpc_{field_name}_kz_sample_{sample_idx}.npy", spectrum_kz_mrpc)
            np.save(self.output_dir / f"spectrum_small_{field_name}_kx_sample_{sample_idx}.npy", spectrum_kx_small)
            np.save(self.output_dir / f"spectrum_small_{field_name}_kz_sample_{sample_idx}.npy", spectrum_kz_small)
            if has_gt:
                np.save(self.output_dir / f"spectrum_gt_{field_name}_kx_sample_{sample_idx}.npy", spectrum_kx_gt)
                np.save(self.output_dir / f"spectrum_gt_{field_name}_kz_sample_{sample_idx}.npy", spectrum_kz_gt)

        print("Energy spectrum comparison plots completed.")

    def generate_prediction_videos(
        self,
        sample_idx: int = 0,
        num_predictions: int = 100,
        fusion_weight: float = 0.5,
        fps: int = 10,
        fusion_interval: int = 10,
        first_fusion_point: int = 31,
    ):
        """Generate videos comparing MR-PC fusion predictions with small-scale baseline.

        Creates three videos:
        1. MR-PC (fused) prediction showing all 3 channels (u, v, w)
        2. Small-scale baseline prediction showing all 3 channels (u, v, w)
        3. Comparison video with GT, MR-PC, and Small-scale side-by-side

        Args:
            sample_idx: Sample index to evaluate
            num_predictions: Number of future steps to predict
            fusion_weight: Fusion weight for MR-PC
            fps: Frames per second for output videos
            fusion_interval: Fusion interval in timesteps
        """
        print(f"\n{'=' * 60}")
        print("Generating prediction videos")
        print(f"{'=' * 60}")

        # Get initial sample from small-scale dataset
        sample = self.small_scale_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)  # (1, T, C, H, W)

        # Perform predictions
        print("\n1. Running MR-PC prediction...")
        pred_seq_mrpc, model_used, fusion_info = self.cross_scale_prediction(
            input_seq, num_predictions, fusion_weight, sample_idx, fusion_interval, first_fusion_point
        )

        print("\n2. Running small-scale baseline prediction...")
        pred_seq_small = self.pure_small_scale_prediction(input_seq, num_predictions)

        # Denormalize predictions
        pred_seq_mrpc_denorm = (
            self.small_scale_dataset.denormalize(torch.from_numpy(pred_seq_mrpc).unsqueeze(0)).cpu().numpy()[0]
        )
        pred_seq_small_denorm = (
            self.small_scale_dataset.denormalize(torch.from_numpy(pred_seq_small).unsqueeze(0)).cpu().numpy()[0]
        )

        # Collect ground truth
        print("\n3. Collecting ground truth for comparison...")
        gt_frames = []
        for t in range(num_predictions):
            if sample_idx + t + 1 < len(self.small_scale_dataset):
                gt_sample = self.small_scale_dataset[sample_idx + t + 1]
                gt_frame = gt_sample["data"]["input_seq"][0, -1].cpu().numpy()  # Last frame of input
                gt_frames.append(gt_frame)
            else:
                gt_frames.append(np.zeros_like(pred_seq_mrpc_denorm[0]))

        gt_seq_denorm = (
            self.small_scale_dataset.denormalize(torch.from_numpy(np.stack(gt_frames)).unsqueeze(0)).cpu().numpy()[0]
        )

        field_names = self.small_scale_dataset.field_names
        T, C, H, W = pred_seq_mrpc_denorm.shape

        print("\n4. Preparing video generation...")
        print(f"   Video shape: {T} frames, {C} channels, {H}x{W} spatial resolution")
        print(f"   Fields: {field_names}")

        # Create output directory
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(exist_ok=True)

        # ========== Video 1: MR-PC Prediction ==========
        print("\n5. Creating MR-PC prediction video...")
        mrpc_video_path = video_dir / f"mrpc_prediction_sample_{sample_idx}.mp4"

        # Calculate color range based on FIRST FRAME of MR-PC prediction (matches evaluation_1plane.py)
        vmin_mrpc = []
        vmax_mrpc = []
        for ch in range(C):
            first_frame = pred_seq_mrpc_denorm[0, ch]
            vmax_val = max(abs(first_frame.min()), abs(first_frame.max()))
            vmin_mrpc.append(-vmax_val)
            vmax_mrpc.append(vmax_val)
        print(f"   MR-PC color ranges: {list(zip(field_names, vmin_mrpc, vmax_mrpc, strict=False))}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ims = []
        titles = []

        for ch, (field_name, ax) in enumerate(zip(field_names, axes, strict=False)):
            im = ax.imshow(
                pred_seq_mrpc_denorm[0, ch],
                cmap="RdBu_r",
                vmin=vmin_mrpc[ch],
                vmax=vmax_mrpc[ch],
                origin="lower",
                animated=True,
            )
            ims.append(im)
            title = ax.set_title(f"{field_name.upper()}, t=0")
            titles.append(title)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        def animate_mrpc(frame):
            for ch in range(C):
                ims[ch].set_array(pred_seq_mrpc_denorm[frame, ch])
                # Update title with current timestep and fusion indicator
                is_fusion = model_used[frame] == "fused" if frame < len(model_used) else False
                fusion_label = " [FUSION]" if is_fusion else ""
                titles[ch].set_text(f"{field_names[ch].upper()}, t={frame}{fusion_label}")
            return ims + titles

        anim = animation.FuncAnimation(fig, animate_mrpc, frames=T, interval=1000 / fps, blit=True, repeat=True)

        try:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="MRPC"), bitrate=1800)
            anim.save(mrpc_video_path, writer=writer)
            print(f"   Saved: {mrpc_video_path}")
        except Exception as e:
            print(f"   Warning: Could not save as MP4: {e}")
            print("   Trying GIF format...")
            mrpc_gif_path = video_dir / f"mrpc_prediction_sample_{sample_idx}.gif"
            anim.save(mrpc_gif_path, writer="pillow", fps=fps)
            print(f"   Saved as GIF: {mrpc_gif_path}")

        plt.close(fig)

        # ========== Video 2: Small-scale Baseline ==========
        print("\n6. Creating small-scale baseline video...")
        small_video_path = video_dir / f"small_prediction_sample_{sample_idx}.mp4"

        # Calculate color range based on FIRST FRAME of Small-scale prediction
        vmin_small = []
        vmax_small = []
        for ch in range(C):
            first_frame = pred_seq_small_denorm[0, ch]
            vmax_val = max(abs(first_frame.min()), abs(first_frame.max()))
            vmin_small.append(-vmax_val)
            vmax_small.append(vmax_val)
        print(f"   Small-scale color ranges: {list(zip(field_names, vmin_small, vmax_small, strict=False))}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ims = []
        titles = []

        for ch, (field_name, ax) in enumerate(zip(field_names, axes, strict=False)):
            im = ax.imshow(
                pred_seq_small_denorm[0, ch],
                cmap="RdBu_r",
                vmin=vmin_small[ch],
                vmax=vmax_small[ch],
                origin="lower",
                animated=True,
            )
            ims.append(im)
            title = ax.set_title(f"{field_name.upper()}, t=0")
            titles.append(title)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        def animate_small(frame):
            for ch in range(C):
                ims[ch].set_array(pred_seq_small_denorm[frame, ch])
                titles[ch].set_text(f"{field_names[ch].upper()}, t={frame}")
            return ims + titles

        anim = animation.FuncAnimation(fig, animate_small, frames=T, interval=1000 / fps, blit=True, repeat=True)

        try:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="SmallScale"), bitrate=1800)
            anim.save(small_video_path, writer=writer)
            print(f"   Saved: {small_video_path}")
        except Exception as e:
            print(f"   Warning: Could not save as MP4: {e}")
            print("   Trying GIF format...")
            small_gif_path = video_dir / f"small_prediction_sample_{sample_idx}.gif"
            anim.save(small_gif_path, writer="pillow", fps=fps)
            print(f"   Saved as GIF: {small_gif_path}")

        plt.close(fig)

        # ========== Video 3: Comparison (GT + MR-PC + Small) ==========
        print("\n7. Creating comparison video...")
        comparison_video_path = video_dir / f"comparison_sample_{sample_idx}.mp4"

        # Calculate color range based on FIRST FRAME of Ground Truth (for fair comparison)
        vmin_gt = []
        vmax_gt = []
        for ch in range(C):
            first_frame = gt_seq_denorm[0, ch]
            vmax_val = max(abs(first_frame.min()), abs(first_frame.max()))
            vmin_gt.append(-vmax_val)
            vmax_gt.append(vmax_val)
        print(f"   Comparison color ranges (from GT): {list(zip(field_names, vmin_gt, vmax_gt, strict=False))}")

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        ims = []
        titles = []

        for ch, field_name in enumerate(field_names):
            # Row 0: Ground Truth
            im_gt = axes[0, ch].imshow(
                gt_seq_denorm[0, ch],
                cmap="RdBu_r",
                vmin=vmin_gt[ch],
                vmax=vmax_gt[ch],
                origin="lower",
                animated=True,
            )
            title_gt = axes[0, ch].set_title(f"GT {field_name.upper()}, t=0")
            axes[0, ch].set_xlabel("x")
            axes[0, ch].set_ylabel("z")
            plt.colorbar(im_gt, ax=axes[0, ch], fraction=0.046, pad=0.04)
            ims.append(im_gt)
            titles.append(title_gt)

            # Row 1: MR-PC
            im_mrpc = axes[1, ch].imshow(
                pred_seq_mrpc_denorm[0, ch],
                cmap="RdBu_r",
                vmin=vmin_gt[ch],
                vmax=vmax_gt[ch],
                origin="lower",
                animated=True,
            )
            title_mrpc = axes[1, ch].set_title(f"MR-PC {field_name.upper()}, t=0")
            axes[1, ch].set_xlabel("x")
            axes[1, ch].set_ylabel("z")
            plt.colorbar(im_mrpc, ax=axes[1, ch], fraction=0.046, pad=0.04)
            ims.append(im_mrpc)
            titles.append(title_mrpc)

            # Row 2: Small-scale
            im_small = axes[2, ch].imshow(
                pred_seq_small_denorm[0, ch],
                cmap="RdBu_r",
                vmin=vmin_gt[ch],
                vmax=vmax_gt[ch],
                origin="lower",
                animated=True,
            )
            title_small = axes[2, ch].set_title(f"Small {field_name.upper()}, t=0")
            axes[2, ch].set_xlabel("x")
            axes[2, ch].set_ylabel("z")
            plt.colorbar(im_small, ax=axes[2, ch], fraction=0.046, pad=0.04)
            ims.append(im_small)
            titles.append(title_small)

        def animate_comparison(frame):
            for ch in range(C):
                ims[ch * 3].set_array(gt_seq_denorm[frame, ch])
                ims[ch * 3 + 1].set_array(pred_seq_mrpc_denorm[frame, ch])
                ims[ch * 3 + 2].set_array(pred_seq_small_denorm[frame, ch])

                # Update titles with current timestep
                titles[ch * 3].set_text(f"GT {field_names[ch].upper()}, t={frame}")

                # Update MR-PC title with fusion indicator
                is_fusion = model_used[frame] == "fused" if frame < len(model_used) else False
                mrpc_title = f"MR-PC {field_names[ch].upper()}, t={frame}"
                if is_fusion:
                    mrpc_title += " [FUSION]"
                titles[ch * 3 + 1].set_text(mrpc_title)

                titles[ch * 3 + 2].set_text(f"Small {field_names[ch].upper()}, t={frame}")

            return ims + titles

        anim = animation.FuncAnimation(fig, animate_comparison, frames=T, interval=1000 / fps, blit=True, repeat=True)

        try:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="Comparison"), bitrate=1800)
            anim.save(comparison_video_path, writer=writer)
            print(f"   Saved: {comparison_video_path}")
        except Exception as e:
            print(f"   Warning: Could not save as MP4: {e}")
            print("   Trying GIF format...")
            comparison_gif_path = video_dir / f"comparison_sample_{sample_idx}.gif"
            anim.save(comparison_gif_path, writer="pillow", fps=fps)
            print(f"   Saved as GIF: {comparison_gif_path}")

        plt.close(fig)

        print(f"\n{'=' * 60}")
        print("Video generation complete!")
        print(f"{'=' * 60}")
        print(f"Videos saved to: {video_dir}")
        print(f"Video duration: {T / fps:.1f} seconds at {fps} fps")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-scale evaluation for 1-plane Flow Swin Transformer")
    parser.add_argument(
        "--small_scale_checkpoint",
        type=str,
        default="/home/sh/CB/icon-thewell-dev/logs/flow_lstm_1plane/runs/2026-01-12_20-37-20-581473/checkpoints/step_19800.ckpt",
        help="Path to small-scale model checkpoint (t spacing)",
    )
    parser.add_argument(
        "--large_scale_checkpoint",
        type=str,
        default="/home/sh/CB/icon-thewell-dev/logs/flow_swin_1plane/"
        "runs/2026-01-10_19-55-28-216423/checkpoints/step_53000.ckpt",
        help="Path to large-scale model checkpoint (10t spacing)",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to evaluate")
    parser.add_argument("--num_predictions", type=int, default=80, help="Number of future steps to predict")
    parser.add_argument(
        "--fusion_weight", type=float, default=0.95, help="Fusion weight α (0-1): x_fused = (1-α)*x_small + α*x_large"
    )
    parser.add_argument(
        "--generate_videos", action="store_true", help="Generate videos comparing MR-PC and small-scale predictions"
    )
    parser.add_argument("--video_fps", type=int, default=10, help="Frames per second for generated videos")
    parser.add_argument(
        "--fusion_interval",
        type=int,
        default=10,
        help="Fusion interval in timesteps (e.g., 1=fuse every t, 2=fuse every 2t, 10=fuse every 10t)",
    )
    parser.add_argument(
        "--first_fusion_point",
        type=int,
        default=31,
        help="Time step of the first fusion point (default: 31). Must be >= 11.",
    )

    args = parser.parse_args()

    # Validate first_fusion_point
    if args.first_fusion_point < 11:
        raise ValueError(f"first_fusion_point must be >= 11, got {args.first_fusion_point}")

    # Calculate minimum sample_idx requirement
    min_sample_idx_required = max(0, 1 - (args.first_fusion_point - 50))
    if args.sample_idx < min_sample_idx_required:
        print(f"WARNING: sample_idx ({args.sample_idx}) < required minimum ({min_sample_idx_required})")
        print(f"         for first_fusion_point={args.first_fusion_point}")
        print("         Historical frames may be incomplete.")

    # Model configurations
    small_scale_cfg = OmegaConf.create(
        {
            "_target_": "src.models.base.swin_transformer.SwinTransformerAuto",
            "input_shape": [256, 256],
            "sequence_length": 5,
            "prediction_horizon": 1,
            "num_channels": 3,  # u, v, w
            "patch_size": [4, 4],
            "embed_dim": 128,
            "depths": [2, 2, 4, 6, 4, 2, 2],
            "num_heads": 8,
            "window_size": [8, 8],
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "drop_rate": 0.1,
            "attn_drop_rate": 0.1,
            "drop_path_rate": 0.1,
            "use_patch_merging": True,
        }
    )

    large_scale_cfg = OmegaConf.create(
        {
            "_target_": "src.models.base.swin_transformer.SwinTransformerAuto",
            "input_shape": [256, 256],
            "sequence_length": 5,
            "prediction_horizon": 1,
            "num_channels": 3,  # u, v, w
            "patch_size": [4, 4],
            "embed_dim": 128,
            "depths": [2, 2, 4, 6, 4, 2, 2],
            "num_heads": 8,
            "window_size": [8, 8],
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "drop_rate": 0.1,
            "attn_drop_rate": 0.1,
            "drop_path_rate": 0.1,
            "use_patch_merging": True,
        }
    )

    # Data configuration
    data_config = {
        "data_dir": "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow/new",
        "input_length": 5,
        "field_names": ["u", "v", "w"],
        "file_pattern": "*u-v-w_scale2-3_ylayer2_ts*.h5",  # Explicitly specify yslice54 to avoid loading other y-slices
        "resolution_scale": (2, 3, 1),
        "y_slice": 54,
        "train_ratio": 0.8,
        "valid_ratio": 0.05,
        "test_ratio": 0.15,
        "enable_normalization": True,
        "norm_stats": "norm_stats_3ch_1plane_u-v-w_scale2-3_ylayer2.json",
    }

    # Create evaluator
    evaluator = CrossScaleEvaluator(
        small_scale_checkpoint=args.small_scale_checkpoint,
        large_scale_checkpoint=args.large_scale_checkpoint,
        small_scale_cfg=small_scale_cfg,
        large_scale_cfg=large_scale_cfg,
        data_config=data_config,
    )

    # Run evaluation
    evaluator.visualize_cross_scale_prediction(
        sample_idx=args.sample_idx,
        num_predictions=args.num_predictions,
        fusion_weight=args.fusion_weight,
        fusion_interval=args.fusion_interval,
        first_fusion_point=args.first_fusion_point,
    )

    # Generate videos if requested
    if args.generate_videos:
        evaluator.generate_prediction_videos(
            sample_idx=args.sample_idx,
            num_predictions=args.num_predictions,
            fusion_weight=args.fusion_weight,
            fps=args.video_fps,
            fusion_interval=args.fusion_interval,
            first_fusion_point=args.first_fusion_point,
        )

    print("\n" + "=" * 60)
    print("MR-PC evaluation complete!")
    print(f"Results saved to: {evaluator.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
