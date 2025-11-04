#!/usr/bin/env python3
"""
Cross-scale evaluation script for 1-plane Flow Swin Transformer.
Alternates between small-scale (t) and large-scale (5t) models to extend autoregressive predictions.

Prediction strategy:
- Small-scale model: predicts 4 consecutive steps (frames spaced by t)
- Large-scale model: predicts 1 step using every 5th frame (frames spaced by 5t)
- Alternates between the two models to achieve long-term predictions

Example prediction sequence:
Small-scale (t):  [1,2,3,4,5] → 6, [2,3,4,5,6] → 7, ..., [20,21,22,23,24] → 25
Large-scale (5t): [1,6,11,16,21] → 26
Small-scale (t):  [22,23,24,25,26] → 27, [23,24,25,26,27] → 28, ..., [25,26,27,28,29] → 30
Large-scale (5t): [6,11,16,21,26] → 31
...
"""

import os
import sys
import warnings
from pathlib import Path

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
            large_scale_checkpoint: Path to large-scale model checkpoint (5t spacing)
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
        print("Loading large-scale model (5t spacing)...")
        print("=" * 60)
        self.large_scale_model = self._load_model(large_scale_checkpoint, large_scale_cfg)

        # Setup datasets
        self.data_config = data_config
        self.small_scale_dataset = self._setup_dataset(time_stride=1)  # t spacing
        self.large_scale_dataset = self._setup_dataset(time_stride=5)  # 5t spacing

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
            module_cfg = OmegaConf.create({"model": model_cfg, "loss_fn": "mse"})
            model = FlowSwin2DLitModule(module_cfg)

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
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
            train_ratio=self.data_config.get("train_ratio", 0.7),
            valid_ratio=self.data_config.get("valid_ratio", 0.15),
            test_ratio=self.data_config.get("test_ratio", 0.15),
            split="test",
            time_stride=time_stride,
            enable_normalization=self.data_config.get("enable_normalization", True),
            norm_stats=self.data_config.get("norm_stats", None),
        )

        print(f"Dataset size: {len(dataset)}")
        return dataset

    def cross_scale_prediction(
        self,
        initial_frames: torch.Tensor,
        num_predictions: int = 100,
        fusion_weight: float = 0.5,
    ) -> tuple[np.ndarray, list[str], list[dict]]:
        """Perform MR-PC (Multi-Resolution Prediction-Correction) cross-scale prediction.

        Strategy:
        1. Warm-up phase: Use small-scale model for 20 steps to build anchor queue B
        2. Main loop: Every 5 small steps + 1 large-scale correction with fusion

        Args:
            initial_frames: Initial input sequence (B, T=5, C, H, W), frames at t=1,2,3,4,5
            num_predictions: Total number of future steps to predict
            fusion_weight: Weight for fusion, x_fused = (1-α)*x_small + α*x_large

        Returns:
            predictions: Array of all predicted frames (num_predictions, C, H, W)
            model_used: List indicating which model was used ('small', 'large', 'fused')
            fusion_info: List of dicts with fusion details for each fused frame
        """
        print(f"\n{'=' * 60}")
        print("Starting MR-PC cross-scale prediction")
        print(f"Total predictions: {num_predictions}")
        print(f"Fusion weight α: {fusion_weight}")
        print(f"{'=' * 60}")

        self.small_scale_model.eval()
        self.large_scale_model.eval()

        predictions = []
        model_used = []
        fusion_info = []

        # Initialize queues
        # S: small-step window (length=5, t-spacing), for small-scale model
        # B: anchor window (length=5, 5t-spacing), for large-scale model
        # all_frames: complete history of all frames

        # initial_frames: (1, 5, C, H, W) -> frames at t=1,2,3,4,5
        initial_list = [initial_frames[0, i].clone() for i in range(5)]
        all_frames = initial_list.copy()  # Complete history

        S = initial_list.copy()  # [x[1], x[2], x[3], x[4], x[5]]
        B = [initial_list[-1].clone()]  # [x[5]] - first anchor at 5t

        current_time = 5  # Current time in units of t

        print("\nInitialization:")
        print("  S (small window): frames at t=[1,2,3,4,5]")
        print("  B (anchor queue): frames at t=[5]")
        print(f"  Current time: {current_time}t")

        with torch.no_grad():
            # ========================================
            # Phase 1: Warm-up (20 small steps to build B)
            # ========================================
            print(f"\n{'=' * 60}")
            print("Phase 1: Warm-up - Building anchor queue B")
            print(f"{'=' * 60}")

            warmup_steps = 20
            for step in range(warmup_steps):
                # Small-scale prediction
                small_input = torch.stack(S, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                next_pred = self.small_scale_model(small_input)[0]  # (C, H, W)

                current_time += 1
                all_frames.append(next_pred.clone())
                predictions.append(next_pred.cpu())
                model_used.append("small")

                # Update S: slide window
                S = S[1:] + [next_pred]

                # Check if this is an anchor point (multiple of 5)
                if current_time % 5 == 0:
                    B.append(next_pred.clone())
                    print(f"  Step {step + 1}/{warmup_steps}: t={current_time}, Small-scale (★ Anchor added to B)")
                else:
                    print(f"  Step {step + 1}/{warmup_steps}: t={current_time}, Small-scale")

            print("\nWarm-up complete!")
            print(f"  B now contains {len(B)} anchors at t={[5 + i * 5 for i in range(len(B))]}")
            print(f"  Current time: {current_time}t")

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

                # Step 1: Small-scale predictions (5 steps)
                small_predictions = []
                for small_step in range(5):
                    if len(predictions) >= num_predictions:
                        break

                    small_input = torch.stack(S, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                    next_pred = self.small_scale_model(small_input)[0]  # (C, H, W)

                    current_time += 1
                    all_frames.append(next_pred.clone())
                    small_predictions.append(next_pred)

                    # For the first 4 steps, directly use small-scale prediction
                    if small_step < 4:
                        predictions.append(next_pred.cpu())
                        model_used.append("small")
                        S = S[1:] + [next_pred]
                        print(f"  Small step {small_step + 1}/5: t={current_time}, using small-scale result")
                    else:
                        # Last step: keep as candidate, will be fused with large-scale
                        print(f"  Small step {small_step + 1}/5: t={current_time}, candidate for fusion")

                if len(predictions) >= num_predictions:
                    break

                # Step 2: Large-scale prediction (jump from k to k+5)
                large_input = torch.stack(B, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                large_pred = self.large_scale_model(large_input)[0]  # (C, H, W)

                print(f"  Large-scale jump: t={current_time - 5} → t={current_time}")

                # Step 3: Fusion
                small_candidate = small_predictions[-1]  # x̂[k+5] from small-scale
                x_fused = (1 - fusion_weight) * small_candidate + fusion_weight * large_pred

                # Record fusion info
                fusion_info.append(
                    {
                        "time": current_time,
                        "cycle": cycle_count,
                        "fusion_weight": fusion_weight,
                        "small_pred": small_candidate.cpu(),
                        "large_pred": large_pred.cpu(),
                        "fused": x_fused.cpu(),
                    }
                )

                # Add fused result
                predictions.append(x_fused.cpu())
                model_used.append("fused")
                all_frames[-1] = x_fused.clone()  # Replace the last frame with fused version

                print(f"  Fusion: x[{current_time}] = {1 - fusion_weight:.2f}*small + {fusion_weight:.2f}*large")

                # Step 4: Update queues for next cycle
                S = S[1:] + [x_fused]  # Update S with fused anchor
                B = B[1:] + [x_fused]  # Update B with new anchor

                print("  Queues updated for next cycle")
                print(f"  S: last 5 frames ending at t={current_time}")
                print(f"  B: anchors at t={[current_time - 20 + i * 5 for i in range(5)]}")

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

    def pure_large_scale_prediction(self, initial_frames: torch.Tensor, num_predictions: int = 100) -> np.ndarray:
        """Pure large-scale autoregressive prediction (baseline)."""
        print(f"\nRunning pure large-scale baseline ({num_predictions} steps)...")

        self.large_scale_model.eval()
        predictions = []

        # For large-scale model, we need to first get frames at 5t spacing
        # Start with initial frames (assume they are at t=1,2,3,4,5)
        # We need frames at 5t intervals, so first build up history using small model

        with torch.no_grad():
            # Phase 1: Build initial large-scale sequence using small model
            S = [initial_frames[0, i].clone() for i in range(5)]
            all_frames = S.copy()

            # Predict up to t=25 to get 5 anchors at 5, 10, 15, 20, 25
            for _step in range(20):
                small_input = torch.stack(S, dim=0).unsqueeze(0)
                next_pred = self.small_scale_model(small_input)[0]
                all_frames.append(next_pred)
                S = S[1:] + [next_pred]

            # Extract anchors at 5t spacing: indices 4, 9, 14, 19, 24 (0-indexed)
            B = [all_frames[i].clone() for i in [4, 9, 14, 19, 24]]

            # Now use large-scale model for remaining predictions
            remaining = num_predictions
            while remaining > 0:
                large_input = torch.stack(B, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
                next_pred = self.large_scale_model(large_input)[0]  # (C, H, W)

                predictions.append(next_pred.cpu())
                B = B[1:] + [next_pred]
                remaining -= 1

        pred_array = torch.stack(predictions, dim=0).cpu().numpy()
        print(f"Pure large-scale baseline complete: {len(predictions)} predictions")
        return pred_array

    def visualize_cross_scale_prediction(
        self,
        sample_idx: int = 0,
        num_predictions: int = 100,
        fusion_weight: float = 0.5,
    ):
        """Visualize MR-PC cross-scale prediction results with baselines."""
        print(f"\n{'=' * 60}")
        print(f"Evaluating sample {sample_idx}")
        print(f"{'=' * 60}")

        # Get initial sample from small-scale dataset
        sample = self.small_scale_dataset[sample_idx]
        input_seq = sample["data"]["input_seq"].to(self.device)  # (1, T, C, H, W)

        # 1. Perform MR-PC cross-scale prediction
        pred_seq_mrpc, model_used, fusion_info = self.cross_scale_prediction(input_seq, num_predictions, fusion_weight)

        # 2. Perform pure small-scale baseline
        pred_seq_small = self.pure_small_scale_prediction(input_seq, num_predictions)

        # 3. Perform pure large-scale baseline
        pred_seq_large = self.pure_large_scale_prediction(input_seq, num_predictions)

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
        ground_truth_frames = []
        for i in range(num_predictions):
            if sample_idx + i < len(self.small_scale_dataset):
                sample_i = self.small_scale_dataset[sample_idx + i]
                target_i = sample_i["label"]  # (1, 1, C, H, W)
                target_denorm = self.small_scale_dataset.denormalize(target_i)
                target_frame = target_denorm.cpu().numpy()[0, 0]  # (C, H, W)
                ground_truth_frames.append(target_frame)
            else:
                if ground_truth_frames:
                    ground_truth_frames.append(ground_truth_frames[-1])

        # Get channel info
        channel_info = self.small_scale_dataset.get_channel_info()
        field_names = channel_info["field_names"]  # ["u", "v", "w"]

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

        # Select 9 points in a 3x3 grid
        H, W = pred_seq_mrpc.shape[-2:]

        # Create 3x3 grid of points
        h_positions = [H // 4, H // 2, 3 * H // 4]  # 1/4, 1/2, 3/4 positions
        w_positions = [W // 4, W // 2, 3 * W // 4]

        points = []
        for h_idx, h_pos in enumerate(h_positions):
            for w_idx, w_pos in enumerate(w_positions):
                points.append((h_pos, w_pos, h_idx, w_idx))

        print(f"  Monitoring {len(points)} points in 3x3 grid")

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

                # Plot ground truth
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

                # Plot ground truth
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
        warmup_end = 20
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

        # Select 9 points in a 3x3 grid (same as temporal evolution)
        H, W = pred_seq.shape[-2:]
        h_positions = [H // 4, H // 2, 3 * H // 4]
        w_positions = [W // 4, W // 2, 3 * W // 4]

        points = []
        for h_idx, h_pos in enumerate(h_positions):
            for w_idx, w_pos in enumerate(w_positions):
                points.append((h_pos, w_pos, h_idx, w_idx))

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

        # Create wavenumber arrays
        kx = np.fft.fftfreq(W, d=1.0) * W  # Streamwise wavenumber
        kz = np.fft.fftfreq(H, d=1.0) * H  # Spanwise wavenumber

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


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-scale evaluation for 1-plane Flow Swin Transformer")
    parser.add_argument(
        "--small_scale_checkpoint",
        type=str,
        default="/home/sh/CB/icon-thewell-dev/logs/flow_swin_1plane/runs/2025-11-02_14-11-12-461089/checkpoints/last.ckpt",
        help="Path to small-scale model checkpoint (t spacing)",
    )
    parser.add_argument(
        "--large_scale_checkpoint",
        type=str,
        default="/home/sh/CB/icon-thewell-dev/logs/flow_swin_1plane/runs/2025-11-02_23-13-52-741233/checkpoints/last.ckpt",
        help="Path to large-scale model checkpoint (5t spacing)",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to evaluate")
    parser.add_argument("--num_predictions", type=int, default=50, help="Number of future steps to predict")
    parser.add_argument(
        "--fusion_weight", type=float, default=0.5, help="Fusion weight α (0-1): x_fused = (1-α)*x_small + α*x_large"
    )

    args = parser.parse_args()

    # Model configurations
    small_scale_cfg = OmegaConf.create(
        {
            "input_shape": [128, 128],
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
        }
    )

    large_scale_cfg = OmegaConf.create(
        {
            "input_shape": [128, 128],
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
        }
    )

    # Data configuration
    data_config = {
        "data_dir": "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow",
        "input_length": 5,
        "field_names": ["u", "v", "w"],
        "file_pattern": "*u-v-w_scale2-3-1_yslice*.h5",
        "resolution_scale": (2, 3, 1),
        "y_slice": 54,
        "train_ratio": 0.7,
        "valid_ratio": 0.15,
        "test_ratio": 0.15,
        "enable_normalization": True,
        "norm_stats": "norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json",
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
    )

    print("\n" + "=" * 60)
    print("MR-PC evaluation complete!")
    print(f"Results saved to: {evaluator.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
