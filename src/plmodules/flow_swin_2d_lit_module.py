import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.plmodules.base_lit_module import BaseLitModule


class FlowSwin2DLitModule(BaseLitModule):
    """Lightning module for 2D flow field prediction using Swin Transformer."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Get loss function from config
        loss_fn = getattr(cfg, "loss_fn", "mse")

        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_fn == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Scheduled Sampling configuration
        self.scheduled_sampling = getattr(cfg, "scheduled_sampling", {})
        self.ss_enabled = self.scheduled_sampling.get("enabled", False)
        self.ss_initial_ratio = self.scheduled_sampling.get(
            "initial_teacher_forcing_ratio", 1.0
        )
        self.ss_final_ratio = self.scheduled_sampling.get(
            "final_teacher_forcing_ratio", 0.0
        )
        self.ss_schedule_type = self.scheduled_sampling.get("schedule_type", "linear")
        self.ss_decay_epochs = self.scheduled_sampling.get("decay_epochs", 50)
        self.ss_start_epoch = self.scheduled_sampling.get("start_epoch", 0)
        self.ss_rollout_only = self.scheduled_sampling.get("rollout_only", True)

        # K-step Rollout configuration
        self.k_step_rollout = getattr(cfg, "k_step_rollout", {})
        self.kr_enabled = self.k_step_rollout.get("enabled", False)
        self.kr_initial_k = self.k_step_rollout.get("initial_k_steps", 1)
        self.kr_max_k = self.k_step_rollout.get("max_k_steps", 16)
        self.kr_schedule = self.k_step_rollout.get("k_increase_schedule", "curriculum")
        self.kr_curriculum_epochs = self.k_step_rollout.get(
            "curriculum_epochs", [10, 20, 30, 40, 50]
        )
        self.kr_curriculum_k_values = self.k_step_rollout.get(
            "curriculum_k_values", [1, 2, 4, 8, 16]
        )
        self.kr_rollout_weight = self.k_step_rollout.get("rollout_loss_weight", 1.0)
        self.kr_single_weight = self.k_step_rollout.get("single_step_loss_weight", 1.0)

        # Step-wise weighting configuration
        self.kr_step_weights = self.k_step_rollout.get(
            "step_weights", [1.0, 1.0, 1.0, 1.0, 1.0]
        )
        self.kr_enable_step_weighting = self.k_step_rollout.get(
            "enable_step_weighting", False
        )

        # Validate step_weights length matches curriculum_k_values
        if len(self.kr_step_weights) != len(self.kr_curriculum_k_values):
            raise ValueError(
                "Mismatch between lengths: "
                f"len(curriculum_k_steps)={len(self.kr_curriculum_k_steps)} "
                f"!= len(curriculum_k_values)={len(self.kr_curriculum_k_values)}"
            )
            # Pad or truncate step_weights to match curriculum_k_values
            if len(self.kr_step_weights) < len(self.kr_curriculum_k_values):
                # Pad with 1.0
                self.kr_step_weights.extend(
                    [1.0]
                    * (len(self.kr_curriculum_k_values) - len(self.kr_step_weights))
                )
            else:
                # Truncate
                self.kr_step_weights = self.kr_step_weights[
                    : len(self.kr_curriculum_k_values)
                ]

        # Cache for steps_per_epoch (will be set properly during training setup)
        self._cached_steps_per_epoch = None

        # Per-channel metrics configuration - dynamically detect number of channels
        self.enable_per_channel_metrics = getattr(
            cfg, "enable_per_channel_metrics", True
        )

        # Get number of channels from model config
        model_num_channels = cfg.model.get("num_channels", 9)

        # Automatically set up channel names based on number of channels
        if model_num_channels == 9:
            # 3-plane configuration (3 planes × 3 fields)
            self.channel_names = [
                "plane0_u_y29",
                "plane0_v_y29",
                "plane0_w_y29",
                "plane1_u_y54",
                "plane1_v_y54",
                "plane1_w_y54",
                "plane2_u_y75",
                "plane2_v_y75",
                "plane2_w_y75",
            ]
            self.num_planes = 3
            self.y_slices = [29, 54, 75]
        elif model_num_channels == 3:
            # 1-plane configuration (1 plane × 3 fields)
            self.channel_names = [
                "u_y54",
                "v_y54",
                "w_y54",
            ]
            self.num_planes = 1
            self.y_slices = [54]
        else:
            # Generic configuration - disable per-channel metrics
            print(
                f"Warning: Unknown channel configuration ({model_num_channels} channels). Per-channel metrics disabled."
            )
            self.enable_per_channel_metrics = False
            self.channel_names = [f"ch{i}" for i in range(model_num_channels)]
            self.num_planes = 1
            self.y_slices = []

        self.num_channels = len(self.channel_names)

    def _steps_per_epoch(self) -> int:
        """Calculate steps per epoch from dataset size and batch size."""
        # Use cached value if available and valid
        if (
            self._cached_steps_per_epoch is not None
            and self._cached_steps_per_epoch > 0
        ):
            return self._cached_steps_per_epoch

        return self._steps_per_epoch_implementation()

    def compute_per_channel_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict:
        """Compute per-channel loss and relative error metrics.

        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)

        Returns:
            dict: Per-channel metrics
        """
        if not self.enable_per_channel_metrics:
            return {}

        metrics = {}

        # Ensure we have the expected number of channels
        if pred.shape[1] != self.num_channels:
            # If number of channels doesn't match, skip per-channel metrics
            return {}

        # Compute per-channel losses and relative errors
        for ch_idx in range(self.num_channels):
            ch_name = self.channel_names[ch_idx]

            # Extract channel data
            pred_ch = pred[:, ch_idx]  # (B, H, W)
            target_ch = target[:, ch_idx]  # (B, H, W)

            # Per-channel MSE loss
            ch_mse = F.mse_loss(pred_ch, target_ch)
            metrics[f"{ch_name}_mse"] = ch_mse

            # Per-channel MAE
            ch_mae = F.l1_loss(pred_ch, target_ch)
            metrics[f"{ch_name}_mae"] = ch_mae

            # Per-channel relative error
            target_flat = target_ch.flatten(start_dim=1)  # (B, H*W)
            pred_flat = pred_ch.flatten(start_dim=1)  # (B, H*W)

            target_norm = torch.norm(target_flat, dim=1, keepdim=True)  # (B, 1)
            error_norm = torch.norm(
                pred_flat - target_flat, dim=1, keepdim=True
            )  # (B, 1)

            ch_rel_error = (error_norm / (target_norm + 1e-8)).mean()
            metrics[f"{ch_name}_rel_error"] = ch_rel_error

        # Compute grouped metrics (average per field across all planes)
        field_names = ["u", "v", "w"]  # Only velocity fields (removed pressure)

        if self.num_planes == 3:
            # 3-plane configuration: average across planes for each field
            for field_idx, field_name in enumerate(field_names):
                # Get indices for this field across all planes (3 fields per plane)
                field_channels = [field_idx + plane_idx * 3 for plane_idx in range(3)]

                # Average metrics for this field
                field_mse = torch.stack(
                    [metrics[f"{self.channel_names[ch]}_mse"] for ch in field_channels]
                ).mean()
                field_mae = torch.stack(
                    [metrics[f"{self.channel_names[ch]}_mae"] for ch in field_channels]
                ).mean()
                field_rel_error = torch.stack(
                    [
                        metrics[f"{self.channel_names[ch]}_rel_error"]
                        for ch in field_channels
                    ]
                ).mean()

                metrics[f"field_{field_name}_avg_mse"] = field_mse
                metrics[f"field_{field_name}_avg_mae"] = field_mae
                metrics[f"field_{field_name}_avg_rel_error"] = field_rel_error

            # Compute plane-wise metrics (average per plane across all fields)
            for plane_idx in range(3):
                plane_channels = [
                    plane_idx * 3 + field_idx for field_idx in range(3)
                ]  # 3 fields per plane

                plane_mse = torch.stack(
                    [metrics[f"{self.channel_names[ch]}_mse"] for ch in plane_channels]
                ).mean()
                plane_mae = torch.stack(
                    [metrics[f"{self.channel_names[ch]}_mae"] for ch in plane_channels]
                ).mean()
                plane_rel_error = torch.stack(
                    [
                        metrics[f"{self.channel_names[ch]}_rel_error"]
                        for ch in plane_channels
                    ]
                ).mean()

                y_slice = self.y_slices[plane_idx]
                metrics[f"plane{plane_idx}_y{y_slice}_avg_mse"] = plane_mse
                metrics[f"plane{plane_idx}_y{y_slice}_avg_mae"] = plane_mae
                metrics[f"plane{plane_idx}_y{y_slice}_avg_rel_error"] = plane_rel_error

        elif self.num_planes == 1:
            # 1-plane configuration: field metrics are the same as individual channel metrics
            for field_idx, field_name in enumerate(field_names):
                ch_name = self.channel_names[field_idx]
                metrics[f"field_{field_name}_avg_mse"] = metrics[f"{ch_name}_mse"]
                metrics[f"field_{field_name}_avg_mae"] = metrics[f"{ch_name}_mae"]
                metrics[f"field_{field_name}_avg_rel_error"] = metrics[
                    f"{ch_name}_rel_error"
                ]

            # Compute overall plane metrics (average across all fields in the single plane)
            plane_mse = torch.stack(
                [metrics[f"{self.channel_names[ch]}_mse"] for ch in range(3)]
            ).mean()
            plane_mae = torch.stack(
                [metrics[f"{self.channel_names[ch]}_mae"] for ch in range(3)]
            ).mean()
            plane_rel_error = torch.stack(
                [metrics[f"{self.channel_names[ch]}_rel_error"] for ch in range(3)]
            ).mean()

            y_slice = self.y_slices[0]
            metrics[f"plane0_y{y_slice}_avg_mse"] = plane_mse
            metrics[f"plane0_y{y_slice}_avg_mae"] = plane_mae
            metrics[f"plane0_y{y_slice}_avg_rel_error"] = plane_rel_error

        return metrics

    def _steps_per_epoch_implementation(self) -> int:
        """Implementation of steps per epoch calculation."""
        try:
            # First try to get from trainer (available during training)
            ntb = getattr(self.trainer, "num_training_batches", None)
            if ntb is not None and ntb != float("inf") and ntb > 0:
                num_batches = int(ntb)
                self._cached_steps_per_epoch = num_batches
                return num_batches

            # Second attempt: get from train_dataloader if available
            if (
                hasattr(self.trainer, "train_dataloader")
                and self.trainer.train_dataloader is not None
            ):
                try:
                    steps = len(self.trainer.train_dataloader)
                    if steps > 0:
                        self._cached_steps_per_epoch = steps
                        return steps
                except (TypeError, AttributeError) as e:
                    print(f"Failed to get length from trainer.train_dataloader: {e}")
                    # Try to get length from underlying dataset if it's a CycleLoader
                    print("Trying to access underlying dataloader from CycleLoader...")
                    try:
                        if hasattr(self.trainer.train_dataloader, "dataloader"):
                            print("Found .dataloader attribute")
                            underlying_loader = self.trainer.train_dataloader.dataloader
                            steps = len(underlying_loader)
                            print(f"Got length {steps} from underlying dataloader")
                            if steps > 0:
                                self._cached_steps_per_epoch = steps
                                return steps
                        elif hasattr(self.trainer.train_dataloader, "_dataloader"):
                            print("Found ._dataloader attribute")
                            underlying_loader = (
                                self.trainer.train_dataloader._dataloader
                            )
                            steps = len(underlying_loader)
                            print(f"Got length {steps} from underlying _dataloader")
                            if steps > 0:
                                self._cached_steps_per_epoch = steps
                                return steps
                        else:
                            print("No .dataloader or ._dataloader attribute found")
                            # Let's see what attributes it actually has
                            attrs = [
                                attr
                                for attr in dir(self.trainer.train_dataloader)
                                if not attr.startswith("__")
                            ]
                            print(
                                f"CycleLoader attributes: {attrs[:10]}..."
                            )  # Show first 10

                            # Try to access 'loaders' attribute
                            if hasattr(self.trainer.train_dataloader, "loaders"):
                                print("Found 'loaders' attribute")
                                loaders = self.trainer.train_dataloader.loaders
                                print(f"Loaders type: {type(loaders)}")
                                # 原：isinstance(loaders, (list, tuple)) and len(loaders) > 0
                                if isinstance(loaders, list | tuple) and loaders:
                                    first_loader = loaders[0]
                                    print(f"First loader type: {type(first_loader)}")

                                    # If it's a dict, examine its contents
                                    if isinstance(first_loader, dict):
                                        print(
                                            f"First loader is dict with keys: {list(first_loader.keys())}"
                                        )
                                        # Look for actual DataLoader in the dict values
                                        for key, value in first_loader.items():
                                            print(f"  Key '{key}': {type(value)}")
                                            try:
                                                if hasattr(value, "__len__"):
                                                    val_len = len(value)
                                                    print(
                                                        f"    Length of '{key}': {val_len}"
                                                    )
                                                    if (
                                                        val_len > 10
                                                    ):  # Reasonable threshold for real dataset
                                                        print(
                                                            f"Found reasonable length {val_len} for key '{key}'"
                                                        )
                                                        self._cached_steps_per_epoch = (
                                                            val_len
                                                        )
                                                        return val_len
                                            except Exception as e4:
                                                print(
                                                    f"    Failed to get length of '{key}': {e4}"
                                                )
                                    else:
                                        # Not a dict, try to get length directly
                                        try:
                                            steps = len(first_loader)
                                            print(
                                                f"Got length {steps} from first loader"
                                            )
                                            if steps > 0:
                                                self._cached_steps_per_epoch = steps
                                                return steps
                                        except Exception as e3:
                                            print(
                                                f"Failed to get length from first loader: {e3}"
                                            )
                    except (TypeError, AttributeError) as e2:
                        print(f"Failed to get length from underlying dataloader: {e2}")
                    pass

            # Third attempt: calculate from datamodule
            if (
                hasattr(self.trainer, "datamodule")
                and self.trainer.datamodule is not None
            ):
                try:
                    train_dataloader = self.trainer.datamodule.train_dataloader()
                    if train_dataloader is not None:
                        steps = len(train_dataloader)
                        if steps > 0:
                            self._cached_steps_per_epoch = steps
                            return steps
                except (TypeError, AttributeError) as e:
                    print(
                        f"Failed to get length from datamodule.train_dataloader(): {e}"
                    )
                    # Try to get length from underlying dataset if it's a CycleLoader
                    try:
                        train_dataloader = self.trainer.datamodule.train_dataloader()
                        if hasattr(train_dataloader, "dataloader"):
                            underlying_loader = train_dataloader.dataloader
                            steps = len(underlying_loader)
                            if steps > 0:
                                self._cached_steps_per_epoch = steps
                                return steps
                        elif hasattr(train_dataloader, "_dataloader"):
                            underlying_loader = train_dataloader._dataloader
                            steps = len(underlying_loader)
                            if steps > 0:
                                self._cached_steps_per_epoch = steps
                                return steps
                    except (TypeError, AttributeError) as e2:
                        print(f"Failed to get length from underlying dataloader: {e2}")
                    pass

            # Fourth attempt: calculate from dataset directly
            print("Attempting to calculate from dataset directly...")
            if (
                hasattr(self.trainer, "datamodule")
                and self.trainer.datamodule is not None
            ):
                try:
                    print("Datamodule exists, checking for train_dataset...")
                    # Try to get dataset size and batch size directly
                    if hasattr(self.trainer.datamodule, "train_dataset"):
                        print("Found train_dataset attribute")
                        dataset_size = len(self.trainer.datamodule.train_dataset)
                        batch_size = getattr(
                            self.trainer.datamodule, "batch_size_per_device", 8
                        )  # default to 8
                        steps = dataset_size // batch_size
                        print(
                            f"Calculated from dataset: {dataset_size} samples / {batch_size} batch_size = {steps} steps"
                        )
                        if steps > 0:
                            self._cached_steps_per_epoch = steps
                            return steps
                    else:
                        print("No train_dataset attribute found")
                        # Let's see what the datamodule has
                        attrs = [
                            attr
                            for attr in dir(self.trainer.datamodule)
                            if not attr.startswith("__")
                        ]
                        print(
                            f"Datamodule attributes: {attrs[:15]}..."
                        )  # Show first 15

                        # Try to get dataset using the available method
                        if hasattr(
                            self.trainer.datamodule, "get_train_dataset_from_cfg"
                        ):
                            print("Found get_train_dataset_from_cfg method")
                            try:
                                train_dataset = (
                                    self.trainer.datamodule.get_train_dataset_from_cfg()
                                )
                                if train_dataset is not None:
                                    # This should give us the 1094 value from len(self.indices)
                                    dataset_size = len(train_dataset)
                                    print(
                                        f"Train dataset size (len(indices)): {dataset_size}"
                                    )

                                    # Get batch_size_per_device from datamodule's configuration
                                    batch_size_per_device = None
                                    if hasattr(self.trainer.datamodule, "cfg"):
                                        batch_size_per_device = getattr(
                                            self.trainer.datamodule.cfg,
                                            "batch_size_per_device",
                                            None,
                                        )

                                    if batch_size_per_device is None:
                                        # Try direct attribute access
                                        batch_size_per_device = getattr(
                                            self.trainer.datamodule,
                                            "batch_size_per_device",
                                            None,
                                        )

                                    if batch_size_per_device is not None:
                                        steps = dataset_size // batch_size_per_device
                                        print(
                                            f"Calculated steps_per_epoch: {dataset_size} samples / "
                                            f"{batch_size_per_device} batch_size = {steps} steps"
                                        )
                                        if steps > 0:
                                            self._cached_steps_per_epoch = steps
                                            return steps
                                    else:
                                        print(
                                            "Could not find batch_size_per_device in datamodule configuration"
                                        )
                            except Exception as e_dataset:
                                print(
                                    f"Failed to get dataset via get_train_dataset_from_cfg: {e_dataset}"
                                )
                except (TypeError, AttributeError) as e:
                    print(f"Failed to calculate from dataset directly: {e}")
                    pass
            else:
                print("No datamodule found")

            # If all methods failed, provide detailed diagnostic info
            debug_info = []
            try:
                if hasattr(self.trainer, "num_training_batches"):
                    debug_info.append(
                        f"trainer.num_training_batches = {self.trainer.num_training_batches}"
                    )
            except Exception:
                debug_info.append("trainer.num_training_batches = <error accessing>")

            try:
                if hasattr(self.trainer, "train_dataloader"):
                    debug_info.append(
                        f"trainer.train_dataloader exists = {self.trainer.train_dataloader is not None}"
                    )
            except Exception:
                debug_info.append("trainer.train_dataloader = <error accessing>")

            try:
                if hasattr(self.trainer, "datamodule"):
                    debug_info.append(
                        f"trainer.datamodule exists = {self.trainer.datamodule is not None}"
                    )
            except Exception:
                debug_info.append("trainer.datamodule = <error accessing>")

            raise RuntimeError(
                "Failed to calculate steps_per_epoch. Unable to get training batch count from trainer, "
                f"train_dataloader, or datamodule. Debug info: {'; '.join(debug_info)}"
            )
        except Exception as e:
            raise RuntimeError(f"Error calculating steps_per_epoch: {e}") from e

    def on_train_start(self) -> None:
        """Called when the train begins."""
        super().on_train_start()
        # Force recalculation of steps_per_epoch now that trainer is fully set up
        self._cached_steps_per_epoch = None
        actual_steps = self._steps_per_epoch()
        print(f"Training started with {actual_steps} steps per epoch")
        print(f"=== CALCULATED STEPS_PER_EPOCH: {actual_steps} ===")

    def forward(self, x: torch.Tensor, return_delta: bool = False) -> torch.Tensor:
        """Forward pass with residual prediction.

        Args:
            x: Input sequence [B, T, C, H, W] (normalized)
            return_delta: If True, return delta prediction. If False, return absolute prediction.

        Returns:
            If return_delta=True: Delta prediction Δu (residual)
            If return_delta=False: Absolute prediction u_{t+1} = u_t + Δu
        """
        # Model outputs delta prediction Δu in normalized space
        delta_pred = self._model_forward(x)  # [B, C, H, W]

        if return_delta:
            return delta_pred
        else:
            # Residual composition: u_{t+1} = u_t + Δu
            x_last = x[:, -1]  # Last frame u_t [B, C, H, W]
            y_hat = x_last + delta_pred  # u_{t+1} = u_t + Δu
            return y_hat

    def get_teacher_forcing_ratio(self, current_epoch: int) -> float:
        """Calculate current teacher forcing ratio based on scheduled sampling."""
        if not self.ss_enabled:
            return 1.0

        # Use global_step instead of epoch since epochs aren't advancing
        steps_per_epoch = self._steps_per_epoch()
        effective_epoch = int(self.global_step // steps_per_epoch)

        if effective_epoch < self.ss_start_epoch:
            return 1.0

        progress = min(
            1.0, (effective_epoch - self.ss_start_epoch) / self.ss_decay_epochs
        )

        if self.ss_schedule_type == "linear":
            ratio = self.ss_initial_ratio - progress * (
                self.ss_initial_ratio - self.ss_final_ratio
            )
        elif self.ss_schedule_type == "exponential":
            ratio = self.ss_final_ratio + (
                self.ss_initial_ratio - self.ss_final_ratio
            ) * math.exp(-3 * progress)
        elif self.ss_schedule_type == "inverse_sigmoid":
            k = 2.0  # steepness parameter
            ratio = self.ss_final_ratio + (
                self.ss_initial_ratio - self.ss_final_ratio
            ) / (1 + math.exp(k * (progress - 0.5)))
        else:
            ratio = self.ss_initial_ratio

        return max(self.ss_final_ratio, min(self.ss_initial_ratio, ratio))

    def get_current_k_steps(self, current_epoch: int) -> int:
        """Calculate current K steps for rollout loss based on curriculum."""
        if not self.kr_enabled:
            return 1

        if self.kr_schedule == "fixed":
            return self.kr_initial_k
        elif self.kr_schedule == "curriculum":
            # Use global_step instead of epoch since epochs aren't advancing
            steps_per_epoch = self._steps_per_epoch()
            effective_epoch = self.global_step // steps_per_epoch

            # Find the appropriate k value based on effective epoch
            k_index = 0  # Start with first k value (k=1)
            for i, epoch_threshold in enumerate(self.kr_curriculum_epochs):
                if effective_epoch >= epoch_threshold:
                    k_index = i + 1  # Move to next k value
                else:
                    break  # Found the right threshold

            # Clamp to available k values
            k_index = min(k_index, len(self.kr_curriculum_k_values) - 1)
            return self.kr_curriculum_k_values[k_index]
        else:
            return self.kr_initial_k

    def rollout_prediction(
        self,
        input_seq: torch.Tensor,
        target_seq: torch.Tensor,
        k_steps: int,
        teacher_forcing_ratio: float,
    ) -> tuple[list, float]:
        """Perform k-step rollout prediction with scheduled sampling using residual prediction."""
        batch_size, seq_len, channels, height, width = input_seq.shape

        # Start with the input sequence (no unnecessary clone)
        current_input = input_seq
        losses = []
        tf_fractions = []  # Track actual teacher forcing fractions

        for step in range(k_steps):
            # Residual prediction: model outputs Δu, then compose u_{t+1} = u_t + Δu
            x_last = current_input[:, -1]  # Last frame u_t [B, C, H, W]
            delta_pred = self.forward(
                current_input, return_delta=True
            )  # Δu [B, C, H, W]
            pred = x_last + delta_pred  # u_{t+1} = u_t + Δu [B, C, H, W]

            # Calculate loss for this step
            if step < target_seq.shape[1]:  # Ensure we have target
                target = target_seq[:, step]  # (B, C, H, W)
                step_loss = self.loss_fn(pred, target)
                losses.append(step_loss)

            # Prepare input for next step with scheduled sampling
            if step < k_steps - 1:  # Not the last step
                # Per-sample teacher forcing decision
                B = input_seq.size(0)
                mask = (
                    torch.rand(B, 1, 1, 1, device=input_seq.device)
                    < teacher_forcing_ratio
                ).float()

                if step < target_seq.shape[1]:
                    # Mix ground truth and prediction based on mask
                    # Both pred and target[:, step] have shape (B, C, H, W)
                    # mask is (B, 1, 1, 1), expand to match (B, C, H, W)
                    mask_expanded = mask.expand(
                        -1, pred.shape[1], pred.shape[2], pred.shape[3]
                    )
                    mixed_frame = (
                        mask_expanded * target_seq[:, step]
                        + (1 - mask_expanded) * pred.detach()
                    )
                    next_frame = mixed_frame.unsqueeze(1)  # (B, 1, C, H, W)
                    # Record actual teacher forcing fraction for this step
                    tf_frac_step = mask.mean().detach()
                    tf_fractions.append(tf_frac_step)
                else:
                    # Use model prediction when no target available
                    next_frame = pred.detach().unsqueeze(1)  # (B, 1, C, H, W)
                    tf_fractions.append(torch.tensor(0.0, device=input_seq.device))

                # Update current_input: remove oldest frame and add new frame
                current_input = torch.cat(
                    [
                        current_input[:, 1:],  # Remove first frame
                        next_frame,
                    ],
                    dim=1,
                )

        # Calculate average teacher forcing fraction
        avg_tf_fraction = (
            torch.stack(tf_fractions).mean()
            if tf_fractions
            else torch.tensor(0.0, device=input_seq.device)
        )

        return losses, avg_tf_fraction.item()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with scheduled sampling and k-step rollout."""
        # Extract data from PyTree structure
        input_seq = batch["data"]["input_seq"]  # (B, T, C, H, W)
        target_seq = batch["label"]  # (B, max_k_steps, C, H, W) for k-step rollout

        # For backward compatibility, get first target frame for single-step loss
        target = target_seq[:, 0]  # (B, C, H, W)

        # Get batch size for logging
        batch_size = target.shape[0]

        current_epoch = self.current_epoch
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(current_epoch)
        k_steps = self.get_current_k_steps(current_epoch)

        # Debug logging for curriculum - use only on_step for continuous monitoring
        self.log(
            "debug/current_epoch",
            float(current_epoch),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        self.log(
            "debug/global_step",
            float(self.global_step),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )

        # Log effective epoch calculated from global_step
        steps_per_epoch = self._steps_per_epoch()
        effective_epoch = self.global_step // steps_per_epoch
        self.log(
            "debug/effective_epoch",
            float(effective_epoch),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )

        # Debug k-step curriculum
        self.log(
            "debug/kr_enabled",
            float(self.kr_enabled),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        # Convert schedule type to numeric for logging (curriculum=1, fixed=0)
        schedule_numeric = 1.0 if self.kr_schedule == "curriculum" else 0.0
        self.log(
            "debug/kr_schedule_is_curriculum",
            schedule_numeric,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )

        total_loss = 0.0

        # K-step rollout loss
        if self.kr_enabled and self.kr_rollout_weight > 0 and k_steps > 1:
            # Use the actual k-step target sequence from dataset
            # Limit k_steps to available targets
            actual_k_steps = min(k_steps, target_seq.shape[1])
            target_k_seq = target_seq[
                :, :actual_k_steps
            ]  # (B, actual_k_steps, C, H, W)

            # Perform rollout prediction
            rollout_losses, actual_tf_fraction = self.rollout_prediction(
                input_seq, target_k_seq, actual_k_steps, teacher_forcing_ratio
            )

            # Standard single-step prediction loss (use first rollout step to avoid duplicate forward)
            if self.kr_single_weight > 0 and rollout_losses:
                single_step_loss = rollout_losses[0]  # Use first step from rollout
                total_loss += self.kr_single_weight * single_step_loss
                # Log single step loss
                self.log(
                    "train/single_step_loss",
                    single_step_loss,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )
        else:
            # Fallback: Standard single-step prediction loss when k_steps <= 1 or rollout disabled
            if self.kr_single_weight > 0:
                # Use residual prediction: u_{t+1} = u_t + Δu
                pred_single = self.forward(
                    input_seq, return_delta=False
                )  # Get composed prediction
                single_step_loss = self.loss_fn(pred_single, target)
                total_loss += self.kr_single_weight * single_step_loss
                # Store for per-channel metrics
                self.current_pred = pred_single.detach()
                self.current_target = target.detach()
                # Log single step loss
                self.log(
                    "train/single_step_loss",
                    single_step_loss,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )
            rollout_losses = []
            actual_tf_fraction = 0.0  # No teacher forcing when not using rollout

        # Process rollout losses (when k_steps > 1)
        if (
            self.kr_enabled
            and self.kr_rollout_weight > 0
            and k_steps > 1
            and rollout_losses
        ):
            # Get current K curriculum index to determine step weights
            # current_k_index = 0
            for _, k_val in enumerate(self.kr_curriculum_k_values):
                if k_steps <= k_val:
                    # current_k_index = i
                    break

            # Avoid double-counting L1 loss: if k_steps>1 and single_step is enabled,
            # use only later steps for rollout loss to avoid overlap with single-step loss
            if self.kr_single_weight > 0 and len(rollout_losses) > 1:
                # Use rollout losses from step 1 onwards (skip first step to avoid double counting)
                later_losses = rollout_losses[1:]
                if later_losses:
                    if self.kr_enable_step_weighting:
                        # Apply step-wise weighting from configuration
                        weighted_losses = []
                        for i, step_loss in enumerate(later_losses):
                            step_idx = (
                                i + 1
                            )  # step 2, 3, 4, ... (0-indexed in later_losses)
                            # Use configured weights, fallback to 1.0
                            weight = self.kr_step_weights[
                                min(step_idx, len(self.kr_step_weights) - 1)
                            ]
                            weighted_losses.append(weight * step_loss)

                        rollout_loss = sum(weighted_losses) / len(weighted_losses)
                    else:
                        rollout_loss = sum(later_losses) / len(later_losses)

                    total_loss += self.kr_rollout_weight * rollout_loss

                    # Log individual step losses and weights
                    for i, step_loss in enumerate(rollout_losses):
                        self.log(
                            f"train/rollout_step{i + 1}_loss",
                            step_loss,
                            on_step=True,
                            on_epoch=False,
                            batch_size=batch_size,
                        )
                        # Log step weights when enabled
                        if self.kr_enable_step_weighting and i > 0:  # skip first step
                            weight = self.kr_step_weights[
                                min(i, len(self.kr_step_weights) - 1)
                            ]
                            self.log(
                                f"train/rollout_step{i + 1}_weight",
                                weight,
                                on_step=True,
                                on_epoch=False,
                                batch_size=batch_size,
                            )
            else:
                # Use all rollout losses
                if self.kr_enable_step_weighting:
                    # Apply step-wise weighting from configuration
                    weighted_losses = []
                    for i, step_loss in enumerate(rollout_losses):
                        # Use configured weights, fallback to 1.0
                        weight = self.kr_step_weights[
                            min(i, len(self.kr_step_weights) - 1)
                        ]
                        weighted_losses.append(weight * step_loss)

                    rollout_loss = sum(weighted_losses) / len(weighted_losses)
                else:
                    rollout_loss = sum(rollout_losses) / len(rollout_losses)

                total_loss += self.kr_rollout_weight * rollout_loss

                # Log individual step losses and weights
                for i, step_loss in enumerate(rollout_losses):
                    self.log(
                        f"train/rollout_step{i + 1}_loss",
                        step_loss,
                        on_step=True,
                        on_epoch=False,
                        batch_size=batch_size,
                    )
                    # Log step weights when enabled
                    if self.kr_enable_step_weighting:
                        weight = self.kr_step_weights[
                            min(i, len(self.kr_step_weights) - 1)
                        ]
                        self.log(
                            f"train/rollout_step{i + 1}_weight",
                            weight,
                            on_step=True,
                            on_epoch=False,
                            batch_size=batch_size,
                        )

            # Log average rollout loss
            avg_rollout_loss = sum(rollout_losses) / len(rollout_losses)
            self.log(
                "train/rollout_loss",
                avg_rollout_loss,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )

            # Log actual teacher forcing fraction
            self.log(
                "train/rollout_tf_actual_frac",
                actual_tf_fraction,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )

        # Always log k_steps when k-step rollout is enabled
        if self.kr_enabled:
            self.log(
                "train/k_steps",
                float(k_steps),
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
            # Log additional k-step rollout configuration state
            self.log(
                "train/kr_max_k_steps",
                float(self.kr_max_k),
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
            self.log(
                "train/kr_rollout_weight",
                self.kr_rollout_weight,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
            self.log(
                "train/kr_single_weight",
                self.kr_single_weight,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )

        # If no rollout or single step, fall back to standard prediction
        if total_loss == 0:
            pred = self.forward(
                input_seq, return_delta=False
            )  # Use residual prediction
            total_loss = self.loss_fn(pred, target)
            # Store for per-channel metrics
            self.current_pred = pred.detach()
            self.current_target = target.detach()

        # Log total loss and curriculum parameters
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Always log teacher forcing ratio when scheduled sampling is enabled
        if self.ss_enabled:
            self.log(
                "train/teacher_forcing_ratio",
                teacher_forcing_ratio,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
            # Also log whether teacher forcing is actually being used (only during rollout with k>1)
            teacher_forcing_active = self.kr_enabled and k_steps > 1
            self.log(
                "train/teacher_forcing_active",
                float(teacher_forcing_active),
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )

            # Optional: Track teacher forcing usage ratio within this batch (for debugging)
            if teacher_forcing_active and self.kr_enabled and k_steps > 1:
                # Estimate how often teacher forcing would be used based on the ratio
                # This is an approximation since we don't track actual TF usage in rollout_prediction
                expected_tf_usage = (
                    teacher_forcing_ratio * (k_steps - 1) / k_steps
                )  # Roughly
                self.log(
                    "train/rollout_expected_tf_usage",
                    expected_tf_usage,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )

        # Compute and log per-channel metrics for training
        if (
            self.enable_per_channel_metrics
            and hasattr(self, "current_pred")
            and hasattr(self, "current_target")
        ):
            # Use the most recent prediction and target from the training step
            try:
                per_channel_metrics = self.compute_per_channel_metrics(
                    self.current_pred, self.current_target
                )

                # Log per-channel metrics (reduced frequency to avoid log spam)
                if (
                    self.global_step % 100 == 0
                ):  # Log every 100 steps for better monitoring
                    for metric_name, metric_value in per_channel_metrics.items():
                        self.log(
                            f"train/{metric_name}",
                            metric_value,
                            on_step=True,
                            on_epoch=False,
                            batch_size=batch_size,
                        )

                # Always log summary metrics
                if "field_u_avg_rel_error" in per_channel_metrics:
                    self.log(
                        "train/field_u_avg_rel_error",
                        per_channel_metrics["field_u_avg_rel_error"],
                        on_step=True,
                        on_epoch=False,
                        batch_size=batch_size,
                    )
                if "field_v_avg_rel_error" in per_channel_metrics:
                    self.log(
                        "train/field_v_avg_rel_error",
                        per_channel_metrics["field_v_avg_rel_error"],
                        on_step=True,
                        on_epoch=False,
                        batch_size=batch_size,
                    )
                if "field_w_avg_rel_error" in per_channel_metrics:
                    self.log(
                        "train/field_w_avg_rel_error",
                        per_channel_metrics["field_w_avg_rel_error"],
                        on_step=True,
                        on_epoch=False,
                        batch_size=batch_size,
                    )
                if "field_p_avg_rel_error" in per_channel_metrics:
                    self.log(
                        "train/field_p_avg_rel_error",
                        per_channel_metrics["field_p_avg_rel_error"],
                        on_step=True,
                        on_epoch=False,
                        batch_size=batch_size,
                    )

            except Exception:
                # Silently skip per-channel metrics if there's an issue
                pass

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        """Validation step."""
        # Extract data from PyTree structure
        input_seq = batch["data"]["input_seq"]  # (B, T, C, H, W)
        target_seq = batch["label"]  # (B, max_k_steps, C, H, W) for k-step rollout

        # For validation, use the first target frame
        target = target_seq[:, 0]  # (B, C, H, W)

        # Forward pass
        pred = self.forward(input_seq, return_delta=False)  # Use residual prediction

        # Compute loss
        loss = self.loss_fn(pred, target)

        # Get batch size for logging
        batch_size = target.shape[0]

        # Log metrics
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Compute additional metrics
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)

        # Relative error
        target_flat = target.flatten(start_dim=2)
        pred_flat = pred.flatten(start_dim=2)
        target_norm = torch.norm(target_flat, dim=2, keepdim=True)
        error_norm = torch.norm(pred_flat - target_flat, dim=2, keepdim=True)
        rel_error = (error_norm / (target_norm + 1e-8)).mean()

        # Spectral metrics - fix dimension handling
        # Remove squeeze to handle [B,C,H,W] directly
        pred_for_fft = (
            pred.squeeze(1) if pred.shape[1] == 1 else pred.mean(dim=1)
        )  # Handle channel dimension properly
        target_for_fft = (
            target.squeeze(1) if target.shape[1] == 1 else target.mean(dim=1)
        )
        pred_fft = torch.fft.fft2(pred_for_fft)
        target_fft = torch.fft.fft2(target_for_fft)
        spectral_error = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        # Additional flow-specific metrics
        # Gradient preservation (important for flow fields)
        pred_grad_x = torch.gradient(pred_for_fft, dim=-1)[0]
        pred_grad_y = torch.gradient(pred_for_fft, dim=-2)[0]
        target_grad_x = torch.gradient(target_for_fft, dim=-1)[0]
        target_grad_y = torch.gradient(target_for_fft, dim=-2)[0]

        grad_error_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_error_y = F.mse_loss(pred_grad_y, target_grad_y)

        self.log(
            "val/mse",
            mse,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/mae",
            mae,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/rel_error",
            rel_error,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/spectral_error",
            spectral_error,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/grad_error_x",
            grad_error_x,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/grad_error_y",
            grad_error_y,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Compute and log per-channel metrics for validation
        if self.enable_per_channel_metrics:
            try:
                per_channel_metrics = self.compute_per_channel_metrics(pred, target)

                # Log all per-channel metrics for validation (full logging)
                for metric_name, metric_value in per_channel_metrics.items():
                    self.log(
                        f"val/{metric_name}",
                        metric_value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

            except Exception as e:
                print(
                    f"Warning: Failed to compute per-channel metrics in validation: {e}"
                )

        # Return metrics dict for save_metric callback
        metrics = {
            "val/loss": loss,
            "val/mse": mse,
            "val/mae": mae,
            "val/rel_error": rel_error,
            "val/spectral_error": spectral_error,
            "val/grad_error_x": grad_error_x,
            "val/grad_error_y": grad_error_y,
        }

        return {"metrics": metrics}

    def test_step(self, batch: Any, batch_idx: int) -> dict:
        """Test step."""
        # Extract data from PyTree structure
        input_seq = batch["data"]["input_seq"]  # (B, T, C, H, W)
        target_seq = batch["label"]  # (B, max_k_steps, C, H, W) for k-step rollout

        # For testing, use the first target frame
        target = target_seq[:, 0]  # (B, C, H, W)

        # Forward pass
        pred = self.forward(input_seq, return_delta=False)  # Use residual prediction

        # Compute loss
        loss = self.loss_fn(pred, target)

        # Get batch size for logging
        batch_size = target.shape[0]

        # Log metrics
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Compute comprehensive test metrics
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)

        # Relative error
        target_flat = target.flatten(start_dim=2)
        pred_flat = pred.flatten(start_dim=2)
        target_norm = torch.norm(target_flat, dim=2, keepdim=True)
        error_norm = torch.norm(pred_flat - target_flat, dim=2, keepdim=True)
        rel_error = (error_norm / (target_norm + 1e-8)).mean()

        # Spectral metrics - fix dimension handling
        # Remove squeeze to handle [B,C,H,W] directly
        pred_for_fft = (
            pred.squeeze(1) if pred.shape[1] == 1 else pred.mean(dim=1)
        )  # Handle channel dimension properly
        target_for_fft = (
            target.squeeze(1) if target.shape[1] == 1 else target.mean(dim=1)
        )
        pred_fft = torch.fft.fft2(pred_for_fft)
        target_fft = torch.fft.fft2(target_for_fft)
        spectral_error = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        # Flow-specific metrics
        pred_grad_x = torch.gradient(pred_for_fft, dim=-1)[0]
        pred_grad_y = torch.gradient(pred_for_fft, dim=-2)[0]
        target_grad_x = torch.gradient(target_for_fft, dim=-1)[0]
        target_grad_y = torch.gradient(target_for_fft, dim=-2)[0]

        grad_error_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_error_y = F.mse_loss(pred_grad_y, target_grad_y)

        # Structure similarity (SSIM-like for flow fields)
        def normalized_cross_correlation(x, y):
            x_flat = x.flatten(start_dim=1)
            y_flat = y.flatten(start_dim=1)
            x_norm = F.normalize(x_flat, dim=1)
            y_norm = F.normalize(y_flat, dim=1)
            return (x_norm * y_norm).sum(dim=1).mean()

        ncc = normalized_cross_correlation(pred_for_fft, target_for_fft)

        self.log(
            "test/mse",
            mse,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test/mae",
            mae,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test/rel_error",
            rel_error,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test/spectral_error",
            spectral_error,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test/grad_error_x",
            grad_error_x,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test/grad_error_y",
            grad_error_y,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test/ncc",
            ncc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Return metrics dict for save_metric callback
        metrics = {
            "test/loss": loss,
            "test/mse": mse,
            "test/mae": mae,
            "test/rel_error": rel_error,
            "test/spectral_error": spectral_error,
            "test/grad_error_x": grad_error_x,
            "test/grad_error_y": grad_error_y,
            "test/ncc": ncc,
        }

        return {"metrics": metrics}

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        input_seq, _ = batch
        return self.forward(input_seq)
