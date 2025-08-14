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
        self.ss_initial_ratio = self.scheduled_sampling.get("initial_teacher_forcing_ratio", 1.0)
        self.ss_final_ratio = self.scheduled_sampling.get("final_teacher_forcing_ratio", 0.0)
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
        self.kr_curriculum_epochs = self.k_step_rollout.get("curriculum_epochs", [10, 20, 30, 40, 50])
        self.kr_curriculum_k_values = self.k_step_rollout.get("curriculum_k_values", [1, 2, 4, 8, 16])
        self.kr_rollout_weight = self.k_step_rollout.get("rollout_loss_weight", 1.0)
        self.kr_single_weight = self.k_step_rollout.get("single_step_loss_weight", 1.0)

    def _steps_per_epoch(self) -> int:
        """Get dynamic steps per epoch from trainer, fallback to 50."""
        try:
            # Lightning will populate this during fit
            return max(1, int(self.trainer.num_training_batches))
        except Exception:
            return 50  # Fallback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self._model_forward(x)

    def get_teacher_forcing_ratio(self, current_epoch: int) -> float:
        """Calculate current teacher forcing ratio based on scheduled sampling."""
        if not self.ss_enabled:
            return 1.0

        # Use global_step instead of epoch since epochs aren't advancing
        steps_per_epoch = self._steps_per_epoch()
        effective_epoch = int(self.global_step // steps_per_epoch)

        if effective_epoch < self.ss_start_epoch:
            return 1.0

        progress = min(1.0, (effective_epoch - self.ss_start_epoch) / self.ss_decay_epochs)

        if self.ss_schedule_type == "linear":
            ratio = self.ss_initial_ratio - progress * (self.ss_initial_ratio - self.ss_final_ratio)
        elif self.ss_schedule_type == "exponential":
            ratio = self.ss_final_ratio + (self.ss_initial_ratio - self.ss_final_ratio) * math.exp(-3 * progress)
        elif self.ss_schedule_type == "inverse_sigmoid":
            k = 2.0  # steepness parameter
            ratio = self.ss_final_ratio + (self.ss_initial_ratio - self.ss_final_ratio) / (
                1 + math.exp(k * (progress - 0.5))
            )
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
        self, input_seq: torch.Tensor, target_seq: torch.Tensor, k_steps: int, teacher_forcing_ratio: float
    ) -> tuple[list, float]:
        """Perform k-step rollout prediction with scheduled sampling."""
        batch_size, seq_len, channels, height, width = input_seq.shape

        # Start with the input sequence (no unnecessary clone)
        current_input = input_seq
        losses = []
        tf_fractions = []  # Track actual teacher forcing fractions

        for step in range(k_steps):
            # Forward pass to predict next frame
            pred = self.forward(current_input)  # Shape: (B, C, H, W)

            # Calculate loss for this step
            if step < target_seq.shape[1]:  # Ensure we have target
                target = target_seq[:, step]  # (B, C, H, W)
                step_loss = self.loss_fn(pred, target)
                losses.append(step_loss)

            # Prepare input for next step with scheduled sampling
            if step < k_steps - 1:  # Not the last step
                # Per-sample teacher forcing decision
                B = input_seq.size(0)
                mask = (torch.rand(B, 1, 1, 1, device=input_seq.device) < teacher_forcing_ratio).float()

                if step < target_seq.shape[1]:
                    # Mix ground truth and prediction based on mask
                    # Both pred and target[:, step] have shape (B, C, H, W)
                    # mask is (B, 1, 1, 1), expand to match (B, C, H, W)
                    mask_expanded = mask.expand(-1, pred.shape[1], pred.shape[2], pred.shape[3])
                    mixed_frame = mask_expanded * target_seq[:, step] + (1 - mask_expanded) * pred.detach()
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
            torch.stack(tf_fractions).mean() if tf_fractions else torch.tensor(0.0, device=input_seq.device)
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
        self.log("debug/current_epoch", float(current_epoch), on_step=True, on_epoch=False, batch_size=batch_size)
        self.log("debug/global_step", float(self.global_step), on_step=True, on_epoch=False, batch_size=batch_size)

        # Log effective epoch calculated from global_step
        steps_per_epoch = self._steps_per_epoch()
        effective_epoch = self.global_step // steps_per_epoch
        self.log("debug/effective_epoch", float(effective_epoch), on_step=True, on_epoch=False, batch_size=batch_size)

        # Debug k-step curriculum
        self.log("debug/kr_enabled", float(self.kr_enabled), on_step=True, on_epoch=False, batch_size=batch_size)
        # Convert schedule type to numeric for logging (curriculum=1, fixed=0)
        schedule_numeric = 1.0 if self.kr_schedule == "curriculum" else 0.0
        self.log(
            "debug/kr_schedule_is_curriculum", schedule_numeric, on_step=True, on_epoch=False, batch_size=batch_size
        )

        total_loss = 0.0

        # K-step rollout loss
        if self.kr_enabled and self.kr_rollout_weight > 0 and k_steps > 1:
            # Use the actual k-step target sequence from dataset
            # Limit k_steps to available targets
            actual_k_steps = min(k_steps, target_seq.shape[1])
            target_k_seq = target_seq[:, :actual_k_steps]  # (B, actual_k_steps, C, H, W)

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
                    "train/single_step_loss", single_step_loss, on_step=True, on_epoch=False, batch_size=batch_size
                )
        else:
            # Fallback: Standard single-step prediction loss when k_steps <= 1 or rollout disabled
            if self.kr_single_weight > 0:
                pred_single = self.forward(input_seq)
                single_step_loss = self.loss_fn(pred_single, target)
                total_loss += self.kr_single_weight * single_step_loss
                # Log single step loss
                self.log(
                    "train/single_step_loss", single_step_loss, on_step=True, on_epoch=False, batch_size=batch_size
                )
            rollout_losses = []
            actual_tf_fraction = 0.0  # No teacher forcing when not using rollout

        # Process rollout losses (when k_steps > 1)
        if self.kr_enabled and self.kr_rollout_weight > 0 and k_steps > 1 and rollout_losses:
            # Avoid double-counting L1 loss: if k_steps>1 and single_step is enabled,
            # use only later steps for rollout loss to avoid overlap with single-step loss
            if self.kr_single_weight > 0 and len(rollout_losses) > 1:
                # Use rollout losses from step 1 onwards (skip first step to avoid double counting)
                later_losses = rollout_losses[1:]
                if later_losses:
                    rollout_loss = sum(later_losses) / len(later_losses)
                    total_loss += self.kr_rollout_weight * rollout_loss

                    # Log individual step losses
                    for i, step_loss in enumerate(rollout_losses):
                        self.log(
                            f"train/rollout_step{i + 1}_loss",
                            step_loss,
                            on_step=True,
                            on_epoch=False,
                            batch_size=batch_size,
                        )
            else:
                # Use all rollout losses
                rollout_loss = sum(rollout_losses) / len(rollout_losses)
                total_loss += self.kr_rollout_weight * rollout_loss

                # Log individual step losses
                for i, step_loss in enumerate(rollout_losses):
                    self.log(
                        f"train/rollout_step{i + 1}_loss",
                        step_loss,
                        on_step=True,
                        on_epoch=False,
                        batch_size=batch_size,
                    )

            # Log average rollout loss
            avg_rollout_loss = sum(rollout_losses) / len(rollout_losses)
            self.log("train/rollout_loss", avg_rollout_loss, on_step=True, on_epoch=False, batch_size=batch_size)

            # Log actual teacher forcing fraction
            self.log(
                "train/rollout_tf_actual_frac", actual_tf_fraction, on_step=True, on_epoch=False, batch_size=batch_size
            )

        # Always log k_steps when k-step rollout is enabled
        if self.kr_enabled:
            self.log("train/k_steps", float(k_steps), on_step=True, on_epoch=False, batch_size=batch_size)
            # Log additional k-step rollout configuration state
            self.log("train/kr_max_k_steps", float(self.kr_max_k), on_step=True, on_epoch=False, batch_size=batch_size)
            self.log(
                "train/kr_rollout_weight", self.kr_rollout_weight, on_step=True, on_epoch=False, batch_size=batch_size
            )
            self.log(
                "train/kr_single_weight", self.kr_single_weight, on_step=True, on_epoch=False, batch_size=batch_size
            )

        # If no rollout or single step, fall back to standard prediction
        if total_loss == 0:
            pred = self.forward(input_seq)
            total_loss = self.loss_fn(pred, target)

        # Log total loss and curriculum parameters
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)

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
                expected_tf_usage = teacher_forcing_ratio * (k_steps - 1) / k_steps  # Roughly
                self.log(
                    "train/rollout_expected_tf_usage",
                    expected_tf_usage,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )

        # Skip expensive metrics calculation during training to save memory
        # These metrics are computed during validation/test steps

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        """Validation step."""
        # Extract data from PyTree structure
        input_seq = batch["data"]["input_seq"]  # (B, T, C, H, W)
        target_seq = batch["label"]  # (B, max_k_steps, C, H, W) for k-step rollout

        # For validation, use the first target frame
        target = target_seq[:, 0]  # (B, C, H, W)

        # Forward pass
        pred = self.forward(input_seq)

        # Compute loss
        loss = self.loss_fn(pred, target)

        # Get batch size for logging
        batch_size = target.shape[0]

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

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
        pred_for_fft = pred.squeeze(1) if pred.shape[1] == 1 else pred.mean(dim=1)  # Handle channel dimension properly
        target_for_fft = target.squeeze(1) if target.shape[1] == 1 else target.mean(dim=1)
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

        self.log("val/mse", mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val/mae", mae, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val/rel_error", rel_error, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(
            "val/spectral_error", spectral_error, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
        )
        self.log("val/grad_error_x", grad_error_x, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val/grad_error_y", grad_error_y, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

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
        pred = self.forward(input_seq)

        # Compute loss
        loss = self.loss_fn(pred, target)

        # Get batch size for logging
        batch_size = target.shape[0]

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

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
        pred_for_fft = pred.squeeze(1) if pred.shape[1] == 1 else pred.mean(dim=1)  # Handle channel dimension properly
        target_for_fft = target.squeeze(1) if target.shape[1] == 1 else target.mean(dim=1)
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

        self.log("test/mse", mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test/mae", mae, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test/rel_error", rel_error, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(
            "test/spectral_error", spectral_error, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size
        )
        self.log("test/grad_error_x", grad_error_x, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test/grad_error_y", grad_error_y, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test/ncc", ncc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

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
