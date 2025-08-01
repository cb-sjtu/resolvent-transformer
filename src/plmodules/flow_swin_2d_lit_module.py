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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self._model_forward(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Extract data from PyTree structure
        input_seq = batch["data"]["input_seq"]  # (B, T, C, H, W)
        target = batch["label"]  # (B, C, H, W)

        # Forward pass
        pred = self.forward(input_seq)

        # Compute loss
        loss = self.loss_fn(pred, target)

        # Get batch size for logging
        batch_size = target.shape[0]

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Compute additional metrics
        with torch.no_grad():
            mse = F.mse_loss(pred, target)
            mae = F.l1_loss(pred, target)

            # Relative error (2D version)
            # Both pred and target are (B, C, H, W)
            # Flatten spatial dimensions to compute norm
            target_flat = target.flatten(start_dim=2)  # (B, C, H*W)
            pred_flat = pred.flatten(start_dim=2)
            target_norm = torch.norm(target_flat, dim=2, keepdim=True)
            error_norm = torch.norm(pred_flat - target_flat, dim=2, keepdim=True)
            rel_error = (error_norm / (target_norm + 1e-8)).mean()

            # Spectral metrics for 2D fields
            pred_fft = torch.fft.fft2(pred.squeeze(1))  # Remove channel dim for FFT: (B,C,H,W) -> (B,H,W)
            target_fft = torch.fft.fft2(target.squeeze(1))
            spectral_error = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

            self.log("train/mse", mse, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("train/mae", mae, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("train/rel_error", rel_error, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("train/spectral_error", spectral_error, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        """Validation step."""
        # Extract data from PyTree structure
        input_seq = batch["data"]["input_seq"]  # (B, T, C, H, W)
        target = batch["label"]  # (B, C, H, W)

        # Target should already be (B, C, H, W) from dataset
        # No reshaping needed

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

        # Spectral metrics
        pred_fft = torch.fft.fft2(pred.squeeze(2))
        target_fft = torch.fft.fft2(target.squeeze(2))
        spectral_error = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        # Additional flow-specific metrics
        # Gradient preservation (important for flow fields)
        pred_grad_x = torch.gradient(pred.squeeze(2), dim=-1)[0]
        pred_grad_y = torch.gradient(pred.squeeze(2), dim=-2)[0]
        target_grad_x = torch.gradient(target.squeeze(2), dim=-1)[0]
        target_grad_y = torch.gradient(target.squeeze(2), dim=-2)[0]

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
        target = batch["label"]  # (B, C, H, W)

        # Target should already be (B, C, H, W) from dataset
        # No reshaping needed

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

        # Spectral metrics
        pred_fft = torch.fft.fft2(pred.squeeze(2))
        target_fft = torch.fft.fft2(target.squeeze(2))
        spectral_error = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        # Flow-specific metrics
        pred_grad_x = torch.gradient(pred.squeeze(2), dim=-1)[0]
        pred_grad_y = torch.gradient(pred.squeeze(2), dim=-2)[0]
        target_grad_x = torch.gradient(target.squeeze(2), dim=-1)[0]
        target_grad_y = torch.gradient(target.squeeze(2), dim=-2)[0]

        grad_error_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_error_y = F.mse_loss(pred_grad_y, target_grad_y)

        # Structure similarity (SSIM-like for flow fields)
        def normalized_cross_correlation(x, y):
            x_flat = x.flatten(start_dim=1)
            y_flat = y.flatten(start_dim=1)
            x_norm = F.normalize(x_flat, dim=1)
            y_norm = F.normalize(y_flat, dim=1)
            return (x_norm * y_norm).sum(dim=1).mean()

        ncc = normalized_cross_correlation(pred.squeeze(2), target.squeeze(2))

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
