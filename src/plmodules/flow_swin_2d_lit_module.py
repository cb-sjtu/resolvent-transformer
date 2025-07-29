from collections.abc import Callable
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowSwin2DLitModule(L.LightningModule):
    """Lightning module for 2D flow field prediction using Swin Transformer."""

    def __init__(
        self, model: nn.Module, optimizer: Callable, scheduler: Callable = None, loss_fn: str = "mse", **kwargs
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

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
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_seq, target = batch

        # Forward pass
        pred = self.forward(input_seq)

        # Compute loss
        loss = self.loss_fn(pred, target)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Compute additional metrics
        with torch.no_grad():
            mse = F.mse_loss(pred, target)
            mae = F.l1_loss(pred, target)

            # Relative error (2D version)
            # Both pred and target are (B, T_pred, C, H, W)
            # Flatten spatial and channel dimensions to compute norm
            target_flat = target.flatten(start_dim=2)  # (B, T_pred, C*H*W)
            pred_flat = pred.flatten(start_dim=2)
            target_norm = torch.norm(target_flat, dim=2, keepdim=True)
            error_norm = torch.norm(pred_flat - target_flat, dim=2, keepdim=True)
            rel_error = (error_norm / (target_norm + 1e-8)).mean()

            # Spectral metrics for 2D fields
            pred_fft = torch.fft.fft2(pred.squeeze(2))  # Remove channel dim for FFT
            target_fft = torch.fft.fft2(target.squeeze(2))
            spectral_error = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

            self.log("train/mse", mse, on_step=True, on_epoch=True)
            self.log("train/mae", mae, on_step=True, on_epoch=True)
            self.log("train/rel_error", rel_error, on_step=True, on_epoch=True)
            self.log("train/spectral_error", spectral_error, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_seq, target = batch

        # Forward pass
        pred = self.forward(input_seq)

        # Compute loss
        loss = self.loss_fn(pred, target)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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

        self.log("val/mse", mse, on_step=False, on_epoch=True)
        self.log("val/mae", mae, on_step=False, on_epoch=True)
        self.log("val/rel_error", rel_error, on_step=False, on_epoch=True)
        self.log("val/spectral_error", spectral_error, on_step=False, on_epoch=True)
        self.log("val/grad_error_x", grad_error_x, on_step=False, on_epoch=True)
        self.log("val/grad_error_y", grad_error_y, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step."""
        input_seq, target = batch

        # Forward pass
        pred = self.forward(input_seq)

        # Compute loss
        loss = self.loss_fn(pred, target)

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

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

        self.log("test/mse", mse, on_step=False, on_epoch=True)
        self.log("test/mae", mae, on_step=False, on_epoch=True)
        self.log("test/rel_error", rel_error, on_step=False, on_epoch=True)
        self.log("test/spectral_error", spectral_error, on_step=False, on_epoch=True)
        self.log("test/grad_error_x", grad_error_x, on_step=False, on_epoch=True)
        self.log("test/grad_error_y", grad_error_y, on_step=False, on_epoch=True)
        self.log("test/ncc", ncc, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        input_seq, _ = batch
        return self.forward(input_seq)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and schedulers."""
        optimizer = self.optimizer(params=self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
