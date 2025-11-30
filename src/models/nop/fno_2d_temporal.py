"""
FNO2D model for temporal flow field prediction on 3-plane data.

This model uses Fourier Neural Operator (FNO) to predict the next timestep
of 3D flow fields represented as 3 y-planes with 4 channels each (u, v, w, p).

Key features:
- Spectral convolutions in frequency domain using neuralop library
- Global receptive field for capturing long-range dependencies
- Parameter-efficient compared to transformer-based models
- Handles non-periodic boundaries with padding strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO2d


class FNO2DTemporalModel(nn.Module):
    """FNO-based model for temporal flow field prediction using neuralop.

    Architecture:
        Input (B, T*C, H, W) = (B, 60, 128, 128)
          ↓
        Lifting: 60 → hidden_channels
          ↓
        FNO2d (neuralop): spectral convolutions
          ↓
        Projection: hidden_channels → 12
          ↓
        Output (B, C, H, W) = (B, 12, 128, 128)

    Args:
        input_shape: Spatial dimensions (H, W)
        sequence_length: Number of input timesteps (default: 5)
        prediction_horizon: Number of future steps to predict (default: 1)
        num_channels: Number of channels per timestep (default: 12 for 3 planes × 4 fields)
        hidden_channels: Hidden dimension for FNO layers
        num_layers: Number of FNO layers
        num_modes: Number of Fourier modes to keep in each dimension [modes_height, modes_width]
        lifting_channels: Number of channels for lifting operation (default: hidden_channels)
        projection_channels: Number of channels for projection operation (default: hidden_channels)
        use_mlp: Whether to use MLP layers in FNO blocks
        non_linearity: Activation function (default: F.gelu)
        stabilizer: Stabilizer for spectral convolution (default: None)
    """

    def __init__(
        self,
        input_shape: tuple[int, int] = (128, 128),
        sequence_length: int = 5,
        prediction_horizon: int = 1,
        num_channels: int = 12,
        hidden_channels: int = 64,
        num_layers: int = 4,
        num_modes: tuple[int, int] = (16, 16),
        lifting_channels: int = None,
        projection_channels: int = None,
        use_mlp: bool = False,
        non_linearity=F.gelu,
        stabilizer: str = None,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_modes = num_modes

        # Total input channels = sequence_length × num_channels
        self.in_channels = sequence_length * num_channels  # 5 × 12 = 60

        # Set default lifting/projection channels
        if lifting_channels is None:
            lifting_channels = hidden_channels * 2
        if projection_channels is None:
            projection_channels = hidden_channels * 2

        # FNO2d from neuralop
        # Let neuralop handle lifting and projection internally
        self.fno = FNO2d(
            n_modes_height=num_modes[0],
            n_modes_width=num_modes[1],
            hidden_channels=hidden_channels,
            in_channels=self.in_channels,  # Input: 60 channels
            out_channels=num_channels,  # Output: 12 channels
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=num_layers,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            channel_mlp_dropout=0.0,
            channel_mlp_expansion=0.5,  # Default expansion
            norm=None,
            skip="soft-gating",  # Use soft gating for skip connections
            separable=False,
            preactivation=False,
        )

    def _init_weights(self):
        """Initialize model weights (neuralop handles FNO weights internally)."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, T*C, H, W)
               where T=sequence_length, C=num_channels

        Returns:
            Output tensor of shape (B, C, H, W) representing the predicted next timestep
        """
        # Handle both input formats
        if x.dim() == 5:
            # Input is (B, T, C, H, W), flatten temporal and channel dimensions
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)
        elif x.dim() == 4:
            # Input is already (B, T*C, H, W)
            B, TC, H, W = x.shape
            assert self.in_channels == TC, f"Expected {self.in_channels} channels, got {TC}"
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        # FNO2d: neuralop handles lifting, spectral convolutions, and projection
        x = self.fno(x)

        return x

    def get_model_info(self) -> dict:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "FNO2D Temporal (neuralop)",
            "input_shape": self.input_shape,
            "sequence_length": self.sequence_length,
            "num_channels": self.num_channels,
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "num_modes": self.num_modes,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


# Unit test
if __name__ == "__main__":
    print("Testing FNO2DTemporalModel with neuralop...")

    # Test with 3-plane flow data
    model = FNO2DTemporalModel(
        input_shape=(128, 128),
        sequence_length=5,
        num_channels=12,
        hidden_channels=64,
        num_layers=4,
        num_modes=(16, 16),
        use_mlp=False,
    )

    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass with 4D input
    print("\nTesting forward pass with 4D input...")
    x_4d = torch.randn(2, 60, 128, 128)  # (B, T*C, H, W)
    y = model(x_4d)
    print(f"  Input shape:  {x_4d.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (2, 12, 128, 128), f"Expected (2, 12, 128, 128), got {y.shape}"

    # Test forward pass with 5D input
    print("\nTesting forward pass with 5D input...")
    x_5d = torch.randn(2, 5, 12, 128, 128)  # (B, T, C, H, W)
    y = model(x_5d)
    print(f"  Input shape:  {x_5d.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (2, 12, 128, 128), f"Expected (2, 12, 128, 128), got {y.shape}"

    print("\n✓ All tests passed!")
    print(f"\nModel uses neuralop's FNO2d with {info['num_layers']} layers")
    print(f"Total parameters: {info['total_params']:,}")
