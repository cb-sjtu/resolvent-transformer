#!/usr/bin/env python3
"""
Test script for 2D Flow Swin Transformer implementation.
"""

import os
import sys

import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from plmodules.flow_swin_2d_lit_module import FlowSwin2DLitModule
from src.datasets.flow_sequence_2d.flow_sequence_2d import FlowSequence2DDataset
from src.models.swin_transformer.swin_transformer_2d import SwinTransformer2D


def test_model():
    """Test the 2D Swin Transformer model."""
    print("Testing 2D Swin Transformer model...")

    # Create model
    model = SwinTransformer2D(
        input_shape=(256, 256),  # 2D shape (z, x) after y-slice
        sequence_length=5,
        prediction_horizon=1,
        embed_dim=96,
        depths=(2, 2, 2),  # Smaller for testing
        num_heads=(3, 6, 12),
        window_size=(7, 7),
        patch_size=(4, 4),
    )

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 5, 1, 256, 256)  # (B, T, C, z, x)

    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Check output shape is correct
    expected_shape = (batch_size, 1, 1, 256, 256)  # (B, T_pred, C, z, x)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    print("✓ Model test passed!")
    return True


def test_dataset():
    """Test the 2D dataset (if data directory exists)."""
    data_dir = "/media/sh/Seagate Basic/RE550/"

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Skipping dataset test.")
        return True

    print("Testing 2D dataset...")

    try:
        # Test FlowSequence2DDataset  original shape (x,y,z)->(768,384,512)
        dataset = FlowSequence2DDataset(
            data_dir=data_dir,
            input_length=5,
            field_name="u",
            resolution_scale=(3, 1, 2),
            y_slice=None,  # Use middle y-slice
            split="train",
        )

        print(f"Dataset length: {len(dataset)}")

        if len(dataset) > 0:
            input_seq, target = dataset[0]
            print(f"Input sequence shape: {input_seq.shape}")
            print(f"Target shape: {target.shape}")

            # Check shapes are correct
            assert input_seq.shape[0] == 5, f"Expected 5 timesteps, got {input_seq.shape[0]}"
            assert input_seq.shape[1] == 1, f"Expected 1 channel, got {input_seq.shape[1]}"
            assert target.shape[0] == 1, f"Expected 1 channel, got {target.shape[0]}"

            print("✓ Dataset test passed!")
        else:
            print("⚠ Dataset is empty")

    except Exception as e:
        print(f"Dataset test failed: {e}")
        return False

    return True


def test_lightning_module():
    """Test the Lightning module."""
    print("Testing Lightning module...")

    # Create model
    model = SwinTransformer2D(
        input_shape=(256, 256),  # Smaller for testing (z, x)
        sequence_length=5,
        prediction_horizon=1,
        embed_dim=48,
        depths=(1, 1),
        num_heads=(3, 6),
        window_size=(8, 8),
        patch_size=(4, 4),
    )

    # Create optimizer (dummy) - not used directly, just for completeness

    # Create Lightning module
    lit_module = FlowSwin2DLitModule(
        model=model, optimizer=lambda params: torch.optim.Adam(params, lr=1e-3), loss_fn="mse"
    )

    # Test forward pass
    batch_size = 2
    input_seq = torch.randn(batch_size, 5, 1, 256, 256)
    target = torch.randn(batch_size, 1, 1, 256, 256)

    # Test training step
    batch = (input_seq, target)
    loss = lit_module.training_step(batch, 0)

    print(f"Training loss: {loss.item():.4f}")

    # Test validation step
    val_loss = lit_module.validation_step(batch, 0)
    print(f"Validation loss: {val_loss.item():.4f}")

    print("✓ Lightning module test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing 2D Flow Swin Transformer Implementation")
    print("=" * 50)

    tests = [
        ("Model", test_model),
        ("Dataset", test_dataset),
        ("Lightning Module", test_lightning_module),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 20} {test_name} {'-' * 20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'=' * 50}")
    print("Test Summary:")
    for test_name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(success for _, success in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    main()
