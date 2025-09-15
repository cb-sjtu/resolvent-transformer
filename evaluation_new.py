#!/usr/bin/env python3
"""
New modular evaluation script for 2D Flow Swin Transformer.

This script replaces the original evaluation.py with a modular architecture:
- Separated concerns into different modules
- Added comprehensive time series monitoring
- Improved visualization capabilities
- Better code organization and maintainability

Key Features:
1. Time series monitoring at configurable points
2. Comprehensive metrics calculation
3. Multi-modal visualization (plots + videos)
4. Modular and extensible design
"""

import argparse
import os
import warnings

import hydra

# Add project root to path
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import our modular evaluation components
from evaluation_modules.flow_evaluator import FlowModelEvaluator
from evaluation_modules.utils import get_default_monitor_points

warnings.filterwarnings("ignore")


def create_custom_monitor_points():
    """Create custom monitoring points - easily configurable."""
    # Example: You can modify these points as needed
    # Points are (z_index, x_index) tuples for the 2D domain
    custom_points = [
        (50, 50),  # Near bottom-left
        (50, 128),  # Bottom-center
        (50, 200),  # Bottom-right
        (128, 50),  # Mid-left
        (128, 128),  # Center
        (128, 200),  # Mid-right
        (200, 50),  # Top-left
        (200, 128),  # Top-center
        (200, 200),  # Top-right
        (75, 75),  # Additional point
    ]
    return custom_points


def run_evaluation(
    checkpoint_path: str,
    save_predictions: bool = False,
    custom_points: bool = False,
    num_samples: int = 3,
    num_future_steps: int = 10,
    output_dir: str = "evaluation_results",
):
    """
    Run comprehensive flow model evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        save_predictions: Whether to save predictions as H5 files
        custom_points: Whether to use custom monitoring points
        num_samples: Number of samples to evaluate per split
        num_future_steps: Number of future steps to predict
        output_dir: Output directory for results
    """
    print("🚀 Starting Modular Flow Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Save predictions: {save_predictions}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per split: {num_samples}")
    print(f"Future steps: {num_future_steps}")

    # Determine monitoring points
    if custom_points:
        monitor_points = create_custom_monitor_points()
        print(f"Using {len(monitor_points)} custom monitoring points")
    else:
        monitor_points = get_default_monitor_points((256, 256))
        print(f"Using {len(monitor_points)} default monitoring points")

    print(f"Monitor points: {monitor_points}")

    # Create evaluator
    evaluator = FlowModelEvaluator(
        checkpoint_path=checkpoint_path,
        model_config=None,  # Will be loaded from checkpoint
        save_predictions=save_predictions,
        monitor_points=monitor_points,
        output_base_dir=output_dir,
    )

    try:
        # Load model and datasets
        evaluator.load_model_and_datasets()

        # Evaluate samples from test set
        print(f"\n{'=' * 50}")
        print("🔍 EVALUATION PHASE")
        print(f"{'=' * 50}")

        dataset_splits = ["test"]  # Can extend to ["train", "val", "test"]

        for split in dataset_splits:
            print(f"\n📊 Evaluating {split} set...")

            dataset = getattr(evaluator, f"{split}_dataset")
            actual_num_samples = min(num_samples, len(dataset))

            for sample_idx in range(actual_num_samples):
                print(f"\n🎯 Sample {sample_idx + 1}/{actual_num_samples}")
                evaluator.evaluate_sample(sample_idx=sample_idx, split=split, num_future=num_future_steps)

        # Create comprehensive time series analysis
        print(f"\n{'=' * 50}")
        print("📈 TIME SERIES ANALYSIS")
        print(f"{'=' * 50}")

        plots_dir, csv_dir = evaluator.create_time_series_summary()

        # Create videos for first sample
        print(f"\n{'=' * 50}")
        print("🎬 VIDEO GENERATION")
        print(f"{'=' * 50}")

        try:
            video_path = evaluator.create_videos(sample_idx=0, num_future=20)
            if video_path:
                print(f"✅ Video created: {video_path}")
            else:
                print("⚠️ Video creation failed")
        except Exception as e:
            print(f"⚠️ Video creation failed: {e}")

        # Final summary
        print(f"\n{'=' * 60}")
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}")
        print(f"📁 Results saved to: {evaluator.output_dir}")
        print(f"📈 Time series plots: {plots_dir}")
        print(f"📊 Time series data: {csv_dir}")

        if save_predictions:
            print(f"💾 Predictions saved to: {evaluator.predictions_dir}")

        print("\n📋 Summary:")
        print(f"  - Evaluated {actual_num_samples} samples")
        print(f"  - Monitored {len(monitor_points)} points")
        print(f"  - Generated {len(monitor_points)} individual time series plots")
        print("  - Created comprehensive metrics and visualizations")

    finally:
        # Cleanup
        evaluator.close_wandb()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Modular Flow Model Evaluation")

    parser.add_argument("checkpoint_path", nargs="?", default=None, help="Path to model checkpoint")

    parser.add_argument("--save-predictions", action="store_true", help="Save prediction results as H5 files")

    parser.add_argument(
        "--custom-points", action="store_true", help="Use custom monitoring points instead of default grid"
    )

    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to evaluate per dataset split")

    parser.add_argument("--num-future-steps", type=int, default=20, help="Number of future steps to predict")

    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory for all results")

    parser.add_argument("--config-overrides", nargs="*", default=[], help="Hydra config overrides")

    args = parser.parse_args()

    # Use default checkpoint path if not provided
    if args.checkpoint_path is None:
        args.checkpoint_path = (
            "/home/sh/CB/icon-thewell-dev/logs/flow_swin_2d/runs/2025-09-12_23-16-43-463283/checkpoints/step_30000.ckpt"
        )
        print(f"Using default checkpoint: {args.checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        print("Please provide a valid checkpoint path.")
        return

    # Initialize Hydra for configuration management
    with hydra.initialize(version_base="1.3", config_path="configs"):
        cfg = hydra.compose(config_name="train_flow_swin_2d", overrides=args.config_overrides)

        # Run evaluation
        run_evaluation(
            checkpoint_path=args.checkpoint_path,
            save_predictions=args.save_predictions,
            custom_points=args.custom_points,
            num_samples=args.num_samples,
            num_future_steps=args.num_future_steps,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
