#!/usr/bin/env python3
"""
New modular evaluation script for 1-plane Flow Swin Transformer.

This script adopts the modular architecture from evaluation_new.py for 1-plane models:
- Separated concerns into different modules
- Added comprehensive 1-plane visualization
- Improved code organization and maintainability
- Fixed WandB step counting issues

Key Features:
1. Modular and extensible design
2. 1-plane specific visualizations (3-channel support for u, v, w)
3. Proper WandB integration without step conflicts
4. Multi-modal visualization (plots + videos)
"""

import argparse
import os
import warnings

# Add project root to path
import rootutils

# Try to import hydra, but handle gracefully if not available
try:
    import hydra

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

warnings.filterwarnings("ignore")

# Import our modular evaluation components
from evaluation_modules.flow_evaluator_1plane import Flow1PlaneEvaluator  # noqa: E402
from evaluation_modules.utils import get_default_monitor_points  # noqa: E402

warnings.filterwarnings("ignore")

# ========================================
# 🎯 CONFIGURATION: Modify this value to change prediction length everywhere
# ========================================
DEFAULT_FUTURE_STEPS = 20  # Number of future steps to predict for 1-plane


def create_1plane_monitor_points():
    """Create 1-plane specific monitoring points - 9 points total (single plane × 9 positions)."""
    # Base 2D positions in the domain (z_index, x_index)
    base_positions = [
        # Adjusted monitoring points for 128x128 domain (indices 0-127)
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

    # For 1-plane, we directly use the 2D positions: (z_idx, x_idx)
    custom_points = base_positions

    return custom_points


def run_comprehensive_1plane_evaluation(
    checkpoint_path: str,
    save_predictions: bool = False,
    custom_points: bool = True,
    num_samples: int = 3,
    num_future_steps: int = DEFAULT_FUTURE_STEPS,
    output_dir: str = "evaluation_1plane_outputs",
):
    """
    Run comprehensive 1-plane flow model evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        save_predictions: Whether to save predictions as H5 files
        custom_points: Whether to use custom monitoring points
        num_samples: Number of samples to evaluate per split
        num_future_steps: Number of future steps to predict
        output_dir: Output directory for results
    """
    print("🚀 Starting Modular 1-Plane Flow Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Save predictions: {save_predictions}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per split: {num_samples}")
    print(f"Future steps: {num_future_steps}")

    # Determine monitoring points for 1-plane
    if custom_points:
        monitor_points = create_1plane_monitor_points()
        print(f"Using {len(monitor_points)} custom 1-plane monitoring points")
    else:
        # Assume 1-plane data uses similar spatial dimensions
        monitor_points = get_default_monitor_points((256, 256))
        print(f"Using {len(monitor_points)} default monitoring points")

    print(f"Monitor points: {monitor_points}")

    # Create 1-plane evaluator
    evaluator = Flow1PlaneEvaluator(
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
        print(f"\n{'=' * 60}")
        print("🔍 1-PLANE EVALUATION PHASE")
        print(f"{'=' * 60}")

        dataset_splits = ["test"]  # Can extend to ["train", "val", "test"]

        for split in dataset_splits:
            print(f"\n📊 Evaluating {split} set...")

            dataset = getattr(evaluator, f"{split}_dataset")
            actual_num_samples = min(num_samples, len(dataset))

            for sample_idx in range(actual_num_samples):
                print(f"\n🎯 Sample {sample_idx + 1}/{actual_num_samples}")
                evaluator.evaluate_1plane_sample(sample_idx=sample_idx, split=split, num_future=num_future_steps)

        # Create comprehensive 1-plane analysis
        print(f"\n{'=' * 60}")
        print("📈 1-PLANE ANALYSIS PHASE")
        print(f"{'=' * 60}")

        evaluator.create_comprehensive_1plane_analysis()

        # Create time series analysis (similar to evaluation_new.py)
        print(f"\n{'=' * 60}")
        print("📊 TIME SERIES ANALYSIS")
        print(f"{'=' * 60}")

        plots_dir, csv_dir = evaluator.create_time_series_summary()

        # Final summary
        print(f"\n{'=' * 70}")
        print("✅ 1-PLANE EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 70}")
        print(f"📁 Results saved to: {evaluator.output_dir}")
        print(f"📈 Time series plots: {plots_dir}")
        print(f"📊 Time series data: {csv_dir}")

        if save_predictions:
            print(f"💾 Predictions saved to: {evaluator.predictions_dir}")

        print("\n📋 1-Plane Summary:")
        print(f"  - Evaluated {num_samples} samples from test set")
        print(f"  - Monitored {len(monitor_points)} 1-plane points")
        print(f"  - Generated {num_future_steps} future steps per sample")
        print(f"  - Generated {len(monitor_points)} individual time series plots")
        print("  - Created comprehensive 1-plane metrics and visualizations")

    except Exception as e:
        print(f"\n❌ 1-Plane evaluation failed: {str(e)}")
        raise

    finally:
        # Cleanup
        try:
            evaluator.close_wandb()
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")


def main() -> None:
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Modular 1-Plane Flow Model Evaluation")

    parser.add_argument("checkpoint_path", nargs="?", default=None, help="Path to model checkpoint")

    parser.add_argument("--save-predictions", action="store_true", help="Save prediction results as H5 files")

    parser.add_argument(
        "--custom-points",
        action="store_true",
        default=True,
        help="Use custom 1-plane monitoring points instead of default grid",
    )

    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to evaluate per dataset split")

    parser.add_argument(
        "--num-future-steps", type=int, default=DEFAULT_FUTURE_STEPS, help="Number of future steps to predict"
    )

    parser.add_argument(
        "--output-dir", type=str, default="evaluation_1plane_outputs", help="Output directory for all results"
    )

    parser.add_argument("--config-overrides", nargs="*", default=[], help="Hydra config overrides")

    args = parser.parse_args()

    # Use default checkpoint path if not provided
    if args.checkpoint_path is None:
        args.checkpoint_path = (
            "/home/sh/CB/icon-thewell-dev/logs/flow_swin_1plane/"
            "runs/2025-11-02_23-13-52-741233/checkpoints/step_41700.ckpt"
        )
        print(f"Using default checkpoint: {args.checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        print("Please provide a valid checkpoint path.")
        return

    print(f"🔧 Starting 1-plane evaluation with checkpoint: {args.checkpoint_path}")

    # Initialize Hydra for configuration management if available
    if HYDRA_AVAILABLE:
        try:
            with hydra.initialize(version_base="1.3", config_path="configs"):
                hydra.compose(config_name="train_flow_swin_1plane", overrides=args.config_overrides)

                # Run evaluation
                run_comprehensive_1plane_evaluation(
                    checkpoint_path=args.checkpoint_path,
                    save_predictions=args.save_predictions,
                    custom_points=args.custom_points,
                    num_samples=args.num_samples,
                    num_future_steps=args.num_future_steps,
                    output_dir=args.output_dir,
                )
        except Exception as e:
            print(f"⚠️ Hydra configuration failed: {e}, running without configuration management")
            # Run evaluation without Hydra
            run_comprehensive_1plane_evaluation(
                checkpoint_path=args.checkpoint_path,
                save_predictions=args.save_predictions,
                custom_points=args.custom_points,
                num_samples=args.num_samples,
                num_future_steps=args.num_future_steps,
                output_dir=args.output_dir,
            )
    else:
        print("⚠️ Hydra not available, running without configuration management")
        # Run evaluation without Hydra
        run_comprehensive_1plane_evaluation(
            checkpoint_path=args.checkpoint_path,
            save_predictions=args.save_predictions,
            custom_points=args.custom_points,
            num_samples=args.num_samples,
            num_future_steps=args.num_future_steps,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
