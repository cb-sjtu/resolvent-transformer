#!/usr/bin/env python3
"""
Flow Data Video Visualization Script

This script creates video visualizations of preprocessed flow data showing temporal evolution
of u, v, w velocity components and velocity magnitude.
"""

import argparse
import glob
import os

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def compute_velocity_magnitude(velocity_data):
    """Compute velocity magnitude from u, v, w components.

    Args:
        velocity_data: Velocity data with shape (C, H, W) where C >= 3 for u, v, w

    Returns:
        Velocity magnitude with shape (H, W)
    """
    if velocity_data.ndim != 3 or velocity_data.shape[0] < 3:
        raise ValueError(
            f"Expected 3D array with at least 3 channels, got shape {velocity_data.shape}"
        )

    u, v, w = velocity_data[0], velocity_data[1], velocity_data[2]
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    return magnitude


def create_preprocessed_flow_videos(
    data_dir: str,
    output_dir: str,
    max_frames: int = 100,
    fps: int = 10,
    field_names: list = None,
    colorbar_percentiles: tuple = (1, 99),
):
    """Create video visualizations of preprocessed flow data showing temporal evolution.

    Args:
        data_dir: Directory containing preprocessed flow data H5 files
        output_dir: Directory to save output videos
        max_frames: Maximum number of frames to process (0 for all)
        fps: Frames per second for the output video
        field_names: List of field names, defaults to ["u", "v", "w"]
        colorbar_percentiles: Percentiles for colorbar range (min, max)
    """
    if field_names is None:
        field_names = ["u", "v", "w"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all preprocessed files for multi-channel data (u-v-w)
    field_names_str = "-".join(field_names)
    pattern = f"{field_names_str}_scale*_yslice*_*.h5"
    file_list = sorted(glob.glob(os.path.join(data_dir, pattern)))

    if not file_list:
        print(f"No files found matching pattern: {pattern}")
        print("Available files in directory:")
        for f in sorted(os.listdir(data_dir))[:10]:
            print(f"  {f}")
        return

    print(f"Found {len(file_list)} files matching pattern: {pattern}")

    # Limit number of frames to process
    if max_frames > 0:
        file_list = file_list[:max_frames]
        print(f"Processing {len(file_list)} frames")

    # Load all data and compute global ranges for consistent color scaling
    all_data = []
    all_magnitudes = []

    print("Loading data and computing ranges...")
    for _, file_path in enumerate(tqdm(file_list, desc="Loading files")):
        with h5py.File(file_path, "r") as f:
            data = f["data"][()]  # Shape: (C, H, W) where C=3 for u, v, w
            all_data.append(data)

            # Compute velocity magnitude
            if data.shape[0] >= 3:
                magnitude = compute_velocity_magnitude(data)
                all_magnitudes.append(magnitude)
            else:
                print(
                    f"Warning: File {file_path} has only {data.shape[0]} channels, expected >= 3"
                )
                all_magnitudes.append(np.zeros((data.shape[1], data.shape[2])))

    all_data = np.array(all_data)  # Shape: (T, C, H, W)
    all_magnitudes = np.array(all_magnitudes)  # Shape: (T, H, W)

    print(f"Loaded data shape: {all_data.shape}")
    print(f"Magnitude shape: {all_magnitudes.shape}")

    # Compute global color ranges for consistent visualization
    channel_ranges = {}
    for c, channel_name in enumerate(field_names[: all_data.shape[1]]):
        channel_data = all_data[:, c, :, :]
        vmin, vmax = np.percentile(channel_data, colorbar_percentiles)
        channel_ranges[channel_name] = (vmin, vmax)
        print(f"{channel_name} channel range: [{vmin:.4f}, {vmax:.4f}]")

    mag_vmin, mag_vmax = np.percentile(all_magnitudes, colorbar_percentiles)
    print(f"Magnitude range: [{mag_vmin:.4f}, {mag_vmax:.4f}]")

    # Create combined video with 4 subplots: u, v, w, magnitude
    create_combined_video(
        all_data,
        all_magnitudes,
        channel_ranges,
        mag_vmin,
        mag_vmax,
        field_names,
        output_dir,
        fps,
        file_list,
    )

    # Create individual channel videos
    create_individual_videos(
        all_data, channel_ranges, field_names, output_dir, fps, file_list
    )

    # Create magnitude-only video
    create_magnitude_video(
        all_magnitudes, mag_vmin, mag_vmax, output_dir, fps, file_list
    )

    print("Video creation completed!")


def create_combined_video(
    all_data,
    all_magnitudes,
    channel_ranges,
    mag_vmin,
    mag_vmax,
    field_names,
    output_dir,
    fps,
    file_list,
):
    """Create combined video showing all channels and magnitude."""
    print("Creating combined video...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Temporal Evolution of Flow Fields", fontsize=16)

    # Initialize plots
    ims = []

    # u channel
    im_u = axes[0, 0].imshow(
        all_data[0, 0],
        cmap="viridis",
        aspect="auto",
        vmin=channel_ranges["u"][0],
        vmax=channel_ranges["u"][1],
    )
    axes[0, 0].set_title("u velocity")
    axes[0, 0].axis("off")
    plt.colorbar(im_u, ax=axes[0, 0], fraction=0.046, pad=0.04)
    ims.append(im_u)

    # v channel
    im_v = axes[0, 1].imshow(
        all_data[0, 1],
        cmap="viridis",
        aspect="auto",
        vmin=channel_ranges["v"][0],
        vmax=channel_ranges["v"][1],
    )
    axes[0, 1].set_title("v velocity")
    axes[0, 1].axis("off")
    plt.colorbar(im_v, ax=axes[0, 1], fraction=0.046, pad=0.04)
    ims.append(im_v)

    # w channel
    im_w = axes[1, 0].imshow(
        all_data[0, 2],
        cmap="viridis",
        aspect="auto",
        vmin=channel_ranges["w"][0],
        vmax=channel_ranges["w"][1],
    )
    axes[1, 0].set_title("w velocity")
    axes[1, 0].axis("off")
    plt.colorbar(im_w, ax=axes[1, 0], fraction=0.046, pad=0.04)
    ims.append(im_w)

    # Magnitude
    im_mag = axes[1, 1].imshow(
        all_magnitudes[0], cmap="plasma", aspect="auto", vmin=mag_vmin, vmax=mag_vmax
    )
    axes[1, 1].set_title("Velocity Magnitude")
    axes[1, 1].axis("off")
    plt.colorbar(im_mag, ax=axes[1, 1], fraction=0.046, pad=0.04)
    ims.append(im_mag)

    # Add timestep text
    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    def animate(frame):
        """Animation function."""
        # Update u, v, w channels
        ims[0].set_data(all_data[frame, 0])
        ims[1].set_data(all_data[frame, 1])
        ims[2].set_data(all_data[frame, 2])

        # Update magnitude
        ims[3].set_data(all_magnitudes[frame])

        # Update timestep text
        time_text.set_text(f"Timestep: {frame + 1}/{len(file_list)}")

        return ims + [time_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(file_list),
        interval=1000 // fps,
        blit=True,
        repeat=True,
    )

    # Save video
    video_path = os.path.join(output_dir, "flow_combined_evolution.mp4")
    save_animation(anim, video_path, fps)
    plt.close(fig)


def create_individual_videos(
    all_data, channel_ranges, field_names, output_dir, fps, file_list
):
    """Create individual videos for each channel."""
    for c, channel_name in enumerate(field_names[: all_data.shape[1]]):
        print(f"Creating video for {channel_name} channel...")

        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
        fig_single.suptitle(
            f"Temporal Evolution of {channel_name.upper()} Velocity", fontsize=14
        )

        vmin, vmax = channel_ranges[channel_name]
        im_single = ax_single.imshow(
            all_data[0, c], cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax
        )
        ax_single.axis("off")
        cb_single = plt.colorbar(im_single, ax=ax_single, fraction=0.046, pad=0.04)
        cb_single.set_label(f"{channel_name} velocity")

        time_text_single = fig_single.text(0.5, 0.02, "", ha="center", fontsize=12)

        def animate_single(frame):
            im_single.set_data(all_data[frame, c])
            time_text_single.set_text(f"Timestep: {frame + 1}/{len(file_list)}")
            return [im_single, time_text_single]

        anim_single = animation.FuncAnimation(
            fig_single,
            animate_single,
            frames=len(file_list),
            interval=1000 // fps,
            blit=True,
            repeat=True,
        )

        # Save individual channel video
        video_path_single = os.path.join(
            output_dir, f"flow_{channel_name}_evolution.mp4"
        )
        save_animation(anim_single, video_path_single, fps)
        plt.close(fig_single)


def create_magnitude_video(
    all_magnitudes, mag_vmin, mag_vmax, output_dir, fps, file_list
):
    """Create magnitude-only video."""
    print("Creating magnitude video...")

    fig_mag, ax_mag = plt.subplots(1, 1, figsize=(8, 6))
    fig_mag.suptitle("Temporal Evolution of Velocity Magnitude", fontsize=14)

    im_mag_only = ax_mag.imshow(
        all_magnitudes[0], cmap="plasma", aspect="auto", vmin=mag_vmin, vmax=mag_vmax
    )
    ax_mag.axis("off")
    cb_mag_only = plt.colorbar(im_mag_only, ax=ax_mag, fraction=0.046, pad=0.04)
    cb_mag_only.set_label("Velocity Magnitude")

    time_text_mag = fig_mag.text(0.5, 0.02, "", ha="center", fontsize=12)

    def animate_magnitude(frame):
        im_mag_only.set_data(all_magnitudes[frame])
        time_text_mag.set_text(f"Timestep: {frame + 1}/{len(file_list)}")
        return [im_mag_only, time_text_mag]

    anim_mag = animation.FuncAnimation(
        fig_mag,
        animate_magnitude,
        frames=len(file_list),
        interval=1000 // fps,
        blit=True,
        repeat=True,
    )

    video_path_mag = os.path.join(output_dir, "flow_magnitude_evolution.mp4")
    save_animation(anim_mag, video_path_mag, fps)
    plt.close(fig_mag)


def save_animation(anim, video_path, fps):
    """Save animation with fallback options."""
    print(f"Saving video to: {video_path}")

    try:
        # Try to save as MP4
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist="Flow Visualization"), bitrate=1800
        )
        anim.save(video_path, writer=writer)
        print(f"Video saved successfully: {video_path}")
    except Exception as e:
        print(f"Error saving MP4: {e}")
        # Fallback to GIF
        gif_path = video_path.replace(".mp4", ".gif")
        print(f"Trying to save as GIF: {gif_path}")
        try:
            anim.save(gif_path, writer="pillow", fps=fps)
            print(f"GIF saved successfully: {gif_path}")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")
            print(
                "Please ensure ffmpeg is installed for MP4 output or pillow for GIF output"
            )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create flow data visualization videos"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sh/CB/icon-thewell-dev/data/preprocessed_flow",
        help="Directory containing preprocessed flow data H5 files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sh/CB/icon-thewell-dev/visualization/videos",
        help="Directory to save output videos",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames to process (0 for all)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for output video"
    )
    parser.add_argument(
        "--fields", nargs="+", default=["u", "v", "w"], help="Field names to visualize"
    )
    parser.add_argument(
        "--percentiles",
        nargs=2,
        type=float,
        default=[1, 99],
        help="Percentiles for colorbar range (min max)",
    )

    args = parser.parse_args()

    print("Flow Data Video Visualization")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max frames: {args.max_frames}")
    print(f"FPS: {args.fps}")
    print(f"Fields: {args.fields}")
    print(f"Colorbar percentiles: {args.percentiles}")
    print()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return

    create_preprocessed_flow_videos(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        fps=args.fps,
        field_names=args.fields,
        colorbar_percentiles=tuple(args.percentiles),
    )


if __name__ == "__main__":
    main()
