#!/usr/bin/env python3
"""
Create video with ALL frames from preprocessed flow data.
Optimized for large datasets with memory-efficient processing.
"""

import glob
import os

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def compute_velocity_magnitude(velocity_data):
    """Compute velocity magnitude from u, v, w components."""
    if velocity_data.ndim != 3 or velocity_data.shape[0] < 3:
        raise ValueError(f"Expected 3D array with at least 3 channels, got shape {velocity_data.shape}")

    u, v, w = velocity_data[0], velocity_data[1], velocity_data[2]
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    return magnitude


def create_all_frames_video(
    data_dir: str = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow",
    output_dir: str = "/home/sh/CB/icon-thewell-dev/visualization/videos",
    fps: int = 20,  # Higher FPS for smoother long video
    field_names: list = None,
    colorbar_percentiles: tuple = (1, 99),
):
    """Create video with ALL frames from preprocessed data."""
    if field_names is None:
        field_names = ["u", "v", "w"]

    os.makedirs(output_dir, exist_ok=True)

    # Find all files
    field_names_str = "-".join(field_names)
    pattern = f"{field_names_str}_scale*_yslice*_*.h5"
    file_list = sorted(glob.glob(os.path.join(data_dir, pattern)))

    if not file_list:
        print(f"No files found matching pattern: {pattern}")
        return

    total_frames = len(file_list)
    print(f"Found {total_frames} files - will process ALL frames")
    print(f"Estimated video duration: {total_frames / fps:.1f} seconds at {fps} FPS")

    # Pre-compute global ranges by sampling every 10th frame
    print("Computing global color ranges from sample frames...")
    sample_indices = range(0, total_frames, max(1, total_frames // 100))  # Sample 100 frames

    all_u, all_v, all_w, all_mag = [], [], [], []

    for i in tqdm(sample_indices, desc="Sampling frames for color range"):
        with h5py.File(file_list[i], "r") as f:
            data = f["data"][()]
            if data.shape[0] >= 3:
                all_u.append(data[0].flatten())
                all_v.append(data[1].flatten())
                all_w.append(data[2].flatten())
                magnitude = compute_velocity_magnitude(data)
                all_mag.append(magnitude.flatten())

    # Compute global ranges
    u_vmin, u_vmax = np.percentile(np.concatenate(all_u), colorbar_percentiles)
    v_vmin, v_vmax = np.percentile(np.concatenate(all_v), colorbar_percentiles)
    w_vmin, w_vmax = np.percentile(np.concatenate(all_w), colorbar_percentiles)
    mag_vmin, mag_vmax = np.percentile(np.concatenate(all_mag), colorbar_percentiles)

    print("Global ranges computed:")
    print(f"  u: [{u_vmin:.6f}, {u_vmax:.6f}]")
    print(f"  v: [{v_vmin:.6f}, {v_vmax:.6f}]")
    print(f"  w: [{w_vmin:.6f}, {w_vmax:.6f}]")
    print(f"  magnitude: [{mag_vmin:.6f}, {mag_vmax:.6f}]")

    # Create the video
    print("Creating combined video with all frames...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Flow Field Evolution - {total_frames} Frames", fontsize=16)

    # Load first frame to initialize plots
    with h5py.File(file_list[0], "r") as f:
        first_data = f["data"][()]
        first_magnitude = compute_velocity_magnitude(first_data)

    # Initialize plots
    im_u = axes[0, 0].imshow(first_data[0], cmap="viridis", aspect="auto", vmin=u_vmin, vmax=u_vmax)
    axes[0, 0].set_title("u velocity")
    axes[0, 0].axis("off")
    plt.colorbar(im_u, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im_v = axes[0, 1].imshow(first_data[1], cmap="viridis", aspect="auto", vmin=v_vmin, vmax=v_vmax)
    axes[0, 1].set_title("v velocity")
    axes[0, 1].axis("off")
    plt.colorbar(im_v, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_w = axes[1, 0].imshow(first_data[2], cmap="viridis", aspect="auto", vmin=w_vmin, vmax=w_vmax)
    axes[1, 0].set_title("w velocity")
    axes[1, 0].axis("off")
    plt.colorbar(im_w, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_mag = axes[1, 1].imshow(first_magnitude, cmap="plasma", aspect="auto", vmin=mag_vmin, vmax=mag_vmax)
    axes[1, 1].set_title("Velocity Magnitude")
    axes[1, 1].axis("off")
    plt.colorbar(im_mag, ax=axes[1, 1], fraction=0.046, pad=0.04)

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    def animate(frame_idx):
        """Load and display frame data."""
        with h5py.File(file_list[frame_idx], "r") as f:
            data = f["data"][()]

        magnitude = compute_velocity_magnitude(data)

        # Update images
        im_u.set_data(data[0])
        im_v.set_data(data[1])
        im_w.set_data(data[2])
        im_mag.set_data(magnitude)

        # Update time text
        time_text.set_text(f"Frame: {frame_idx + 1}/{total_frames} | Time: {frame_idx / fps:.1f}s")

        return [im_u, im_v, im_w, im_mag, time_text]

    # Create animation
    print(f"Creating animation with {total_frames} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=1000 // fps, blit=True, repeat=True)

    # Try to save as MP4 first, then GIF
    video_path = os.path.join(output_dir, f"flow_all_{total_frames}_frames.mp4")

    try:
        print(f"Saving MP4 video: {video_path}")
        writer = animation.FFMpegWriter(
            fps=fps,
            metadata=dict(artist="Flow Visualization", title=f"Flow Evolution {total_frames} frames"),
            bitrate=2400,  # Higher bitrate for better quality
        )
        anim.save(video_path, writer=writer)
        print(f"✅ MP4 video saved: {video_path}")

        # Get file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"   File size: {file_size_mb:.1f} MB")

    except Exception as e:
        print(f"❌ MP4 failed: {e}")

        # Fallback to GIF (will be very large!)
        gif_path = os.path.join(output_dir, f"flow_all_{total_frames}_frames.gif")
        print(f"Trying GIF (warning: will be VERY large): {gif_path}")

        try:
            anim.save(gif_path, writer="pillow", fps=fps)
            file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
            print(f"✅ GIF saved: {gif_path}")
            print(f"   File size: {file_size_mb:.1f} MB")
        except Exception as e2:
            print(f"❌ GIF also failed: {e2}")

    plt.close(fig)
    print("✅ Video creation completed!")


if __name__ == "__main__":
    print("Creating video with ALL frames from preprocessed flow data")
    print("=" * 60)

    create_all_frames_video(
        fps=20,  # 20 FPS for ~54 second video with 1081 frames
    )
