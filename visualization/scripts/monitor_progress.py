#!/usr/bin/env python3
"""Monitor the progress of video creation."""

import os
import subprocess
import time


def monitor_progress():
    """Monitor video creation progress."""
    log_file = "all_frames_output.log"
    video_dir = "../videos"

    print("🎬 监控视频创建进度...")
    print("=" * 40)

    start_time = time.time()

    while True:
        # Check if log file exists and show last few lines
        if os.path.exists(log_file):
            try:
                with open(log_file) as f:
                    lines = f.readlines()
                    if lines:
                        # Show last non-empty line
                        for line in reversed(lines):
                            if line.strip():
                                print(f"最新状态: {line.strip()}")
                                break
            except:
                pass

        # Check if video files are being created
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) if f.endswith((".mp4", ".gif"))]
            if video_files:
                print(f"发现视频文件: {len(video_files)} 个")
                for vf in video_files:
                    file_path = os.path.join(video_dir, vf)
                    if os.path.exists(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        print(f"  - {vf}: {size_mb:.1f} MB")

        # Check if process is still running
        try:
            result = subprocess.run(["pgrep", "-f", "create_all_frames_video.py"], capture_output=True, text=True)
            if not result.stdout.strip():
                print("✅ 视频创建进程已完成!")
                break
        except:
            pass

        elapsed = time.time() - start_time
        print(f"运行时间: {elapsed / 60:.1f} 分钟")
        print("-" * 40)

        time.sleep(10)  # Check every 10 seconds

    # Final status
    print("\n🎉 最终结果:")
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith((".mp4", ".gif"))]
        for vf in sorted(video_files):
            file_path = os.path.join(video_dir, vf)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"✅ {vf}: {size_mb:.1f} MB")


if __name__ == "__main__":
    monitor_progress()
