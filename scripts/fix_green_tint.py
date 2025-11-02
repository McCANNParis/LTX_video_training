#!/usr/bin/env python3
"""
Quick fix for green tint by adjusting color channels.
"""

import argparse
import imageio
import numpy as np


def fix_green_tint(video_path, output_path, green_reduction=0.6):
    """Reduce green channel to fix tint."""

    print(f"Loading {video_path}...")
    video = imageio.mimread(video_path)
    video = np.array(video)

    print(f"Original shape: {video.shape}")
    print(f"Original RGB means: R={video[:,:,:,0].mean():.1f}, G={video[:,:,:,1].mean():.1f}, B={video[:,:,:,2].mean():.1f}")

    # Reduce green channel
    video_fixed = video.copy().astype(np.float32)
    video_fixed[:,:,:,1] = video_fixed[:,:,:,1] * green_reduction

    # Renormalize to maintain brightness
    mean_before = video.mean()
    mean_after = video_fixed.mean()
    video_fixed = video_fixed * (mean_before / mean_after)

    video_fixed = np.clip(video_fixed, 0, 255).astype(np.uint8)

    print(f"Fixed RGB means: R={video_fixed[:,:,:,0].mean():.1f}, G={video_fixed[:,:,:,1].mean():.1f}, B={video_fixed[:,:,:,2].mean():.1f}")

    print(f"Saving to {output_path}...")
    fps = 30  # Default FPS
    imageio.mimsave(output_path, video_fixed, fps=fps)

    print("✓ Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input video with green tint")
    parser.add_argument("--output", type=str, required=True, help="Output corrected video")
    parser.add_argument("--green_reduction", type=float, default=0.6, help="Green reduction factor (default: 0.6)")

    args = parser.parse_args()
    fix_green_tint(args.input, args.output, args.green_reduction)


if __name__ == "__main__":
    main()
