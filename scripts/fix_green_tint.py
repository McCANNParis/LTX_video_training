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

    # Use reader to avoid memory issues
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get('fps', 30)

    # Process frames in batches
    frames_fixed = []
    r_sum, g_sum, b_sum = 0, 0, 0
    r_sum_fixed, g_sum_fixed, b_sum_fixed = 0, 0, 0
    total_pixels = 0

    print("Processing frames...")
    for i, frame in enumerate(reader):
        if i % 30 == 0:
            print(f"  Frame {i}...")

        # Calculate stats for first frame
        if i == 0:
            print(f"  Frame shape: {frame.shape}")
            print(f"  Original RGB: R={frame[:,:,0].mean():.1f}, G={frame[:,:,1].mean():.1f}, B={frame[:,:,2].mean():.1f}")

        # Accumulate stats
        r_sum += frame[:,:,0].sum()
        g_sum += frame[:,:,1].sum()
        b_sum += frame[:,:,2].sum()
        total_pixels += frame.shape[0] * frame.shape[1]

        # Fix green channel
        frame_fixed = frame.astype(np.float32)
        frame_fixed[:,:,1] = frame_fixed[:,:,1] * green_reduction

        # Renormalize to maintain brightness
        mean_before = frame.mean()
        mean_after = frame_fixed.mean()
        frame_fixed = frame_fixed * (mean_before / mean_after)

        frame_fixed = np.clip(frame_fixed, 0, 255).astype(np.uint8)

        # Accumulate fixed stats
        r_sum_fixed += frame_fixed[:,:,0].sum()
        g_sum_fixed += frame_fixed[:,:,1].sum()
        b_sum_fixed += frame_fixed[:,:,2].sum()

        frames_fixed.append(frame_fixed)

    reader.close()

    # Print overall stats
    num_frames = len(frames_fixed)
    print(f"\nProcessed {num_frames} frames")
    print(f"Original RGB means: R={r_sum/(total_pixels*num_frames):.1f}, G={g_sum/(total_pixels*num_frames):.1f}, B={b_sum/(total_pixels*num_frames):.1f}")
    print(f"Fixed RGB means: R={r_sum_fixed/(total_pixels*num_frames):.1f}, G={g_sum_fixed/(total_pixels*num_frames):.1f}, B={b_sum_fixed/(total_pixels*num_frames):.1f}")

    print(f"\nSaving to {output_path}...")
    imageio.mimsave(output_path, frames_fixed, fps=fps)

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
