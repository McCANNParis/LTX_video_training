#!/usr/bin/env python3
"""
Test base model without LoRA to diagnose generation issues.
"""

import argparse
import torch
from diffusers import LTXPipeline
import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="test_base.mp4")
    parser.add_argument("--negative_prompt", type=str, default="worst quality, inconsistent motion, blurry")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("="*60)
    print("Testing Base LTX-Video Model (No LoRA)")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.width}x{args.height}x{args.num_frames}")
    print("="*60)

    # Load pipeline
    print("\nLoading LTX-Video pipeline...")
    pipeline = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16
    )
    pipeline = pipeline.to("cuda")

    # Generate
    print(f"\nGenerating video...")
    print(f"  Steps: {args.steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.inference_mode():
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )

    # Debug: Check output structure
    print(f"\nDebug - Output type: {type(output)}")
    if hasattr(output, '__dict__'):
        print(f"Debug - Output attributes: {list(output.__dict__.keys())}")

    # Extract frames
    if hasattr(output, 'frames'):
        video = output.frames[0]
        print("Debug - Using output.frames[0]")
    elif hasattr(output, 'videos'):
        video = output.videos[0]
        print("Debug - Using output.videos[0]")
    else:
        video = output[0]
        print("Debug - Using output[0]")

    print(f"Debug - Video type: {type(video)}")

    # Handle list format
    if isinstance(video, list):
        print(f"Debug - Converting list of {len(video)} frames")
        video = np.array(video)

    # Convert tensor to numpy
    if isinstance(video, torch.Tensor):
        print(f"Debug - Converting tensor, shape: {video.shape}, dtype: {video.dtype}")
        video = video.cpu().numpy()

    print(f"Debug - Final numpy shape: {video.shape}, dtype: {video.dtype}")
    print(f"Debug - Value range: [{video.min():.3f}, {video.max():.3f}]")

    # Normalize to uint8
    if video.dtype != np.uint8:
        if video.max() > 1.0:
            print("Debug - Normalizing from 0-255 range")
            video = video / 255.0
        else:
            print("Debug - Already in 0-1 range")
        video = (video.clip(0, 1) * 255).astype(np.uint8)

    print(f"Debug - After uint8 conversion: shape={video.shape}, range=[{video.min()}, {video.max()}]")

    # Save
    print(f"\nSaving to {args.output}...")
    imageio.mimsave(args.output, video, fps=args.fps)

    print(f"\n✓ Done!")
    print(f"  Duration: {args.num_frames / args.fps:.2f}s")
    print(f"  Resolution: {args.width}x{args.height}")
    print("="*60)


if __name__ == "__main__":
    main()
