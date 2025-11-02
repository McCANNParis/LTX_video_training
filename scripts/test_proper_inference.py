#!/usr/bin/env python3
"""
Test LTX-Video with PROPER settings from official documentation.

Key fixes:
1. Use LTXConditionPipeline (not LTXPipeline)
2. guidance_scale=1.0 for distilled model
3. decode_timestep=0.05, image_cond_noise_scale=0.025 for 0.9+ models
4. Enable VAE tiling
"""

import argparse
import torch
from diffusers import LTXConditionPipeline
from diffusers.utils import export_to_video
import imageio
import numpy as np


def test_proper_inference(args):
    """Test with proper pipeline and settings."""

    print("="*60)
    print("LTX-Video PROPER Inference Test")
    print("="*60)
    print("Using official documentation settings:")
    print("  - LTXConditionPipeline")
    print(f"  - guidance_scale=1.0 (distilled model)")
    print("  - decode_timestep=0.05")
    print("  - image_cond_noise_scale=0.025")
    print("  - VAE tiling enabled")
    print("="*60)

    # Load pipeline properly
    print("\nLoading LTXConditionPipeline...")
    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    # Enable VAE tiling (from docs)
    print("Enabling VAE tiling...")
    pipe.vae.enable_tiling()

    # Generate
    print(f"\nGenerating video...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}")
    print(f"  Guidance: 1.0 (CORRECTED from 3.5)")
    print(f"  Seed: {args.seed}")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.inference_mode():
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=1.0,  # CORRECTED: 1.0 for distilled models
            generator=generator,
            decode_timestep=0.05,  # NEW: for VAE 0.9+
            image_cond_noise_scale=0.025,  # NEW: for VAE 0.9+
            mu=0.3,  # NEW: Required for dynamic shifting in scheduler
        )

    # Extract frames
    if hasattr(output, 'frames'):
        video = output.frames[0]
    elif hasattr(output, 'videos'):
        video = output.videos[0]
    else:
        video = output[0]

    if isinstance(video, list):
        video = np.array(video)

    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()

    if video.dtype != np.uint8:
        if video.max() > 1.0:
            video = video / 255.0
        video = (video.clip(0, 1) * 255).astype(np.uint8)

    # Analyze
    r_mean = video[:,:,:,0].mean()
    g_mean = video[:,:,:,1].mean()
    b_mean = video[:,:,:,2].mean()
    green_ratio = g_mean / ((r_mean + b_mean) / 2)

    print(f"\nResults:")
    print(f"  Shape: {video.shape}")
    print(f"  RGB means: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
    print(f"  Green ratio: {green_ratio:.2f}")

    if green_ratio > 1.5:
        print(f"  ⚠️  GREEN TINT STILL PRESENT")
    elif green_ratio > 1.2:
        print(f"  ⚠️  Slight green tint")
    else:
        print(f"  ✓✓✓ COLORS BALANCED - GREEN TINT FIXED!")

    # Save
    print(f"\nSaving to {args.output}...")
    imageio.mimsave(args.output, video, fps=args.fps)

    print(f"\n✓ Done!")
    print("="*60)

    return video, r_mean, g_mean, b_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str,
                       default="A scenic mountain landscape, camera slowly panning right")
    parser.add_argument("--output", type=str, default="test_proper_settings.mp4")
    parser.add_argument("--negative_prompt", type=str,
                       default="worst quality, inconsistent motion, blurry")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    test_proper_inference(args)


if __name__ == "__main__":
    main()
