#!/usr/bin/env python3
"""
Test using component-based loading (same as original working script).
This loads each component separately instead of using from_pretrained on the full pipeline.
"""

import argparse
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel, LTXPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="test_components.mp4")
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
    print("Testing with Component-Based Loading")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.width}x{args.height}x{args.num_frames}")
    print("="*60)

    # Load components separately (like the working script)
    print("\nLoading components...")

    print("  - Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")

    print("  - Text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    )

    print("  - VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )

    print("  - Transformer...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    print("  - Scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="scheduler"
    )

    # Move to GPU
    print("\nMoving to GPU...")
    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    # Create pipeline from components
    print("Creating pipeline...")
    pipeline = LTXPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

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

    # Extract frames (same as working script)
    if hasattr(output, 'frames'):
        video = output.frames[0]
    elif hasattr(output, 'videos'):
        video = output.videos[0]
    else:
        video = output[0]

    # Handle list format
    if isinstance(video, list):
        video = np.array(video)

    # Convert tensor to numpy
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()

    # Normalize
    if video.dtype != np.uint8:
        if video.max() > 1.0:
            video = video / 255.0
        video = (video.clip(0, 1) * 255).astype(np.uint8)

    print(f"\nVideo stats:")
    print(f"  Shape: {video.shape}")
    print(f"  Dtype: {video.dtype}")
    print(f"  Range: [{video.min()}, {video.max()}]")
    print(f"  Mean per channel: R={video[:,:,:,0].mean():.1f}, G={video[:,:,:,1].mean():.1f}, B={video[:,:,:,2].mean():.1f}")

    # Save
    print(f"\nSaving to {args.output}...")
    imageio.mimsave(args.output, video, fps=args.fps)

    print(f"\n✓ Done!")
    print(f"  Duration: {args.num_frames / args.fps:.2f}s")
    print(f"  Resolution: {args.width}x{args.height}")
    print("="*60)


if __name__ == "__main__":
    main()
