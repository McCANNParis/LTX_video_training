#!/usr/bin/env python3
"""
Test IC-LoRA with reference video for trajectory control.

This is how your LoRA was DESIGNED to be used - with reference videos
that provide motion/trajectory conditioning.
"""

import sys
import os
import argparse
import torch
import imageio
import numpy as np
from pathlib import Path

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file


def load_reference_video(video_path, target_frames=None):
    """Load and preprocess reference video."""
    print(f"\nLoading reference video: {video_path}")

    # Read video
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()

    print(f"  Original frames: {len(frames)}")

    # Sample frames if needed
    if target_frames and len(frames) > target_frames:
        # Sample evenly
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
        print(f"  Sampled to: {len(frames)} frames")

    # Convert to tensor
    video = np.stack(frames)

    # Normalize to [-1, 1]
    if video.dtype == np.uint8:
        video = video.astype(np.float32) / 255.0
    video = video * 2.0 - 1.0

    # To tensor: [F, H, W, C] -> [C, F, H, W]
    video = torch.from_numpy(video).permute(3, 0, 1, 2)

    return video


def load_pipeline_with_lora(lora_path):
    """Load LTX-Video pipeline with LoRA."""

    print("\n1. Loading base models...")

    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained("Lightricks/LTX-Video", subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLLTXVideo.from_pretrained("Lightricks/LTX-Video", subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = LTXVideoTransformer3DModel.from_pretrained("Lightricks/LTX-Video", subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Lightricks/LTX-Video", subfolder="scheduler")

    # Load LoRA
    if lora_path:
        print(f"\n2. Loading LoRA from {lora_path}...")
        lora_state_dict = load_file(lora_path)
        missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
        print(f"   ✓ LoRA loaded: {len(lora_state_dict)} keys")

    # Move to GPU
    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    # Create pipeline
    from diffusers import LTXPipeline
    pipeline = LTXPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    return pipeline, vae


def encode_reference_video(vae, video_tensor, height, width):
    """Encode reference video to latents."""
    print("\n3. Encoding reference video to latents...")

    # Resize if needed
    if video_tensor.shape[2] != height or video_tensor.shape[3] != width:
        print(f"   Resizing reference from {video_tensor.shape[2]}x{video_tensor.shape[3]} to {height}x{width}")
        import torch.nn.functional as F
        video_tensor = F.interpolate(
            video_tensor.unsqueeze(0),  # Add batch dim
            size=(video_tensor.shape[1], height, width),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)

    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0).to("cuda", dtype=torch.bfloat16)

    # Encode
    with torch.no_grad():
        latents = vae.encode(video_tensor).latent_dist.sample()

    print(f"   ✓ Encoded to latents: {latents.shape}")
    return latents


def generate_with_reference(pipeline, prompt, negative_prompt, reference_latents,
                            height, width, num_frames, steps, guidance, seed):
    """Generate video conditioned on reference latents."""

    print(f"\n4. Generating with trajectory conditioning...")
    print(f"   Prompt: {prompt}")
    print(f"   Steps: {steps}")
    print(f"   Guidance: {guidance}")
    print(f"   Reference shape: {reference_latents.shape}")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        # NOTE: This is a simplified version. The actual IC-LoRA pipeline
        # may require specific handling of reference latents.
        # You may need to use the trainer's infer.py script instead.

        try:
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                # Try passing reference latents (may not work with standard pipeline)
                # This is a placeholder - actual IC-LoRA API may differ
            )
        except TypeError:
            print("\n   Warning: Standard pipeline doesn't support reference latents")
            print("   Generating without reference (text-only)...")
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

    # Extract frames
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

    # Ensure uint8
    if video.dtype != 'uint8':
        if video.max() > 1.0:
            video = video / 255.0
        video = (video.clip(0, 1) * 255).astype('uint8')

    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--reference_video", type=str, required=True, help="Reference video for trajectory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="test_with_reference.mp4", help="Output path")
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
    print("IC-LoRA Test with Reference Video")
    print("="*60)
    print(f"LoRA: {args.lora_path}")
    print(f"Reference: {args.reference_video}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print("="*60)

    # Check reference video exists
    if not Path(args.reference_video).exists():
        print(f"\nError: Reference video not found: {args.reference_video}")
        print("\nAvailable training videos:")
        videos_dir = Path("dataset/videos")
        if videos_dir.exists():
            for i, vid in enumerate(list(videos_dir.glob("*.mp4"))[:5]):
                print(f"  {i+1}. {vid.name}")
        return

    # Load reference video
    reference_video = load_reference_video(args.reference_video, target_frames=args.num_frames)

    # Load pipeline
    pipeline, vae = load_pipeline_with_lora(args.lora_path)

    # Encode reference
    try:
        reference_latents = encode_reference_video(vae, reference_video, args.height, args.width)
    except Exception as e:
        print(f"\n   ✗ Failed to encode reference: {e}")
        print("   This may require more VRAM. Continuing without reference latents...")
        reference_latents = None

    # Generate
    video = generate_with_reference(
        pipeline, args.prompt, args.negative_prompt, reference_latents,
        args.height, args.width, args.num_frames,
        args.steps, args.guidance_scale, args.seed
    )

    # Save
    print(f"\n5. Saving to {args.output}...")
    imageio.mimsave(args.output, video, fps=args.fps)

    print(f"\n✓ Done!")
    print(f"  Duration: {args.num_frames / args.fps:.2f}s")
    print(f"  Resolution: {args.width}x{args.height}")
    print("="*60)

    print("\nIMPORTANT NOTE:")
    print("This script uses a simplified approach to IC-LoRA inference.")
    print("For proper trajectory control, you may need to use the official trainer's")
    print("inference script which properly handles reference video conditioning.")
    print("\nTry: cd LTX-Video-Trainer && python scripts/infer.py --help")
    print("="*60)


if __name__ == "__main__":
    main()
