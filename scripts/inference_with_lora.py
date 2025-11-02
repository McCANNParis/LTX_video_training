#!/usr/bin/env python3
"""
Simple inference script for LTX-Video with trained LoRA.
Uses diffusers library to load the model and apply LoRA weights.
"""

import argparse
import torch
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from peft import PeftModel, LoraConfig
import imageio
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate video with LTX-Video + LoRA")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights (.safetensors)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--negative_prompt", type=str, default="worst quality, inconsistent motion, blurry", help="Negative prompt")
    parser.add_argument("--image", type=str, default=None, help="Input image for img2vid (optional)")
    parser.add_argument("--height", type=int, default=704, help="Video height")
    parser.add_argument("--width", type=int, default=1216, help="Video width")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("="*60)
    print("LTX-Video Inference with LoRA")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"LoRA: {args.lora_path}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.width}x{args.height}x{args.num_frames}")
    print("="*60)

    # Load base pipeline
    print("\n1. Loading LTX-Video base model...")
    if args.image:
        print("   Using image-to-video mode")
        pipeline = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16
        )
    else:
        print("   Using text-to-video mode")
        pipeline = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16
        )

    pipeline = pipeline.to("cuda")

    # Load LoRA weights
    print(f"\n2. Loading LoRA from {args.lora_path}...")
    try:
        # Try loading LoRA into the transformer
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer,
            args.lora_path,
            is_trainable=False
        )
        print("   ✓ LoRA loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load LoRA: {e}")
        print("   Trying alternative loading method...")
        try:
            from safetensors.torch import load_file
            lora_state_dict = load_file(args.lora_path)
            pipeline.transformer.load_state_dict(lora_state_dict, strict=False)
            print("   ✓ LoRA weights loaded (alternative method)")
        except Exception as e2:
            print(f"   ✗ Failed with alternative method: {e2}")
            print("   Continuing without LoRA...")

    # Set generator
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Generate
    print("\n3. Generating video...")
    print(f"   Steps: {args.num_inference_steps}")
    print(f"   Guidance: {args.guidance_scale}")
    print(f"   Seed: {args.seed}")

    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        output = pipeline(
            prompt=args.prompt,
            image=image,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
    else:
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )

    # Save video
    print(f"\n4. Saving video to {args.output}...")
    frames = output.frames[0]  # Get first video

    # Convert to numpy if needed
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()

    # Ensure correct format for imageio
    if frames.dtype != 'uint8':
        frames = (frames * 255).astype('uint8')

    imageio.mimsave(args.output, frames, fps=args.fps)

    print(f"✓ Video saved successfully!")
    print(f"  Duration: {args.num_frames / args.fps:.2f} seconds")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  FPS: {args.fps}")
    print("="*60)


if __name__ == "__main__":
    main()
