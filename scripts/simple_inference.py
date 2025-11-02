#!/usr/bin/env python3
"""
Simple inference using the official LTX-Video-Trainer pipeline code.
This works with the same code that was used for training.
"""

import sys
import os
import argparse
import torch

# Add the trainer to path - try to find it
trainer_paths = [
    '/workspace/LTX_video_training/LTX-Video-Trainer',
    os.path.join(os.path.dirname(__file__), '..', 'LTX-Video-Trainer')
]

for path in trainer_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

# Try importing, if fails use diffusers directly
try:
    from ltxv_trainer.ltxv_pipeline import LTXVPipeline
    USE_TRAINER_PIPELINE = True
except ImportError:
    print("Warning: Could not import trainer pipeline, using diffusers...")
    from diffusers import LTXPipeline as LTXVPipeline
    USE_TRAINER_PIPELINE = False
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
import imageio


def load_model_with_lora(lora_path):
    """Load LTX-Video model and apply LoRA weights."""

    print("\n1. Loading base models...")

    # Load tokenizer and text encoder
    tokenizer = T5Tokenizer.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    )

    # Load VAE
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )

    # Load transformer
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="scheduler"
    )

    print("\n2. Loading LoRA weights...")
    if lora_path:
        try:
            lora_state_dict = load_file(lora_path)

            # Load LoRA weights into transformer
            missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)

            print(f"   ✓ Loaded LoRA weights")
            print(f"   - Loaded keys: {len(lora_state_dict)}")
            if missing:
                print(f"   - Missing keys: {len(missing)}")
            if unexpected:
                print(f"   - Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"   ✗ Failed to load LoRA: {e}")
            print("   Continuing with base model...")

    # Move to GPU
    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    # Create pipeline
    pipeline = LTXVPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.mp4")
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
    print("LTX-Video Inference with LoRA")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"LoRA: {args.lora_path}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.width}x{args.height}x{args.num_frames}")
    print("="*60)

    # Load model
    pipeline = load_model_with_lora(args.lora_path)

    # Set seed
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Generate
    print(f"\n3. Generating video...")
    print(f"   Steps: {args.steps}")
    print(f"   Guidance: {args.guidance_scale}")
    print(f"   Seed: {args.seed}")

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

    # Extract frames
    video = output.frames[0] if hasattr(output, 'frames') else output.videos[0]

    # Convert to numpy if needed
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()

    # Ensure uint8
    if video.dtype != 'uint8':
        video = (video.clip(0, 1) * 255).astype('uint8')

    # Save
    print(f"\n4. Saving to {args.output}...")
    imageio.mimsave(args.output, video, fps=args.fps)

    print(f"✓ Done!")
    print(f"  Duration: {args.num_frames / args.fps:.2f}s")
    print(f"  Resolution: {args.width}x{args.height}")
    print("="*60)


if __name__ == "__main__":
    main()
