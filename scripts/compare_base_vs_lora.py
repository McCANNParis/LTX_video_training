#!/usr/bin/env python3
"""
Compare base model vs LoRA side-by-side.
Generates the same prompt with and without LoRA to see the difference.
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, '/workspace/LTX_video_training/LTX-Video-Trainer')

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
import imageio
import numpy as np


def load_pipeline(lora_path=None):
    """Load LTX-Video pipeline, optionally with LoRA."""

    print(f"\nLoading {'base model' if not lora_path else 'model with LoRA'}...")

    from diffusers import LTXPipeline

    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained("Lightricks/LTX-Video", subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLLTXVideo.from_pretrained("Lightricks/LTX-Video", subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = LTXVideoTransformer3DModel.from_pretrained("Lightricks/LTX-Video", subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Lightricks/LTX-Video", subfolder="scheduler")

    if lora_path:
        print(f"   Loading LoRA from {lora_path}...")
        lora_state_dict = load_file(lora_path)
        missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
        print(f"   ✓ LoRA loaded: {len(lora_state_dict)} keys")

    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    from ltxv_trainer.ltxv_pipeline import LTXVPipeline
    return LTXVPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )


def generate_video(pipeline, prompt, negative_prompt, height, width, num_frames, steps, guidance, seed):
    """Generate a video."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
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

    if isinstance(video, list):
        video = np.array(video)

    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()

    if video.dtype != 'uint8':
        if video.max() > 1.0:
            video = video / 255.0
        video = (video.clip(0, 1) * 255).astype('uint8')

    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="worst quality, inconsistent motion, blurry")
    parser.add_argument("--output_base", type=str, default="comparison_base.mp4")
    parser.add_argument("--output_lora", type=str, default="comparison_lora.mp4")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("="*60)
    print("Base Model vs LoRA Comparison")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"LoRA: {args.lora_path}")
    print(f"Resolution: {args.width}x{args.height}x{args.num_frames}")
    print("="*60)

    # Generate with base model
    print("\n1. Generating with BASE MODEL...")
    pipeline_base = load_pipeline(lora_path=None)
    video_base = generate_video(
        pipeline_base, args.prompt, args.negative_prompt,
        args.height, args.width, args.num_frames,
        args.steps, args.guidance_scale, args.seed
    )
    print(f"   Saving to {args.output_base}...")
    imageio.mimsave(args.output_base, video_base, fps=args.fps)
    print(f"   ✓ Base model done")

    # Clear memory
    del pipeline_base
    torch.cuda.empty_cache()

    # Generate with LoRA
    print("\n2. Generating with LORA...")
    pipeline_lora = load_pipeline(lora_path=args.lora_path)
    video_lora = generate_video(
        pipeline_lora, args.prompt, args.negative_prompt,
        args.height, args.width, args.num_frames,
        args.steps, args.guidance_scale, args.seed
    )
    print(f"   Saving to {args.output_lora}...")
    imageio.mimsave(args.output_lora, video_lora, fps=args.fps)
    print(f"   ✓ LoRA done")

    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)
    print(f"Base model: {args.output_base}")
    print(f"With LoRA:  {args.output_lora}")
    print("\nCompare these videos to see the LoRA's effect.")
    print("If they look the same, the LoRA may not be working properly.")
    print("="*60)


if __name__ == "__main__":
    main()
