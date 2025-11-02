#!/usr/bin/env python3
"""
Test with proper guidance scale based on official config.

Official config uses progressive guidance: [1, 1, 6, 8, 6, 1, 1]
Diffusers doesn't support array guidance, so we'll use average: ~4.14
"""

import argparse
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel, LTXConditionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
import imageio
import numpy as np
from pathlib import Path


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    """Calculate mu for dynamic timestep shifting."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def test_with_official_guidance(lora_path=None, args=None):
    """Test with guidance scale closer to official config."""

    model_name = "BASE" if not lora_path else "LORA"
    print("="*60)
    print(f"Testing: {model_name}")
    print("="*60)

    # Load components
    print("\nLoading components...")
    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained("Lightricks/LTX-Video", subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLLTXVideo.from_pretrained("Lightricks/LTX-Video", subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = LTXVideoTransformer3DModel.from_pretrained("Lightricks/LTX-Video", subfolder="transformer", torch_dtype=torch.bfloat16)

    # Load LoRA if provided
    if lora_path:
        print(f"Loading LoRA: {Path(lora_path).name}...")
        lora_state_dict = load_file(lora_path)
        missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
        print(f"  ✓ LoRA loaded: {len(lora_state_dict)} keys")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Lightricks/LTX-Video", subfolder="scheduler")

    # Move to GPU
    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    # Create pipeline
    pipe = LTXConditionPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # Configure scheduler
    latent_height = args.height // 8
    latent_width = args.width // 8
    image_seq_len = args.num_frames * latent_height * latent_width
    mu = calculate_shift(image_seq_len)

    if hasattr(pipe.scheduler.config, 'use_dynamic_shifting'):
        pipe.scheduler.config.use_dynamic_shifting = False

    pipe.vae.enable_tiling()

    # Official config uses progressive guidance: [1, 1, 6, 8, 6, 1, 1]
    # Average: (1+1+6+8+6+1+1)/7 = 24/7 ≈ 3.43
    # But peak is 8, so let's try different values

    guidance = args.guidance_scale
    print(f"\nGeneration settings:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Guidance scale: {guidance}")
    print(f"  NOTE: Official config uses progressive [1,1,6,8,6,1,1]")
    print(f"        Diffusers doesn't support this, using constant {guidance}")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.inference_mode():
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=guidance,
            generator=generator,
            decode_timestep=0.05,
            image_cond_noise_scale=0.025,
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
    print(f"  RGB means: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
    print(f"  Green ratio: {green_ratio:.2f}")

    if green_ratio > 1.5:
        print(f"  ⚠️  GREEN TINT")
    elif green_ratio > 1.2:
        print(f"  ⚠️  Slight green tint")
    else:
        print(f"  ✓ Colors balanced")

    return video, r_mean, g_mean, b_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str,
                       default="output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors")
    parser.add_argument("--prompt", type=str,
                       default="A scenic mountain landscape, camera slowly panning right")
    parser.add_argument("--negative_prompt", type=str,
                       default="worst quality, inconsistent motion, blurry")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                       help="Official uses progressive [1,1,6,8,6,1,1], try 3.5 or 5.0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("="*60)
    print("BASE MODEL vs LoRA with Proper Guidance")
    print("="*60)
    print(f"Testing guidance_scale={args.guidance_scale}")
    print("(Official config uses progressive guidance)")
    print("="*60)

    # Test base
    print("\n### TEST 1: BASE MODEL ###")
    video_base, r_base, g_base, b_base = test_with_official_guidance(None, args)
    print(f"\nSaving to guidance_base.mp4...")
    imageio.mimsave("guidance_base.mp4", video_base, fps=args.fps)
    print("  ✓ Saved")

    # Clear memory
    torch.cuda.empty_cache()

    # Test LoRA
    print("\n### TEST 2: WITH LORA ###")
    video_lora, r_lora, g_lora, b_lora = test_with_official_guidance(args.lora_path, args)
    print(f"\nSaving to guidance_lora.mp4...")
    imageio.mimsave("guidance_lora.mp4", video_lora, fps=args.fps)
    print("  ✓ Saved")

    # Compare
    diff = abs(r_base - r_lora) + abs(g_base - g_lora) + abs(b_base - b_lora)
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"BASE:  R={r_base:.1f}, G={g_base:.1f}, B={b_base:.1f}")
    print(f"LORA:  R={r_lora:.1f}, G={g_lora:.1f}, B={b_lora:.1f}")
    print(f"\nRGB difference: {diff:.1f}")

    if diff < 5:
        print("⚠️  Videos appear IDENTICAL - LoRA not activating")
        print("\nPossible reasons:")
        print("  1. IC-LoRA trained for trajectory control needs reference videos")
        print("  2. Diffusers doesn't fully support IC-LoRA conditioning")
        print("  3. Need to use official LTX-Video inference (not diffusers)")
    else:
        print("✓✓✓ Videos are DIFFERENT - LoRA is working!")
    print("="*60)


if __name__ == "__main__":
    main()
