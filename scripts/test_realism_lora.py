#!/usr/bin/env python3
"""
Test Realism LoRA - Simple text-to-video inference.

This script tests the realism LoRA trained WITHOUT trajectory conditioning.
It should work with text prompts alone and improve video realism.
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


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate mu for dynamic timestep shifting."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def load_pipeline(lora_path=None, args=None):
    """Load LTXConditionPipeline with optional LoRA."""

    print(f"\n{'='*60}")
    print(f"Loading {'BASE MODEL' if not lora_path else 'REALISM LORA MODEL'}")
    print(f"{'='*60}")

    # Load components separately to apply LoRA to transformer
    print("  Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")

    print("  Loading text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    )

    print("  Loading VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )

    print("  Loading transformer...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # Load LoRA if provided
    if lora_path:
        print(f"  Loading Realism LoRA weights...")
        lora_state_dict = load_file(lora_path)
        missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
        print(f"    ✓ LoRA loaded: {len(lora_state_dict)} keys")
        if unexpected:
            print(f"    Note: {len(unexpected)} unexpected keys (normal for LoRA)")

    print("  Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="scheduler"
    )

    # Move to GPU
    print("  Moving to GPU...")
    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    # Create pipeline
    print("  Creating LTXConditionPipeline...")
    pipe = LTXConditionPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # Calculate mu and disable dynamic shifting
    latent_height = args.height // 8
    latent_width = args.width // 8
    image_seq_len = args.num_frames * latent_height * latent_width
    mu = calculate_shift(image_seq_len)
    print(f"  Calculated mu: {mu:.4f}")

    if hasattr(pipe.scheduler.config, 'use_dynamic_shifting'):
        pipe.scheduler.config.use_dynamic_shifting = False

    # Enable VAE tiling
    print("  Enabling VAE tiling...")
    pipe.vae.enable_tiling()

    return pipe


def generate_video(pipe, args):
    """Generate video with proper settings."""

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"\n  Generating...")
    print(f"    Prompt: {args.prompt}")
    print(f"    Steps: {args.steps}")
    print(f"    Guidance: 3.5")

    with torch.inference_mode():
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=3.5,  # Higher guidance for text-only generation
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

    return video


def analyze_video(video, name):
    """Analyze and print video statistics."""

    r_mean = video[:,:,:,0].mean()
    g_mean = video[:,:,:,1].mean()
    b_mean = video[:,:,:,2].mean()
    green_ratio = g_mean / ((r_mean + b_mean) / 2)

    print(f"\n  Results for {name}:")
    print(f"    RGB means: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
    print(f"    Green ratio: {green_ratio:.2f}")

    if green_ratio > 1.5:
        print(f"    ⚠️  GREEN TINT")
    elif green_ratio > 1.2:
        print(f"    ⚠️  Slight green tint")
    else:
        print(f"    ✓ Colors balanced")

    return r_mean, g_mean, b_mean


def main():
    parser = argparse.ArgumentParser(description="Test Realism LoRA with text prompts")
    parser.add_argument("--lora_path", type=str,
                       default="output/realism_lora/checkpoints/lora_weights_step_05000.safetensors",
                       help="Path to LoRA checkpoint")
    parser.add_argument("--prompt", type=str,
                       default="A person walking in a park, photorealistic, high detail, natural lighting",
                       help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str,
                       default="worst quality, low quality, blurry, distorted, unrealistic, cartoon",
                       help="Negative prompt")
    parser.add_argument("--output", type=str, default="realism_lora_test.mp4",
                       help="Output video filename")
    parser.add_argument("--compare", action="store_true",
                       help="Compare base model vs LoRA side-by-side")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Check if LoRA file exists
    if not Path(args.lora_path).exists():
        print(f"Error: LoRA checkpoint not found at {args.lora_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path("output/realism_lora/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("lora_weights_step_*.safetensors"))
            for ckpt in checkpoints:
                print(f"  - {ckpt}")
        else:
            print("  No checkpoints found. Run training first:")
            print("    ./scripts/train_realism_lora.sh")
        return

    print("="*60)
    print("REALISM LORA TEST")
    print("Text-to-Video Generation")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"LoRA: {args.lora_path}")
    print("="*60)

    results = []

    if args.compare:
        # Test 1: Base model
        print("\n\n### TEST 1: BASE MODEL ###")
        pipe_base = load_pipeline(lora_path=None, args=args)
        video_base = generate_video(pipe_base, args)
        r_base, g_base, b_base = analyze_video(video_base, "BASE MODEL")
        results.append(("BASE", r_base, g_base, b_base))

        output_base = args.output.replace(".mp4", "_base.mp4")
        print(f"  Saving to {output_base}...")
        imageio.mimsave(output_base, video_base, fps=args.fps)
        print(f"  ✓ Saved")

        # Clear memory
        del pipe_base
        torch.cuda.empty_cache()

        # Test 2: With LoRA
        print("\n\n### TEST 2: WITH REALISM LORA ###")
        pipe_lora = load_pipeline(lora_path=args.lora_path, args=args)
        video_lora = generate_video(pipe_lora, args)
        r_lora, g_lora, b_lora = analyze_video(video_lora, "WITH LORA")
        results.append(("LORA", r_lora, g_lora, b_lora))

        output_lora = args.output.replace(".mp4", "_lora.mp4")
        print(f"  Saving to {output_lora}...")
        imageio.mimsave(output_lora, video_lora, fps=args.fps)
        print(f"  ✓ Saved")

        # Summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)

        for name, r, g, b in results:
            green_ratio = g / ((r + b) / 2)
            color_status = "✓ OK" if green_ratio < 1.2 else f"⚠️ GREEN ({green_ratio:.2f}x)"
            print(f"{name:12s}: R={r:5.1f} G={g:5.1f} B={b:5.1f} - {color_status}")

        # Compare base vs LoRA
        diff = abs(r_base - r_lora) + abs(g_base - g_lora) + abs(b_base - b_lora)

        print("\n" + "="*60)
        print("BASE vs LoRA Difference")
        print("="*60)
        print(f"RGB difference: {diff:.1f}")

        if diff < 5:
            print("⚠️  Videos appear similar")
            print("    Try training for more steps or adjusting LoRA rank/alpha")
        else:
            print(f"✓ Videos are DIFFERENT!")
            print("    Realism LoRA is affecting the output!")

        print("\nOutput files:")
        print(f"  - {output_base} (base model)")
        print(f"  - {output_lora} (with realism LoRA)")
        print("="*60)

    else:
        # Just test with LoRA
        print("\n### GENERATING WITH REALISM LORA ###")
        pipe_lora = load_pipeline(lora_path=args.lora_path, args=args)
        video_lora = generate_video(pipe_lora, args)
        analyze_video(video_lora, "REALISM LORA")

        print(f"\n  Saving to {args.output}...")
        imageio.mimsave(args.output, video_lora, fps=args.fps)
        print(f"  ✓ Saved")

        print("\nOutput file:")
        print(f"  - {args.output}")
        print("="*60)


if __name__ == "__main__":
    main()
