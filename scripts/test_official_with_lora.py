#!/usr/bin/env python3
"""
Test base model vs LoRA using OFFICIAL LTX-Video inference code.
Uses LTXVideoCondition for image-to-video generation.

Based on: https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled
"""

import argparse
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from diffusers import LTXVideoTransformer3DModel
from safetensors.torch import load_file
from pathlib import Path


def round_to_nearest_resolution_acceptable_by_vae(pipe, height, width):
    """Round resolution to be acceptable by VAE."""
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width


def load_lora_into_transformer(transformer, lora_path):
    """Load LoRA weights into transformer."""
    print(f"  Loading LoRA: {Path(lora_path).name}...")
    lora_state_dict = load_file(lora_path)
    missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
    print(f"    ✓ LoRA loaded: {len(lora_state_dict)} keys")
    if unexpected:
        print(f"    Note: {len(unexpected)} unexpected keys (normal for LoRA)")
    return transformer


def test_with_official_inference(lora_path=None, args=None):
    """Test using official LTX-Video inference code."""

    model_name = "BASE MODEL" if not lora_path else "WITH LORA"
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Load pipelines
    print("\nLoading LTXConditionPipeline...")

    if lora_path:
        # Load transformer separately to apply LoRA
        print("  Loading base transformer...")
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )

        # Apply LoRA
        transformer = load_lora_into_transformer(transformer, lora_path)

        # Load pipeline with LoRA'd transformer
        pipe = LTXConditionPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
    else:
        # Load base pipeline
        pipe = LTXConditionPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16
        )

    print("Loading upsampler...")
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        "Lightricks/ltxv-spatial-upscaler-0.9.8",
        vae=pipe.vae,
        torch_dtype=torch.bfloat16
    )

    print("Moving to GPU...")
    pipe.to("cuda")
    pipe_upsample.to("cuda")

    print("Enabling VAE tiling...")
    pipe.vae.enable_tiling()

    # Prepare conditioning
    print("\nPreparing conditioning image...")
    if args.conditioning_image:
        print(f"  Loading image: {args.conditioning_image}")
        image = load_image(args.conditioning_image)
    else:
        print("  Using default penguin image")
        image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png")

    # Compress image using video compression (as model was trained on videos)
    video = load_video(export_to_video([image]))
    condition1 = LTXVideoCondition(video=video, frame_index=0)

    # Settings
    downscale_factor = 2 / 3
    downscaled_height, downscaled_width = int(args.height * downscale_factor), int(args.width * downscale_factor)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(pipe, downscaled_height, downscaled_width)

    print(f"\nGeneration settings:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Downscaled resolution: {downscaled_width}x{downscaled_height}")
    print(f"  Final resolution: {args.width}x{args.height}")
    print(f"  Num frames: {args.num_frames}")
    print(f"  Seed: {args.seed}")

    generator = torch.Generator().manual_seed(args.seed)

    # Part 1. Generate video at smaller resolution
    print("\n[1/4] Generating at lower resolution...")
    latents = pipe(
        conditions=[condition1],
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=generator,
        output_type="latent",
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
    ).frames
    print("  ✓ Done")

    # Part 2. Upscale generated video using latent upsampler
    print("\n[2/4] Upscaling latents (2x)...")
    upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
    upscaled_latents = pipe_upsample(
        latents=latents,
        output_type="latent"
    ).frames
    print("  ✓ Done")

    # Part 3. Denoise the upscaled video with few steps
    print("\n[3/4] Denoising upscaled video...")
    video = pipe(
        conditions=[condition1],
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=upscaled_width,
        height=upscaled_height,
        num_frames=args.num_frames,
        denoise_strength=0.4,  # 4 inference steps out of 10
        num_inference_steps=10,
        latents=upscaled_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=generator,
        output_type="pil",
    ).frames[0]
    print("  ✓ Done")

    # Part 4. Downscale to expected resolution
    print(f"\n[4/4] Downscaling to {args.width}x{args.height}...")
    video = [frame.resize((args.width, args.height)) for frame in video]
    print("  ✓ Done")

    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str,
                       default="output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors")
    parser.add_argument("--conditioning_image", type=str, default=None,
                       help="Path to conditioning image (uses penguin if not specified)")
    parser.add_argument("--prompt", type=str,
                       default="A cinematic landscape with beautiful lighting and smooth camera movement")
    parser.add_argument("--negative_prompt", type=str,
                       default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("="*60)
    print("OFFICIAL LTX-Video Inference Test")
    print("Base Model vs LoRA Comparison")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} @ {args.fps}fps")
    print(f"LoRA: {args.lora_path}")
    print("="*60)

    # Test 1: Base model
    print("\n\n" + "="*60)
    print("TEST 1: BASE MODEL")
    print("="*60)
    video_base = test_with_official_inference(lora_path=None, args=args)
    output_base = "official_base.mp4"
    print(f"\nSaving to {output_base}...")
    export_to_video(video_base, output_base, fps=args.fps)
    print(f"✓ Saved: {output_base}")

    # Clear memory
    torch.cuda.empty_cache()

    # Test 2: With LoRA
    print("\n\n" + "="*60)
    print("TEST 2: WITH LORA")
    print("="*60)
    video_lora = test_with_official_inference(lora_path=args.lora_path, args=args)
    output_lora = "official_lora.mp4"
    print(f"\nSaving to {output_lora}...")
    export_to_video(video_lora, output_lora, fps=args.fps)
    print(f"✓ Saved: {output_lora}")

    # Summary
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"Base model output: {output_base}")
    print(f"LoRA output:       {output_lora}")
    print("\nCompare the two videos to see if LoRA is working!")
    print("If they look different, your LoRA is affecting the output.")
    print("="*60)


if __name__ == "__main__":
    main()
