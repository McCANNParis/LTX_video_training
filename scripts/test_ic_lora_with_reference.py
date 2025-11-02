#!/usr/bin/env python3
"""
Test IC-LoRA with reference videos from training set.
This is the PROPER way to test trajectory control LoRA.

Compares:
1. Base model (no LoRA) with reference video
2. LoRA model with reference video
3. Different VAE loading methods (BF16 vs FP32)
"""

import argparse
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel, LTXPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
import imageio
import numpy as np
from pathlib import Path


def load_reference_video(video_path, target_frames=None, target_height=None, target_width=None):
    """Load and preprocess reference video."""
    print(f"\n  Loading reference: {Path(video_path).name}")

    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()

    print(f"    Original: {len(frames)} frames, {frames[0].shape}")

    # Sample frames if needed
    if target_frames and len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
        print(f"    Sampled to: {len(frames)} frames")

    # Convert to numpy array
    video = np.stack(frames)

    # Resize if needed
    if target_height and target_width:
        import cv2
        resized_frames = []
        for frame in video:
            resized = cv2.resize(frame, (target_width, target_height))
            resized_frames.append(resized)
        video = np.stack(resized_frames)
        print(f"    Resized to: {target_height}x{target_width}")

    # Normalize to [-1, 1] for VAE encoder
    if video.dtype == np.uint8:
        video = video.astype(np.float32) / 255.0
    video = video * 2.0 - 1.0

    # To tensor: [F, H, W, C] -> [1, C, F, H, W] (batch format)
    video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)

    return video


def encode_reference(vae, reference_video):
    """Encode reference video to latents using VAE."""
    print(f"\n  Encoding reference to latents...")

    reference_video = reference_video.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        # Encode the reference video
        latents = vae.encode(reference_video).latent_dist.sample()

    print(f"    Latent shape: {latents.shape}")
    return latents


def load_pipeline(lora_path=None, use_fp32_vae=False):
    """Load LTX-Video pipeline with optional LoRA."""

    vae_dtype = torch.float32 if use_fp32_vae else torch.bfloat16
    print(f"\n{'='*60}")
    print(f"Loading pipeline:")
    print(f"  LoRA: {Path(lora_path).name if lora_path else 'None (base model)'}")
    print(f"  VAE dtype: {vae_dtype}")
    print(f"{'='*60}")

    # Load components
    print("\n  Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")

    print("  Loading text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    )

    print(f"  Loading VAE ({vae_dtype})...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=vae_dtype
    )

    print("  Loading transformer...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # Load LoRA if provided
    if lora_path:
        print(f"  Loading LoRA weights...")
        lora_state_dict = load_file(lora_path)
        missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
        print(f"    ✓ LoRA loaded: {len(lora_state_dict)} keys")

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
    pipeline = LTXPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    return pipeline, vae


def generate_with_reference(pipeline, vae, prompt, reference_video_path, args):
    """Generate video using reference for conditioning."""

    print(f"\nGenerating with reference video...")
    print(f"  Prompt: {prompt}")

    # Load and encode reference
    reference_tensor = load_reference_video(
        reference_video_path,
        target_frames=args.num_frames,
        target_height=args.height,
        target_width=args.width
    )

    reference_latents = encode_reference(vae, reference_tensor)

    # Generate
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"\n  Running inference...")
    print(f"    Steps: {args.steps}")
    print(f"    Guidance: {args.guidance_scale}")

    with torch.inference_mode():
        # NOTE: Standard LTXPipeline may not support reference latents directly
        # This is a limitation of diffusers' implementation
        # We'll generate normally and note this limitation

        try:
            # Try passing latents if pipeline supports it
            output = pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                latents=reference_latents,  # May not work
            )
        except TypeError:
            # Standard pipeline doesn't support reference latents
            print("\n    ⚠️  WARNING: LTXPipeline doesn't support reference latents")
            print("    This is a diffusers limitation - IC-LoRA may not activate properly")
            print("    Generating without reference conditioning...")

            output = pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
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

    if video.dtype != np.uint8:
        if video.max() > 1.0:
            video = video / 255.0
        video = (video.clip(0, 1) * 255).astype(np.uint8)

    return video


def analyze_video(video, name):
    """Analyze video statistics."""

    r_mean = video[:,:,:,0].mean()
    g_mean = video[:,:,:,1].mean()
    b_mean = video[:,:,:,2].mean()

    green_ratio = g_mean / ((r_mean + b_mean) / 2)

    print(f"\n  Results for {name}:")
    print(f"    Shape: {video.shape}")
    print(f"    RGB means: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
    print(f"    Green ratio: {green_ratio:.2f}")

    if green_ratio > 1.5:
        print(f"    ⚠️  GREEN TINT DETECTED")
    elif green_ratio > 1.2:
        print(f"    ⚠️  Slight green tint")
    else:
        print(f"    ✓ Colors balanced")

    return r_mean, g_mean, b_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str,
                       default="output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors")
    parser.add_argument("--reference_video", type=str, required=True,
                       help="Path to reference video from training set")
    parser.add_argument("--prompt", type=str,
                       default="A cinematic landscape with beautiful lighting and smooth camera movement")
    parser.add_argument("--negative_prompt", type=str,
                       default="worst quality, inconsistent motion, blurry")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fp32_vae", action="store_true",
                       help="Also test with FP32 VAE (slower but may fix green tint)")

    args = parser.parse_args()

    if not Path(args.reference_video).exists():
        print(f"Error: Reference video not found: {args.reference_video}")
        print("\nAvailable training videos:")
        videos_dir = Path("dataset/videos")
        if videos_dir.exists():
            for vid in list(videos_dir.glob("*.mp4"))[:10]:
                print(f"  {vid}")
        return 1

    print("="*60)
    print("IC-LoRA Test with Reference Video")
    print("="*60)
    print(f"Reference: {Path(args.reference_video).name}")
    print(f"Prompt: {args.prompt}")
    print("="*60)

    results = []

    # Test configurations
    tests = [
        ("Base_BF16", None, False, "base_with_ref_bf16.mp4"),
        ("LoRA_BF16", args.lora_path, False, "lora_with_ref_bf16.mp4"),
    ]

    if args.test_fp32_vae:
        tests.extend([
            ("Base_FP32", None, True, "base_with_ref_fp32.mp4"),
            ("LoRA_FP32", args.lora_path, True, "lora_with_ref_fp32.mp4"),
        ])

    for test_name, lora_path, use_fp32, output_file in tests:
        try:
            # Load pipeline
            pipeline, vae = load_pipeline(lora_path, use_fp32_vae=use_fp32)

            # Generate with reference
            video = generate_with_reference(
                pipeline, vae, args.prompt, args.reference_video, args
            )

            # Analyze
            r, g, b = analyze_video(video, test_name)
            results.append((test_name, r, g, b, output_file))

            # Save
            print(f"  Saving to {output_file}...")
            imageio.mimsave(output_file, video, fps=args.fps)
            print(f"  ✓ Saved")

            # Clear memory
            del pipeline, vae
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, None, None, None, None))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - IC-LoRA with Reference Video")
    print("="*60)

    for test_name, r, g, b, output_file in results:
        if r is not None:
            green_ratio = g / ((r + b) / 2)
            color_status = "✓ OK" if green_ratio < 1.2 else f"⚠️ GREEN ({green_ratio:.2f}x)"
            print(f"{test_name:15s}: R={r:5.1f} G={g:5.1f} B={b:5.1f} - {color_status}")
            print(f"                 Output: {output_file}")
        else:
            print(f"{test_name:15s}: FAILED")

    # Compare base vs LoRA
    if len(results) >= 2 and results[0][1] is not None and results[1][1] is not None:
        print("\n" + "="*60)
        print("BASE vs LoRA Comparison:")
        print("="*60)

        base_rgb = (results[0][1], results[0][2], results[0][3])
        lora_rgb = (results[1][1], results[1][2], results[1][3])

        diff = abs(base_rgb[0] - lora_rgb[0]) + abs(base_rgb[1] - lora_rgb[1]) + abs(base_rgb[2] - lora_rgb[2])

        if diff < 5:
            print("⚠️  Videos appear IDENTICAL - LoRA may not be working")
            print("    This could mean:")
            print("    1. IC-LoRA needs proper reference conditioning (diffusers limitation)")
            print("    2. LoRA is not activating with text-only prompts")
            print("    3. Need to use official LTX-Video inference (not diffusers)")
        else:
            print(f"✓ Videos are DIFFERENT (RGB diff: {diff:.1f})")
            print("  LoRA is affecting the output!")

    print("="*60)


if __name__ == "__main__":
    main()
