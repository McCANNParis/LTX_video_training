#!/usr/bin/env python3
"""
Test base model vs LoRA using LTXConditionPipeline with proper settings.
Simplified version without upsampler.
"""

import argparse
import torch
from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from diffusers import LTXVideoTransformer3DModel
from safetensors.torch import load_file
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


def load_lora_into_transformer(transformer, lora_path):
    """Load LoRA weights into transformer."""
    print(f"  Loading LoRA: {Path(lora_path).name}...")
    lora_state_dict = load_file(lora_path)
    missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
    print(f"    ✓ LoRA loaded: {len(lora_state_dict)} keys")
    if unexpected:
        print(f"    Note: {len(unexpected)} unexpected keys (normal for LoRA)")
    return transformer


def test_with_conditioning(lora_path=None, args=None):
    """Test using LTXConditionPipeline with image conditioning."""

    model_name = "BASE MODEL" if not lora_path else "WITH LORA"
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Load pipeline
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

    print("Moving to GPU...")
    pipe.to("cuda")

    # Calculate mu and disable dynamic shifting
    print("Configuring scheduler...")
    latent_height = args.height // 8
    latent_width = args.width // 8
    image_seq_len = args.num_frames * latent_height * latent_width
    mu = calculate_shift(image_seq_len)
    print(f"  Calculated mu: {mu:.4f}")

    if hasattr(pipe.scheduler.config, 'use_dynamic_shifting'):
        pipe.scheduler.config.use_dynamic_shifting = False
        print("  Disabled dynamic shifting")

    print("Enabling VAE tiling...")
    pipe.vae.enable_tiling()

    # Prepare conditioning
    print("\nPreparing conditioning...")
    if args.conditioning_image:
        print(f"  Loading image: {args.conditioning_image}")
        image = load_image(args.conditioning_image)
    else:
        print("  Using default penguin image")
        image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png")

    # Compress image using video compression (as model was trained on videos)
    video = load_video(export_to_video([image]))
    condition1 = LTXVideoCondition(video=video, frame_index=0)

    print(f"\nGeneration settings:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Num frames: {args.num_frames}")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed}")

    generator = torch.Generator().manual_seed(args.seed)

    # Generate video
    print("\nGenerating video...")
    video = pipe(
        conditions=[condition1],
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=generator,
        output_type="pil",
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
    ).frames[0]
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
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("="*60)
    print("LTXConditionPipeline Test")
    print("Base Model vs LoRA Comparison")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} @ {args.fps}fps")
    print(f"LoRA: {Path(args.lora_path).name}")
    print("="*60)

    # Test 1: Base model
    print("\n\n" + "="*60)
    print("TEST 1: BASE MODEL")
    print("="*60)
    video_base = test_with_conditioning(lora_path=None, args=args)
    output_base = "condition_base.mp4"
    print(f"\nSaving to {output_base}...")
    export_to_video(video_base, output_base, fps=args.fps)
    print(f"✓ Saved: {output_base}")

    # Clear memory
    torch.cuda.empty_cache()

    # Test 2: With LoRA
    print("\n\n" + "="*60)
    print("TEST 2: WITH LORA")
    print("="*60)
    video_lora = test_with_conditioning(lora_path=args.lora_path, args=args)
    output_lora = "condition_lora.mp4"
    print(f"\nSaving to {output_lora}...")
    export_to_video(video_lora, output_lora, fps=args.fps)
    print(f"✓ Saved: {output_lora}")

    # Summary
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"Base model output: {output_base}")
    print(f"LoRA output:       {output_lora}")
    print("\nCompare the two videos:")
    print("  - If DIFFERENT → Your LoRA is working! 🎉")
    print("  - If IDENTICAL → LoRA may need specific reference video handling")
    print("="*60)


if __name__ == "__main__":
    main()
