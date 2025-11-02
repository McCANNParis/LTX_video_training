#!/usr/bin/env python3
"""
Test base LTX-Video 13B dev model from HuggingFace.
Try different loading methods to diagnose green tint.
"""

import argparse
import torch
from diffusers import LTXPipeline
import imageio
import numpy as np


def test_from_pretrained_simple(args):
    """Test 1: Simple from_pretrained (what we've been using)"""

    print("\n" + "="*60)
    print("TEST 1: Simple from_pretrained")
    print("="*60)

    print("\nLoading pipeline with from_pretrained...")
    pipeline = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16
    )
    pipeline = pipeline.to("cuda")

    return pipeline


def test_from_pretrained_explicit_vae(args):
    """Test 2: Explicitly specify VAE subfolder"""

    print("\n" + "="*60)
    print("TEST 2: Explicit VAE from subfolder")
    print("="*60)

    from transformers import T5EncoderModel, T5Tokenizer
    from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    print("\nLoading components separately...")

    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")

    text_encoder = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    )

    # Try loading VAE with explicit dtype settings
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )

    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="scheduler"
    )

    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    pipeline = LTXPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    return pipeline


def test_with_fp32_vae(args):
    """Test 3: Try FP32 for VAE instead of BF16"""

    print("\n" + "="*60)
    print("TEST 3: FP32 VAE (instead of BF16)")
    print("="*60)

    from transformers import T5EncoderModel, T5Tokenizer
    from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    print("\nLoading components with FP32 VAE...")

    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")

    text_encoder = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    )

    # Try FP32 for VAE
    print("  Loading VAE in FP32 (might help with color accuracy)...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.float32  # FP32 instead of BF16
    )

    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="scheduler"
    )

    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    pipeline = LTXPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    return pipeline


def generate_video(pipeline, args):
    """Generate video with the pipeline"""

    print(f"\nGenerating video...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Steps: {args.steps}")
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


def analyze_video(video, test_name):
    """Analyze and print video statistics"""

    r_mean = video[:,:,:,0].mean()
    g_mean = video[:,:,:,1].mean()
    b_mean = video[:,:,:,2].mean()

    print(f"\nResults for {test_name}:")
    print(f"  Shape: {video.shape}")
    print(f"  RGB means: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")

    # Check if green is dominant
    green_ratio = g_mean / ((r_mean + b_mean) / 2)
    print(f"  Green/Average ratio: {green_ratio:.2f}")

    if green_ratio > 1.5:
        print(f"  ⚠️  GREEN TINT DETECTED (G is {green_ratio:.1f}x higher)")
    elif green_ratio > 1.2:
        print(f"  ⚠️  Slight green tint (G is {green_ratio:.1f}x higher)")
    else:
        print(f"  ✓ Colors appear balanced")

    return r_mean, g_mean, b_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "1", "2", "3"],
                       help="Which test to run (1=simple, 2=explicit, 3=fp32, all=all tests)")
    parser.add_argument("--prompt", type=str,
                       default="A scenic mountain landscape, camera slowly panning right")
    parser.add_argument("--negative_prompt", type=str,
                       default="worst quality, inconsistent motion, blurry")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("="*60)
    print("LTX-Video 13B Dev - Green Tint Diagnosis")
    print("="*60)
    print(f"Testing different loading methods...")
    print(f"Prompt: {args.prompt}")
    print("="*60)

    tests = []
    if args.test in ["all", "1"]:
        tests.append(("Test1_Simple", test_from_pretrained_simple, "test_method1.mp4"))
    if args.test in ["all", "2"]:
        tests.append(("Test2_Explicit", test_from_pretrained_explicit_vae, "test_method2.mp4"))
    if args.test in ["all", "3"]:
        tests.append(("Test3_FP32", test_with_fp32_vae, "test_method3.mp4"))

    results = []

    for test_name, test_func, output_file in tests:
        try:
            # Load pipeline
            pipeline = test_func(args)

            # Generate video
            video = generate_video(pipeline, args)

            # Analyze
            r, g, b = analyze_video(video, test_name)
            results.append((test_name, r, g, b, output_file))

            # Save
            print(f"  Saving to {output_file}...")
            imageio.mimsave(output_file, video, fps=args.fps)
            print(f"  ✓ Saved")

            # Clear memory
            del pipeline
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            results.append((test_name, None, None, None, None))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, r, g, b, output_file in results:
        if r is not None:
            green_ratio = g / ((r + b) / 2)
            status = "✓ BALANCED" if green_ratio < 1.2 else f"⚠️ GREEN TINT ({green_ratio:.2f}x)"
            print(f"{test_name:20s}: R={r:5.1f} G={g:5.1f} B={b:5.1f} - {status}")
            print(f"                      Output: {output_file}")
        else:
            print(f"{test_name:20s}: FAILED")

    print("="*60)

    # Recommendation
    best_test = min(results, key=lambda x: (x[2] / ((x[1] + x[3]) / 2)) if x[1] is not None else 999)
    if best_test[1] is not None:
        print(f"\n✓ BEST RESULT: {best_test[0]}")
        print(f"  Use this method for future generation")


if __name__ == "__main__":
    main()
