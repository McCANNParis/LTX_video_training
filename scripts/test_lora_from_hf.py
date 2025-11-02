#!/usr/bin/env python3
"""
Test LoRA loaded from HuggingFace Hub
"""

import argparse
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel, LTXPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download
import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_repo", type=str, required=True, help="HuggingFace repo (username/repo-name)")
    parser.add_argument("--lora_filename", type=str, default="lora_weights_step_04750.safetensors")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="test_hf_lora.mp4")
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
    print("Testing LoRA from HuggingFace Hub")
    print("="*60)
    print(f"Repository: {args.lora_repo}")
    print(f"LoRA file: {args.lora_filename}")
    print(f"Prompt: {args.prompt}")
    print("="*60)

    # Download LoRA from HuggingFace
    print("\nDownloading LoRA from HuggingFace Hub...")
    lora_path = hf_hub_download(
        repo_id=args.lora_repo,
        filename=args.lora_filename
    )
    print(f"✓ Downloaded to: {lora_path}")

    # Load base model components
    print("\nLoading base model components...")

    print("  - Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")

    print("  - Text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    )

    print("  - VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )

    print("  - Transformer...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # Load LoRA weights
    print(f"\nLoading LoRA weights from HuggingFace...")
    from safetensors.torch import load_file
    lora_state_dict = load_file(lora_path)

    missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
    print(f"✓ LoRA loaded: {len(lora_state_dict)} keys")
    if unexpected:
        print(f"  Note: {len(unexpected)} unexpected keys (normal for LoRA)")

    print("  - Scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="scheduler"
    )

    # Move to GPU
    print("\nMoving to GPU...")
    text_encoder = text_encoder.to("cuda")
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")

    # Create pipeline
    print("Creating pipeline...")
    pipeline = LTXPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # Generate
    print(f"\nGenerating video...")
    print(f"  Steps: {args.steps}")
    print(f"  Guidance: {args.guidance_scale}")
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

    print(f"\nVideo stats:")
    print(f"  Shape: {video.shape}")
    print(f"  Range: [{video.min()}, {video.max()}]")
    print(f"  Mean per channel: R={video[:,:,:,0].mean():.1f}, G={video[:,:,:,1].mean():.1f}, B={video[:,:,:,2].mean():.1f}")

    print(f"\nSaving to {args.output}...")
    imageio.mimsave(args.output, video, fps=args.fps)

    print(f"\n✓ Done!")
    print(f"  Duration: {args.num_frames / args.fps:.2f}s")
    print(f"  Resolution: {args.width}x{args.height}")
    print("="*60)


if __name__ == "__main__":
    main()
