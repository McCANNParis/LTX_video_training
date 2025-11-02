#!/usr/bin/env python3
"""
Convert LoRA weights from official LTX-Video-Trainer format to ComfyUI format.

The official trainer uses PEFT-style keys:
  - transformer.transformer_blocks.X.attn1.to_k.lora_A.weight
  - transformer.transformer_blocks.X.attn1.to_k.lora_B.weight

ComfyUI typically expects diffusers-style keys:
  - lora_A.weight → lora_down.weight (down projection)
  - lora_B.weight → lora_up.weight (up projection)
"""

import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file, load_file
from collections import OrderedDict


def inspect_lora(lora_path):
    """Show structure of LoRA file."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {lora_path}")
    print(f"{'='*60}\n")

    state_dict = load_file(lora_path)

    print(f"Total keys: {len(state_dict)}")
    print(f"\nSample keys (first 10):")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        shape = state_dict[key].shape
        print(f"  {i+1}. {key}")
        print(f"     Shape: {shape}\n")

    # Count key types
    lora_a_count = sum(1 for k in state_dict.keys() if 'lora_A' in k)
    lora_b_count = sum(1 for k in state_dict.keys() if 'lora_B' in k)

    print(f"Statistics:")
    print(f"  - lora_A keys: {lora_a_count}")
    print(f"  - lora_B keys: {lora_b_count}")
    print(f"  - Total parameters: {sum(p.numel() for p in state_dict.values()):,}")

    return state_dict


def convert_key_format(key, conversion_type="diffusers"):
    """
    Convert key from trainer format to ComfyUI format.

    Conversion types:
    - "diffusers": lora_A → lora_down, lora_B → lora_up
    - "comfy_standard": Standard ComfyUI format with dots
    - "kohya": Kohya-ss style format
    """

    if conversion_type == "diffusers":
        # Simple rename: lora_A → lora_down, lora_B → lora_up
        new_key = key.replace('.lora_A.', '.lora_down.')
        new_key = new_key.replace('.lora_B.', '.lora_up.')
        return new_key

    elif conversion_type == "comfy_standard":
        # Keep as-is but ensure proper prefix
        # ComfyUI might expect keys without "transformer." prefix
        if key.startswith('transformer.transformer_blocks.'):
            # Remove first "transformer." prefix
            new_key = key.replace('transformer.transformer_blocks.', 'transformer_blocks.')
            new_key = new_key.replace('.lora_A.', '.lora_down.')
            new_key = new_key.replace('.lora_B.', '.lora_up.')
            return new_key
        return key

    elif conversion_type == "kohya":
        # Kohya-ss format: lora_unet_transformer_blocks_X_...
        new_key = key.replace('transformer.', 'lora_unet_')
        new_key = new_key.replace('.', '_')
        new_key = new_key.replace('_lora_A_weight', '.alpha')
        new_key = new_key.replace('_lora_B_weight', '.lora_up.weight')
        return new_key

    elif conversion_type == "no_prefix":
        # Remove all "transformer." prefixes
        new_key = key
        if key.startswith('transformer.'):
            new_key = key[len('transformer.'):]
        new_key = new_key.replace('.lora_A.', '.lora_down.')
        new_key = new_key.replace('.lora_B.', '.lora_up.')
        return new_key

    return key


def convert_lora(input_path, output_path, conversion_type="diffusers", add_metadata=True):
    """Convert LoRA from trainer format to ComfyUI format."""

    print(f"\n{'='*60}")
    print(f"Converting LoRA")
    print(f"{'='*60}\n")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Format: {conversion_type}\n")

    # Load original LoRA
    state_dict = load_file(input_path)

    # Convert keys
    new_state_dict = OrderedDict()
    conversions = []

    for old_key, value in state_dict.items():
        new_key = convert_key_format(old_key, conversion_type)
        new_state_dict[new_key] = value

        if old_key != new_key:
            conversions.append((old_key, new_key))

    # Show sample conversions
    print(f"Sample key conversions (first 5):")
    for i, (old_key, new_key) in enumerate(conversions[:5]):
        print(f"  {i+1}. OLD: {old_key}")
        print(f"     NEW: {new_key}\n")

    print(f"Total keys converted: {len(conversions)}")
    print(f"Total keys unchanged: {len(state_dict) - len(conversions)}")

    # Add metadata
    metadata = {}
    if add_metadata:
        metadata = {
            "converted_from": "ltxv-official-trainer",
            "conversion_type": conversion_type,
            "original_file": str(input_path.name),
            "model": "LTX-Video-13B",
            "lora_rank": "128",
            "training_type": "trajectory-control"
        }

    # Save converted LoRA
    print(f"\nSaving to: {output_path}")
    save_file(new_state_dict, output_path, metadata=metadata)

    print(f"✓ Conversion complete!")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return new_state_dict


def create_all_formats(input_path, output_dir):
    """Create multiple conversion formats to test."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = ["diffusers", "comfy_standard", "no_prefix", "kohya"]

    print(f"\n{'='*60}")
    print(f"Creating all format variants")
    print(f"{'='*60}\n")

    for fmt in formats:
        output_path = output_dir / f"{input_path.stem}_comfyui_{fmt}.safetensors"
        try:
            convert_lora(input_path, output_path, conversion_type=fmt)
            print(f"✓ Created: {output_path.name}\n")
        except Exception as e:
            print(f"✗ Failed {fmt}: {e}\n")

    print(f"\n{'='*60}")
    print(f"All conversions complete!")
    print(f"{'='*60}\n")
    print(f"Output directory: {output_dir}")
    print(f"\nTry each format in ComfyUI to see which one works:")
    for fmt in formats:
        print(f"  - {input_path.stem}_comfyui_{fmt}.safetensors")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LTX-Video LoRA from official trainer to ComfyUI format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input LoRA file (.safetensors)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: input_comfyui.safetensors)"
    )
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["diffusers", "comfy_standard", "kohya", "no_prefix", "all"],
        default="diffusers",
        help="Conversion type (default: diffusers). Use 'all' to create all variants."
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect the LoRA structure without converting"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Inspect
    if args.inspect_only:
        inspect_lora(input_path)
        return

    # Convert
    if args.type == "all":
        output_dir = input_path.parent / "converted"
        create_all_formats(input_path, output_dir)
    else:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_comfyui.safetensors"

        # Show inspection first
        inspect_lora(input_path)

        # Convert
        convert_lora(input_path, output_path, conversion_type=args.type)

        print(f"\n{'='*60}")
        print(f"Usage in ComfyUI:")
        print(f"{'='*60}\n")
        print(f"1. Copy {output_path.name} to ComfyUI/models/loras/")
        print(f"2. In ComfyUI, use 'Load LoRA' node")
        print(f"3. Select: {output_path.name}")
        print(f"4. Set strength to 0.8-1.0 for trajectory control")


if __name__ == "__main__":
    main()
