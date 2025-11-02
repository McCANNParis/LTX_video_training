#!/usr/bin/env python3
"""
Test using OFFICIAL LTX-Video inference (not diffusers).
This is the recommended way from Lightricks.
"""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="test_official.mp4")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("="*60)
    print("Using OFFICIAL LTX-Video Inference")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print("="*60)

    try:
        from ltx_video.inference import infer, InferenceConfig

        print("\n✓ Official LTX-Video library found")
        print("\nGenerating...")

        infer(InferenceConfig(
            pipeline_config="configs/ltxv-13b-0.9.8-distilled.yaml",  # Or appropriate config
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            output_path=args.output,
            seed=args.seed,
        ))

        print(f"\n✓ Done! Saved to {args.output}")

    except ImportError as e:
        print(f"\n✗ Official LTX-Video library not installed")
        print(f"\nTo install:")
        print(f"  pip install git+https://github.com/Lightricks/LTX-Video.git")
        print(f"\nOr clone and install locally:")
        print(f"  git clone https://github.com/Lightricks/LTX-Video.git")
        print(f"  cd LTX-Video")
        print(f"  pip install -e .")
        return 1

    print("="*60)

if __name__ == "__main__":
    main()
