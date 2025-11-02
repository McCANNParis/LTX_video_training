#!/usr/bin/env python3
"""
Upload trained LoRA to HuggingFace Hub
"""

import argparse
from huggingface_hub import HfApi, create_repo
import os

def upload_lora(lora_path, repo_name, token=None):
    """Upload LoRA weights to HuggingFace Hub."""

    print("="*60)
    print("Uploading LoRA to HuggingFace Hub")
    print("="*60)

    # Initialize API
    api = HfApi(token=token)

    # Get username
    user_info = api.whoami(token=token)
    username = user_info['name']

    full_repo_name = f"{username}/{repo_name}"
    print(f"\nRepository: {full_repo_name}")

    # Create repo if it doesn't exist
    try:
        print(f"\nCreating repository...")
        create_repo(
            repo_id=full_repo_name,
            token=token,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository ready")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the LoRA file
    print(f"\nUploading {lora_path}...")
    api.upload_file(
        path_or_fileobj=lora_path,
        path_in_repo=os.path.basename(lora_path),
        repo_id=full_repo_name,
        token=token,
    )

    print(f"\n✓ Upload complete!")
    print(f"\nYour LoRA is now available at:")
    print(f"https://huggingface.co/{full_repo_name}")
    print("="*60)

    return full_repo_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA .safetensors file")
    parser.add_argument("--repo_name", type=str, required=True, help="Repository name (e.g., 'my-ltx-lora')")
    parser.add_argument("--token", type=str, default=None, help="HF token (optional if logged in)")

    args = parser.parse_args()

    if not os.path.exists(args.lora_path):
        print(f"Error: LoRA file not found: {args.lora_path}")
        return 1

    upload_lora(args.lora_path, args.repo_name, args.token)


if __name__ == "__main__":
    main()
