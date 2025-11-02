#!/usr/bin/env python3
"""Download the community-fixed VAE"""

from huggingface_hub import hf_hub_download
import os

print("Downloading community-fixed VAE...")
print("This may take a few minutes...")

# Download the VAE file
try:
    vae_path = hf_hub_download(
        repo_id="spacepxl/ltx-video-0.9-vae-finetune",
        filename="ltx-video-v0.9-vae_finetune_all.safetensors",
        local_dir="/workspace/LTX_video_training/models/fixed_vae",
        local_dir_use_symlinks=False
    )
    print(f"\n✓ Downloaded to: {vae_path}")

    # Also download the decoder-only version
    vae_decoder_path = hf_hub_download(
        repo_id="spacepxl/ltx-video-0.9-vae-finetune",
        filename="ltx-video-v0.9-vae_finetune_decoder_only.safetensors",
        local_dir="/workspace/LTX_video_training/models/fixed_vae",
        local_dir_use_symlinks=False
    )
    print(f"✓ Downloaded to: {vae_decoder_path}")

    print("\nFiles ready to use!")

except Exception as e:
    print(f"\n✗ Error: {e}")
