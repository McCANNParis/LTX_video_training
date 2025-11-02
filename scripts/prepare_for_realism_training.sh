#!/bin/bash
# Prepare Data for Simple Realism LoRA Training
# Text-to-video training without trajectory conditioning

set -e

cd /workspace/LTX_video_training

echo "========================================"
echo "Preparing Data for Realism LoRA Training"
echo "========================================"
echo ""

# Check if dataset exists
if [ ! -d "dataset/videos" ]; then
    echo "Error: Dataset not found. Please ensure dataset/videos/ contains training videos!"
    exit 1
fi

if [ ! -d "dataset/captions" ]; then
    echo "Error: Captions not found. Please ensure dataset/captions/ contains caption files!"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(ls -1 dataset/videos/*.mp4 2>/dev/null | wc -l)
CAPTION_COUNT=$(ls -1 dataset/captions/*.txt 2>/dev/null | wc -l)

echo "Found $VIDEO_COUNT training videos"
echo "Found $CAPTION_COUNT caption files"
echo ""

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "Error: No videos found in dataset/videos/"
    exit 1
fi

if [ "$CAPTION_COUNT" -eq 0 ]; then
    echo "Error: No captions found in dataset/captions/"
    exit 1
fi

# Create dataset.json for text-only training
echo "Creating dataset.json for text-only training..."

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

dataset_json = []

# Get all videos
videos_dir = Path('dataset/videos')
captions_dir = Path('dataset/captions')

for video_file in sorted(videos_dir.glob('*.mp4')):
    # Find corresponding caption file
    # Assuming caption file has same name with .txt extension
    caption_file = captions_dir / f"{video_file.stem}.txt"

    if not caption_file.exists():
        print(f"Warning: No caption found for {video_file.name}, skipping...")
        continue

    # Read caption
    with open(caption_file, 'r') as f:
        caption = f.read().strip()

    # Add to dataset (no reference_video for text-only training)
    dataset_json.append({
        "video": str(video_file.absolute()),
        "caption": caption
    })

# Save dataset.json
with open('dataset_realism.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f"✓ Created dataset_realism.json with {len(dataset_json)} entries")
PYTHON_SCRIPT

echo ""

# Run official preprocessing (without reference videos)
echo "Running preprocessing for text-only training..."
echo "This will create .precomputed directory with VAE latents"
echo ""

cd LTX-Video-Trainer

# Set PYTHONPATH to include the scripts directory
export PYTHONPATH=/workspace/LTX_video_training/LTX-Video-Trainer:$PYTHONPATH

# Preprocess without reference videos
# Resolution format: "WxHxF" (width x height x frames)
# We use 704x1216x121 for our videos
python scripts/preprocess_dataset.py \
    /workspace/LTX_video_training/dataset_realism.json \
    --resolution-buckets "704x1216x121" \
    --video-column "video" \
    --caption-column "caption" \
    --output-dir /workspace/LTX_video_training/preprocessed_realism \
    --vae-tiling \
    --batch-size 1

echo ""
echo "✓ Preprocessing complete!"
echo ""
echo "Created:"
echo "  /workspace/LTX_video_training/preprocessed_realism/.precomputed/"
echo ""
echo "Now you can train with:"
echo "  ./scripts/train_realism_lora.sh"
echo ""
