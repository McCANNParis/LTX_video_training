#!/bin/bash
# Prepare Data for Simple Realism LoRA Training
# Text-to-video training without trajectory conditioning

set -e

cd /workspace/LTX_video_training

echo "========================================"
echo "Preparing Data for Realism LoRA Training"
echo "========================================"
echo ""

# Check if raw_videos directory exists
if [ ! -d "raw_videos" ]; then
    echo "Error: raw_videos/ directory not found!"
    echo ""
    echo "Please create the directory and add your training data:"
    echo "  mkdir -p raw_videos"
    echo "  # Add your video files (.mp4) and caption files (.txt) with matching names"
    echo ""
    echo "Example structure:"
    echo "  raw_videos/"
    echo "    video_001.mp4"
    echo "    video_001.txt"
    echo "    video_002.mp4"
    echo "    video_002.txt"
    exit 1
fi

# Count videos and captions
VIDEO_COUNT=$(ls -1 raw_videos/*.mp4 2>/dev/null | wc -l)
CAPTION_COUNT=$(ls -1 raw_videos/*.txt 2>/dev/null | wc -l)

echo "Found $VIDEO_COUNT training videos"
echo "Found $CAPTION_COUNT caption files"
echo ""

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "Error: No videos found in raw_videos/"
    echo "Please add .mp4 files to raw_videos/"
    exit 1
fi

if [ "$CAPTION_COUNT" -eq 0 ]; then
    echo "Error: No captions found in raw_videos/"
    echo "Please add .txt caption files to raw_videos/"
    exit 1
fi

# Create dataset.json for text-only training
echo "Creating dataset.json for text-only training..."

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

dataset_json = []
raw_videos_dir = Path('raw_videos')

# Get all videos in raw_videos/
for video_file in sorted(raw_videos_dir.glob('*.mp4')):
    # Find corresponding caption file (same name with .txt extension)
    caption_file = raw_videos_dir / f"{video_file.stem}.txt"

    if not caption_file.exists():
        print(f"Warning: No caption found for {video_file.name}, skipping...")
        continue

    # Read caption
    with open(caption_file, 'r') as f:
        caption = f.read().strip()

    if not caption:
        print(f"Warning: Empty caption for {video_file.name}, skipping...")
        continue

    # Add to dataset with absolute path
    dataset_json.append({
        "video": str(video_file.absolute()),
        "caption": caption
    })

if len(dataset_json) == 0:
    print("Error: No valid video+caption pairs found!")
    print("Make sure each .mp4 file has a corresponding .txt file with the same name.")
    exit(1)

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

# Fix preprocessing bug: conditions are saved to raw_videos/ instead of preprocessed_realism/conditions/
# This happens because the preprocessing script uses absolute paths incorrectly
cd /workspace/LTX_video_training

echo ""
echo "Moving preprocessed files to correct locations..."

# Move conditions from raw_videos/ to preprocessed_realism/conditions/
if [ -f "raw_videos/*.pt" ] 2>/dev/null; then
    mv raw_videos/*.pt preprocessed_realism/conditions/ 2>/dev/null && \
        echo "✓ Moved condition files from raw_videos/ to preprocessed_realism/conditions/"
fi

# Also check for nested subdirectories (in case preprocessing behavior changes)
if [ -d "preprocessed_realism/conditions/raw_videos" ]; then
    mv preprocessed_realism/conditions/raw_videos/*.pt preprocessed_realism/conditions/ 2>/dev/null || true
    rmdir preprocessed_realism/conditions/raw_videos 2>/dev/null || true
fi

if [ -d "preprocessed_realism/latents/raw_videos" ]; then
    mv preprocessed_realism/latents/raw_videos/*.pt preprocessed_realism/latents/ 2>/dev/null || true
    rmdir preprocessed_realism/latents/raw_videos 2>/dev/null || true
fi

echo ""
echo "✓ Preprocessing complete!"
echo ""
echo "Created:"
echo "  /workspace/LTX_video_training/preprocessed_realism/conditions/"
echo "  /workspace/LTX_video_training/preprocessed_realism/latents/"
echo ""
echo "Now you can train with:"
echo "  export TORCH_COMPILE_DISABLE=1"
echo "  ./scripts/train_realism_lora.sh"
echo ""
