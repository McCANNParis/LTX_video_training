#!/bin/bash
# Prepare Our Data for Official Trainer
# Converts our preprocessed data to official trainer format

set -e

cd /workspace/LTX_video_training

echo "========================================"
echo "Preparing Data for Official Trainer"
echo "========================================"
echo ""

# Check if dataset exists
if [ ! -d "dataset/videos" ]; then
    echo "Error: Dataset not found. Run prepare_dataset_single_source.py first!"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(ls -1 dataset/videos/*.mp4 2>/dev/null | wc -l)
echo "Found $VIDEO_COUNT training videos"
echo ""

# Create dataset.json for official trainer
echo "Creating dataset.json..."

python3 << 'PYTHON_SCRIPT'
import json
import csv
from pathlib import Path

# Read our train.csv
dataset_json = []
with open('dataset/train.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Get absolute paths
        video_path = str(Path(row['video_path']).absolute())
        caption_path = str(Path(row['caption_path']).absolute())
        traj_path = str(Path(row['trajectory_path']).absolute())

        # Read caption
        with open(caption_path, 'r') as cf:
            caption = cf.read().strip()

        dataset_json.append({
            "video": video_path,
            "caption": caption,
            "reference_video": traj_path  # Trajectory visualization as reference
        })

# Save dataset.json
with open('dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f"Created dataset.json with {len(dataset_json)} entries")
PYTHON_SCRIPT

echo "✓ Created dataset.json"
echo ""

# Run official preprocessing
echo "Running official trainer preprocessing..."
echo "This will create .precomputed directory with VAE latents"
echo ""

cd LTX-Video-Trainer

# Set PYTHONPATH to include the scripts directory
export PYTHONPATH=/workspace/LTX_video_training/LTX-Video-Trainer:$PYTHONPATH

# Note: dataset path is a positional argument
# Resolution format: "WxHxF" (width x height x frames)
# We use 704x1216x121 for our trajectory videos
python scripts/preprocess_dataset.py \
    /workspace/LTX_video_training/dataset.json \
    --resolution-buckets "704x1216x121" \
    --video-column "video" \
    --caption-column "caption" \
    --reference-column "reference_video" \
    --output-dir /workspace/LTX_video_training/preprocessed_official \
    --vae-tiling \
    --batch-size 1

echo ""
echo "✓ Preprocessing complete!"
echo ""
echo "Created:"
echo "  /workspace/LTX_video_training/preprocessed_official/.precomputed/"
echo ""
echo "Now you can train with:"
echo "  ./scripts/train_with_official.sh"
echo ""
