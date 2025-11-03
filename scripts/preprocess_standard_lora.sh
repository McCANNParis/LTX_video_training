#!/bin/bash
# Preprocess raw videos for standard LoRA training
# Reads from: /workspace/LTX_video_training/raw_videos/*.mp4 + *.txt
# Outputs to: /workspace/LTX_video_training/preprocessed_standard/

set -e

echo "=================================="
echo "Data Preprocessing for Standard LoRA"
echo "=================================="
echo ""

# Navigate to trainer directory
cd /workspace/LTX_video_training/LTX-Video-Trainer

# Check raw data
echo "Checking raw data..."
if [ ! -d "/workspace/LTX_video_training/raw_videos" ]; then
    echo "ERROR: /workspace/LTX_video_training/raw_videos/ not found!"
    exit 1
fi

VIDEO_COUNT=$(find /workspace/LTX_video_training/raw_videos -name "*.mp4" | wc -l)
TXT_COUNT=$(find /workspace/LTX_video_training/raw_videos -name "*.txt" | wc -l)

echo "Found:"
echo "  Videos: $VIDEO_COUNT"
echo "  Captions (.txt): $TXT_COUNT"
echo ""

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "ERROR: No .mp4 files found!"
    exit 1
fi

if [ "$VIDEO_COUNT" -ne "$TXT_COUNT" ]; then
    echo "WARNING: Video count ($VIDEO_COUNT) != Caption count ($TXT_COUNT)"
    echo "Each video should have a matching .txt file with the same name."
    echo ""
fi

echo "Expected format:"
echo "  video_001.mp4  -> video_001.txt"
echo "  video_002.mp4  -> video_002.txt"
echo "  ..."
echo ""

read -p "Continue with preprocessing? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "Starting preprocessing..."
echo "This will:"
echo "  1. Load videos and captions"
echo "  2. Encode videos to latents with VAE"
echo "  3. Generate text embeddings"
echo "  4. Save preprocessed data"
echo ""

# Create output directory
mkdir -p /workspace/LTX_video_training/preprocessed_standard

# Run preprocessing
python -m ltxv_trainer.cli preprocess \
    --config ../configs/standard_text_lora_h100.yaml \
    2>&1 | tee ../output/preprocessing_standard.log

echo ""
echo "=================================="
echo "Preprocessing Complete!"
echo "=================================="
echo ""
echo "Preprocessed data saved to:"
echo "  /workspace/LTX_video_training/preprocessed_standard/"
echo ""
echo "You can now start training:"
echo "  bash scripts/train_standard_lora.sh"
echo ""
