#!/bin/bash
# Helper script to set up your training data properly

cd /workspace/LTX_video_training

echo "========================================="
echo "Training Data Setup Helper"
echo "========================================="
echo ""

# Check if raw_videos exists
if [ ! -d "raw_videos" ]; then
    echo "Creating raw_videos/ directory..."
    mkdir -p raw_videos
    echo "✓ Created raw_videos/"
fi

# Count current files
VIDEO_COUNT=$(ls -1 raw_videos/*.mp4 2>/dev/null | wc -l)
CAPTION_COUNT=$(ls -1 raw_videos/*.txt 2>/dev/null | wc -l)

echo "Current status:"
echo "  Videos: $VIDEO_COUNT"
echo "  Captions: $CAPTION_COUNT"
echo ""

if [ $VIDEO_COUNT -eq 0 ]; then
    echo "⚠️  No videos found in raw_videos/"
    echo ""
    echo "Please upload your training videos (.mp4) and captions (.txt) to:"
    echo "  /workspace/LTX_video_training/raw_videos/"
    echo ""
    echo "Each video needs a matching caption file:"
    echo "  video_001.mp4 → video_001.txt"
    echo "  x6FCa26YpKUIvIMOCLGL.mp4 → x6FCa26YpKUIvIMOCLGL.txt"
    echo ""
    echo "After uploading, run this script again to verify."
    exit 0
fi

if [ $VIDEO_COUNT -ne $CAPTION_COUNT ]; then
    echo "⚠️  Mismatch: $VIDEO_COUNT videos but $CAPTION_COUNT captions"
    echo ""
    echo "Each .mp4 file needs a matching .txt file with the same name."
    echo ""
    echo "Videos without captions:"
    for video in raw_videos/*.mp4; do
        basename=$(basename "$video" .mp4)
        if [ ! -f "raw_videos/${basename}.txt" ]; then
            echo "  - ${basename}.mp4 (missing ${basename}.txt)"
        fi
    done
    echo ""
    exit 0
fi

echo "✓ Found $VIDEO_COUNT videos with matching captions!"
echo ""
echo "Next steps:"
echo "  1. Run preprocessing:"
echo "     ./scripts/prepare_for_realism_training.sh"
echo ""
echo "  2. Start training:"
echo "     export TORCH_COMPILE_DISABLE=1"
echo "     ./scripts/train_realism_lora.sh"
echo ""
