#!/bin/bash
# Quick setup checker for Realism LoRA training

echo "========================================"
echo "Realism LoRA Training - Setup Checker"
echo "========================================"
echo ""

BASE_DIR="/workspace/LTX_video_training"
cd "$BASE_DIR"

echo "Base directory: $BASE_DIR"
echo ""

# Check scripts
echo "✓ Scripts found:"
ls -1 scripts/quick_start_realism.sh scripts/prepare_for_realism_training.sh scripts/train_realism_lora.sh scripts/test_realism_lora.py 2>/dev/null | sed 's/^/  /'
echo ""

# Check dataset
if [ -d "raw_videos" ]; then
    VIDEO_COUNT=$(ls -1 raw_videos/*.mp4 2>/dev/null | wc -l)
    CAPTION_COUNT=$(ls -1 raw_videos/*.txt 2>/dev/null | wc -l)
    echo "✓ Dataset folder exists: raw_videos/"
    echo "  Videos: $VIDEO_COUNT"
    echo "  Captions: $CAPTION_COUNT"
    
    if [ $VIDEO_COUNT -gt 0 ] && [ $CAPTION_COUNT -gt 0 ]; then
        echo "  Status: READY TO PREPROCESS"
        echo ""
        echo "Next step: ./scripts/prepare_for_realism_training.sh"
    else
        echo "  Status: EMPTY - Add your videos and captions"
        echo ""
        echo "Add files to: $BASE_DIR/raw_videos/"
    fi
else
    echo "⚠ Dataset folder missing: raw_videos/"
    echo ""
    echo "Create it with:"
    echo "  mkdir -p $BASE_DIR/raw_videos"
    echo ""
    echo "Then add your .mp4 and .txt files with matching names"
fi
echo ""

# Check if preprocessed
if [ -d "preprocessed_realism/.precomputed" ]; then
    echo "✓ Preprocessed data found"
    echo "  Status: READY TO TRAIN"
    echo ""
    echo "Next step: ./scripts/train_realism_lora.sh"
else
    echo "○ Not yet preprocessed"
fi
echo ""

# Check if trained
if [ -d "output/realism_lora/checkpoints" ]; then
    CHECKPOINT_COUNT=$(ls -1 output/realism_lora/checkpoints/*.safetensors 2>/dev/null | wc -l)
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "✓ Training checkpoints found: $CHECKPOINT_COUNT"
        echo "  Latest checkpoint:"
        ls -1t output/realism_lora/checkpoints/*.safetensors 2>/dev/null | head -1 | sed 's/^/  /'
        echo ""
        echo "Next step: Test your LoRA"
        echo "  python scripts/test_realism_lora.py --compare"
    fi
else
    echo "○ Not yet trained"
fi

echo ""
echo "========================================"
echo "Quick Commands:"
echo "========================================"
echo "Interactive guide:  ./scripts/quick_start_realism.sh"
echo "Prepare dataset:    ./scripts/prepare_for_realism_training.sh"
echo "Train:              ./scripts/train_realism_lora.sh"
echo "Test:               python scripts/test_realism_lora.py --compare"
echo "========================================"
