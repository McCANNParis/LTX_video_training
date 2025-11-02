#!/bin/bash
# Quick Start Script for Realism LoRA Training
# This script walks you through the complete workflow

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

cd /workspace/LTX_video_training

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        LTX Video - Realism LoRA Quick Start Guide         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check setup
echo -e "${YELLOW}[STEP 1/4]${NC} Checking setup..."
if [ ! -d "LTX-Video-Trainer" ]; then
    echo -e "${RED}✗ Official trainer not found${NC}"
    echo ""
    echo "Running setup (this will take a few minutes)..."
    ./scripts/setup_official_trainer.sh
else
    echo -e "${GREEN}✓ Official trainer found${NC}"
fi

if [ ! -f "LTX-Video-Trainer/configs/realism_lora.yaml" ]; then
    echo -e "${RED}✗ Realism config not found${NC}"
    echo "This should have been created. Please check the installation."
    exit 1
else
    echo -e "${GREEN}✓ Realism config found${NC}"
fi
echo ""

# Step 2: Check dataset
echo -e "${YELLOW}[STEP 2/4]${NC} Checking dataset..."
VIDEO_COUNT=$(ls -1 dataset/videos/*.mp4 2>/dev/null | wc -l)
CAPTION_COUNT=$(ls -1 dataset/captions/*.txt 2>/dev/null | wc -l)

if [ $VIDEO_COUNT -eq 0 ] || [ $CAPTION_COUNT -eq 0 ]; then
    echo -e "${RED}✗ Dataset not found${NC}"
    echo ""
    echo "Your dataset should have this structure:"
    echo "  dataset/"
    echo "    videos/       # Your MP4 videos"
    echo "    captions/     # Matching .txt caption files"
    echo ""
    echo "Please add your training data to dataset/ and run this script again."
    exit 1
else
    echo -e "${GREEN}✓ Found $VIDEO_COUNT videos and $CAPTION_COUNT captions${NC}"

    # Check if preprocessed
    if [ ! -d "preprocessed_realism/.precomputed" ]; then
        echo ""
        echo "Dataset needs preprocessing..."
        read -p "Run preprocessing now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ./scripts/prepare_for_realism_training.sh
        else
            echo "Run preprocessing manually:"
            echo "  ./scripts/prepare_for_realism_training.sh"
            exit 0
        fi
    else
        echo -e "${GREEN}✓ Dataset already preprocessed${NC}"
    fi
fi
echo ""

# Step 3: Check GPU
echo -e "${YELLOW}[STEP 3/4]${NC} Checking GPU..."
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 2>/dev/null || echo "0")
if [ "$NUM_GPUS" -eq 0 ]; then
    echo -e "${RED}✗ No GPUs detected${NC}"
    echo "GPU is required for training."
    exit 1
else
    echo -e "${GREEN}✓ Detected $NUM_GPUS GPU(s)${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

    # Estimate training time
    if [ "$NUM_GPUS" -eq 1 ]; then
        echo "  Estimated training time: ~17.5 hours (5000 steps)"
    elif [ "$NUM_GPUS" -eq 2 ]; then
        echo "  Estimated training time: ~8-9 hours (5000 steps)"
    elif [ "$NUM_GPUS" -ge 4 ]; then
        echo "  Estimated training time: ~4-5 hours (5000 steps)"
    fi
fi
echo ""

# Step 4: Start training
echo -e "${YELLOW}[STEP 4/4]${NC} Ready to train!"
echo ""
echo "Training configuration:"
echo "  LoRA rank: 128 (654M parameters)"
echo "  Steps: 5000"
echo "  Batch size: 4 × 2 × $NUM_GPUS = $((8 * NUM_GPUS)) (effective)"
echo "  Learning rate: 1.0e-4"
echo "  Output: output/realism_lora/checkpoints/"
echo ""
echo "Checkpoints will be saved every 250 steps."
echo "You can test intermediate checkpoints during training."
echo ""
read -p "Start training now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}Starting training...${NC}"
    echo ""
    echo "Tip: You can monitor training with:"
    echo "  tail -f logs/train_realism_*.log"
    echo "  watch -n 1 nvidia-smi"
    echo ""
    sleep 2

    export TORCH_COMPILE_DISABLE=1
    ./scripts/train_realism_lora.sh
else
    echo ""
    echo "When you're ready to train, run:"
    echo "  export TORCH_COMPILE_DISABLE=1"
    echo "  ./scripts/train_realism_lora.sh"
    echo ""
    echo "See REALISM_LORA_GUIDE.md for detailed documentation."
fi
