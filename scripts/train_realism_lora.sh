#!/bin/bash
# Train Simple Realism LoRA using Official LTX-Video Trainer
# Text-to-video training without trajectory conditioning

set -e

echo "========================================="
echo "LTX Video Realism LoRA Training"
echo "Text-to-Video Mode (No Trajectory)"
echo "========================================="
echo ""

# Ensure we're in the right directory
cd /workspace/LTX_video_training

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if official trainer is set up
if [ ! -d "LTX-Video-Trainer" ]; then
    echo -e "${RED}Error: Official trainer not found!${NC}"
    echo "Run setup first:"
    echo "  ./scripts/setup_official_trainer.sh"
    exit 1
fi

# Check if config exists
if [ ! -f "LTX-Video-Trainer/configs/realism_lora.yaml" ]; then
    echo -e "${RED}Error: Realism training config not found!${NC}"
    echo "The config file should have been created at:"
    echo "  LTX-Video-Trainer/configs/realism_lora.yaml"
    exit 1
fi

# Check preprocessed data
if [ ! -d "preprocessed_realism/.precomputed" ]; then
    echo -e "${RED}Error: Preprocessed data not found!${NC}"
    echo "Run dataset preparation first:"
    echo "  ./scripts/prepare_for_realism_training.sh"
    echo ""
    echo "Make sure you have your videos and captions in raw_videos/ first:"
    echo "  raw_videos/"
    echo "    video_001.mp4"
    echo "    video_001.txt"
    echo "    ..."
    exit 1
fi

# Count videos from dataset manifest
if [ -f "dataset_realism.json" ]; then
    VIDEO_COUNT=$(python3 -c "import json; print(len(json.load(open('dataset_realism.json'))))")
    echo "Dataset: $VIDEO_COUNT videos"
else
    echo "Dataset: preprocessed_realism/.precomputed/"
fi
echo ""

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    echo -e "${RED}Error: No GPUs detected!${NC}"
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "Detected GPUs: $NUM_GPUS"

# Set effective batch size based on number of GPUs
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU training enabled: Using $NUM_GPUS GPUs"
    echo "Note: Effective batch size will be: batch_size × gradient_accumulation × num_gpus"
    echo "      Current config: 4 × 2 × $NUM_GPUS = $((8 * NUM_GPUS))"
else
    echo "Single GPU training"
    echo "Effective batch size: batch_size × gradient_accumulation = 4 × 2 = 8"
fi
echo ""

# Create output directory
mkdir -p output/realism_lora
mkdir -p logs

# Timestamp for log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/workspace/LTX_video_training/logs/train_realism_${TIMESTAMP}.log"

echo "Starting training..."
echo "Config: LTX-Video-Trainer/configs/realism_lora.yaml"
echo "Output: /workspace/LTX_video_training/output/realism_lora"
echo "Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  watch -n 1 nvidia-smi"
echo ""

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Disable torch.compile if set
if [ -n "$TORCH_COMPILE_DISABLE" ]; then
    echo "Note: torch.compile is disabled via TORCH_COMPILE_DISABLE"
    echo ""
fi

# Change to trainer directory
cd LTX-Video-Trainer

# Launch training with detected GPUs
echo "Launching accelerate with $NUM_GPUS GPU(s)..."
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    accelerate launch \
        --mixed_precision=bf16 \
        --num_processes=1 \
        --num_machines=1 \
        scripts/train.py \
        configs/realism_lora.yaml \
        2>&1 | tee "$LOG_FILE"
else
    # Multi-GPU training
    accelerate launch \
        --mixed_precision=bf16 \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --num_machines=1 \
        scripts/train.py \
        configs/realism_lora.yaml \
        2>&1 | tee "$LOG_FILE"
fi

# Return to main directory
cd ..

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Checkpoints saved to:"
echo "  /workspace/LTX_video_training/output/realism_lora/"
echo ""
echo "Test your LoRA with:"
echo "  python scripts/test_realism_lora.py"
echo ""
echo "View with TensorBoard:"
echo "  tensorboard --logdir /workspace/LTX_video_training/output/realism_lora/tensorboard"
echo ""
