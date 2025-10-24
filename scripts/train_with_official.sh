#!/bin/bash
# Train Trajectory Control using Official LTX-Video Trainer
# This script runs everything from /workspace/LTX_video_training

set -e

echo "========================================="
echo "LTX Video Trajectory Control Training"
echo "Using Official Trainer"
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
if [ ! -f "LTX-Video-Trainer/configs/trajectory_control_h100.yaml" ]; then
    echo -e "${RED}Error: Training config not found!${NC}"
    echo "Run setup first:"
    echo "  ./scripts/setup_official_trainer.sh"
    exit 1
fi

# Check dataset
VIDEO_COUNT=$(ls -1 dataset/videos/*.mp4 2>/dev/null | wc -l)
if [ $VIDEO_COUNT -eq 0 ]; then
    echo -e "${RED}Error: No videos found in dataset!${NC}"
    echo "Run dataset preparation first:"
    echo "  python scripts/prepare_dataset_single_source.py \\"
    echo "      --input_dir /workspace/LTX_video_training/raw_videos \\"
    echo "      --output_dir /workspace/LTX_video_training/dataset \\"
    echo "      --use_existing_captions"
    exit 1
fi

echo "Dataset: $VIDEO_COUNT videos"
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Create output directory
mkdir -p output/trajectory_control_official
mkdir -p logs

# Timestamp for log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/workspace/LTX_video_training/logs/train_official_${TIMESTAMP}.log"

echo "Starting training..."
echo "Config: LTX-Video-Trainer/configs/trajectory_control_h100.yaml"
echo "Output: /workspace/LTX_video_training/output/trajectory_control_official"
echo "Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  watch -n 1 nvidia-smi"
echo ""

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Change to trainer directory
cd LTX-Video-Trainer

# Launch training
echo "Launching accelerate..."
accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=1 \
    --num_machines=1 \
    --dynamo_backend=inductor \
    scripts/train.py \
    configs/trajectory_control_h100.yaml \
    2>&1 | tee "$LOG_FILE"

# Return to main directory
cd ..

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Checkpoints saved to:"
echo "  /workspace/LTX_video_training/output/trajectory_control_official/"
echo ""
echo "View with TensorBoard:"
echo "  tensorboard --logdir /workspace/LTX_video_training/output/trajectory_control_official/tensorboard"
echo ""
