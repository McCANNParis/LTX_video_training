#!/bin/bash
# H100 Training Launch Script for Trajectory-Guided LTX Video

set -e

echo "========================================"
echo "LTX Video Trajectory Training - H100"
echo "========================================"
echo ""

# Check if config exists
CONFIG_PATH="/workspace/LTX_video_training/configs/trajectory_iclora_h100.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Check if dataset exists
DATASET_PATH="/workspace/LTX_video_training/dataset/train.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Please run dataset preparation first."
    exit 1
fi

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Count training samples
NUM_SAMPLES=$(tail -n +2 "$DATASET_PATH" | wc -l)
echo "Training samples: $NUM_SAMPLES"
echo ""

# Set environment variables for H100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Create output directory
mkdir -p /workspace/LTX_video_training/output/trajectory_iclora_h100
mkdir -p /workspace/LTX_video_training/logs

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/workspace/LTX_video_training/logs/train_h100_${TIMESTAMP}.log"

echo "Starting training..."
echo "Config: $CONFIG_PATH"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor training with:"
echo "  tail -f $LOG_FILE"
echo "  watch -n 1 nvidia-smi"
echo ""

# Launch training with accelerate
cd /workspace/LTX_video_training

accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=1 \
    --num_machines=1 \
    --dynamo_backend=inductor \
    src/training/train_trajectory_iclora.py \
    --config "$CONFIG_PATH" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Checkpoints saved to:"
echo "  /workspace/LTX_video_training/output/trajectory_iclora_h100/"
echo ""
echo "To resume training, edit the config:"
echo "  resume_from_checkpoint: 'latest'"
echo ""
