#!/bin/bash

# Training launcher script for Trajectory-Guided LTX Video
# Usage: ./scripts/train.sh [config_path] [num_gpus]

set -e

CONFIG_PATH=${1:-"configs/trajectory_iclora_high_quality.yaml"}
NUM_GPUS=${2:-1}

echo "=================================="
echo "LTX Video Trajectory Training"
echo "=================================="
echo "Config: $CONFIG_PATH"
echo "GPUs: $NUM_GPUS"
echo "=================================="

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check if dataset exists
if [ ! -f "dataset/train.csv" ]; then
    echo "Warning: dataset/train.csv not found"
    echo "Please run: python scripts/prepare_dataset.py"
    exit 1
fi

# Create output directory
OUTPUT_DIR=$(grep "output_dir:" $CONFIG_PATH | awk '{print $2}' | tr -d '"')
mkdir -p $OUTPUT_DIR

# Set environment variables for optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Launch training
if [ $NUM_GPUS -eq 1 ]; then
    echo "Launching single GPU training..."
    accelerate launch \
        --mixed_precision bf16 \
        src/training/train_trajectory_iclora.py \
        --config $CONFIG_PATH
else
    echo "Launching multi-GPU training on $NUM_GPUS GPUs..."
    accelerate launch \
        --multi_gpu \
        --num_processes $NUM_GPUS \
        --mixed_precision bf16 \
        src/training/train_trajectory_iclora.py \
        --config $CONFIG_PATH
fi

echo "=================================="
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=================================="
