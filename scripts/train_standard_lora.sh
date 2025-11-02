#!/bin/bash
# Train Standard Text-to-Video LoRA
# NO trajectory conditioning - for improving realism

set -e

echo "=================================="
echo "Standard Text-to-Video LoRA Training"
echo "=================================="
echo ""
echo "Configuration:"
echo "  - Mode: Text-to-video (NO trajectory conditioning)"
echo "  - LoRA Rank: 128"
echo "  - Steps: 10,000"
echo "  - Batch Size: 1 (effective 8 with gradient accumulation)"
echo "  - Learning Rate: 1e-4"
echo "  - Dataset: Same 42 videos"
echo "  - GPU: H100 optimized"
echo "=================================="
echo ""

# Navigate to trainer directory
cd /workspace/LTX_video_training/LTX-Video-Trainer

# Check if preprocessed data exists
if [ ! -d "/workspace/LTX_video_training/preprocessed_official" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Run preprocessing first:"
    echo "  python -m ltxv_trainer.cli preprocess --config ../configs/standard_text_lora_h100.yaml"
    exit 1
fi

# Create output directory
mkdir -p /workspace/LTX_video_training/output/standard_text_lora

echo "Starting training..."
echo ""

# Run training
python -m ltxv_trainer.cli train \
    --config ../configs/standard_text_lora_h100.yaml \
    2>&1 | tee ../output/standard_text_lora/train.log

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo ""
echo "Checkpoints saved to:"
echo "  /workspace/LTX_video_training/output/standard_text_lora/checkpoints/"
echo ""
echo "To test your LoRA:"
echo "  cd /workspace/LTX_video_training"
echo "  python scripts/test_with_proper_guidance.py \\"
echo "    --lora_path output/standard_text_lora/checkpoints/lora_weights_step_XXXXX.safetensors \\"
echo "    --prompt \"Your prompt here\" \\"
echo "    --seed 42"
echo ""
