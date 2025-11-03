#!/bin/bash
# Install Dependencies for Realism LoRA Training
# Run this on your RunPod container after git pull

set -e

echo "========================================"
echo "Installing LTX-Video-Trainer Dependencies"
echo "========================================"
echo ""

cd /workspace/LTX_video_training

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    pip install uv
fi

# Install LTX-Video-Trainer
echo ""
echo "Installing LTX-Video-Trainer and all dependencies..."
cd LTX-Video-Trainer
pip install -e . --no-cache-dir

cd ..

echo ""
echo "========================================"
echo "✓ Installation Complete!"
echo "========================================"
echo ""
echo "You can now run:"
echo "  ./check_setup.sh"
echo "  ./scripts/quick_start_realism.sh"
echo ""
