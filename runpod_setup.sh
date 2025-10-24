#!/bin/bash

# RunPod Environment Setup Script for Trajectory-Guided LTX Video Training
# This script configures the RunPod environment for training and inference

set -e

echo "=================================="
echo "RunPod Setup for LTX Video Training"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on RunPod
if [ ! -d "/workspace" ]; then
    echo -e "${YELLOW}Warning: /workspace not found. Creating it...${NC}"
    mkdir -p /workspace
fi

# Set workspace directory
WORKSPACE_DIR="/workspace/LTX_video_training"
cd /workspace

echo -e "\n${GREEN}Step 1: Cloning/Updating Repository${NC}"
if [ -d "$WORKSPACE_DIR" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd $WORKSPACE_DIR
    git pull
else
    echo "Cloning repository..."
    # Note: Update this URL with your actual repository
    git clone https://github.com/your-repo/LTX_video_training.git
    cd $WORKSPACE_DIR
fi

echo -e "\n${GREEN}Step 2: Checking PyTorch Installation${NC}"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo -e "${RED}PyTorch not properly installed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}Step 3: Installing Dependencies${NC}"

# Update pip
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install --no-cache-dir \
    diffusers>=0.30.0 \
    transformers>=4.35.0 \
    accelerate>=0.25.0 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0 \
    opencv-python-headless>=4.8.0 \
    pillow>=10.0.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    pandas>=2.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0 \
    click>=8.1.0 \
    wandb>=0.15.0 \
    tensorboard>=2.14.0

# Install Jupyter extensions if not present
echo "Setting up Jupyter extensions..."
pip install --no-cache-dir \
    ipywidgets \
    jupyter_contrib_nbextensions \
    jupyterlab-git

# Enable widgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix

echo -e "\n${GREEN}Step 4: Creating Directory Structure${NC}"

# Create necessary directories
mkdir -p $WORKSPACE_DIR/raw_videos
mkdir -p $WORKSPACE_DIR/dataset/{videos,trajectories,captions,metadata}
mkdir -p $WORKSPACE_DIR/models
mkdir -p $WORKSPACE_DIR/output
mkdir -p $WORKSPACE_DIR/checkpoints
mkdir -p $WORKSPACE_DIR/logs
mkdir -p $WORKSPACE_DIR/validation_outputs
mkdir -p $WORKSPACE_DIR/notebooks
mkdir -p $WORKSPACE_DIR/tmp

echo "Directory structure created."
echo "Upload your raw videos to: $WORKSPACE_DIR/raw_videos"

echo -e "\n${GREEN}Step 5: Setting Up Environment Variables${NC}"

# Create .env file
cat > $WORKSPACE_DIR/.env << EOF
# RunPod Environment Configuration
WORKSPACE_DIR=/workspace/LTX_video_training
CUDA_HOME=/usr/local/cuda
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false
HF_HOME=/workspace/.cache/huggingface
WANDB_DIR=/workspace/wandb
TRANSFORMERS_CACHE=/workspace/.cache/transformers
TORCH_HOME=/workspace/.cache/torch
EOF

# Source environment variables
source $WORKSPACE_DIR/.env
export $(cat $WORKSPACE_DIR/.env | grep -v '^#' | grep -v '^$' | xargs)

echo "Environment variables configured."

echo -e "\n${GREEN}Step 6: Configuring Accelerate${NC}"

# Create accelerate config for single/multi-GPU
mkdir -p ~/.cache/huggingface/accelerate

cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_machines: 1
num_processes: $(nvidia-smi --list-gpus | wc -l)
use_cpu: false
EOF

echo "Accelerate configured for $(nvidia-smi --list-gpus | wc -l) GPU(s)."

echo -e "\n${GREEN}Step 7: Setting Up Git LFS (for large files)${NC}"
if command -v git-lfs &> /dev/null; then
    git lfs install
    echo "Git LFS installed successfully."
else
    echo -e "${YELLOW}Git LFS not found. Skipping (optional - install with: apt-get install git-lfs)${NC}"
fi

echo -e "\n${GREEN}Step 8: Creating RunPod Helper Scripts${NC}"

# Create GPU monitor script
cat > $WORKSPACE_DIR/scripts/monitor_gpu.sh << 'EOF'
#!/bin/bash
# GPU Monitoring Script
watch -n 1 nvidia-smi
EOF
chmod +x $WORKSPACE_DIR/scripts/monitor_gpu.sh

# Create training status monitor
cat > $WORKSPACE_DIR/scripts/monitor_training.sh << 'EOF'
#!/bin/bash
# Training Status Monitor
if [ -z "$1" ]; then
    LOG_DIR="/workspace/LTX_video_training/logs"
else
    LOG_DIR="$1"
fi

echo "Monitoring training logs in: $LOG_DIR"
tail -f $LOG_DIR/*.log 2>/dev/null || echo "No log files found in $LOG_DIR"
EOF
chmod +x $WORKSPACE_DIR/scripts/monitor_training.sh

# Create cleanup script
cat > $WORKSPACE_DIR/scripts/cleanup.sh << 'EOF'
#!/bin/bash
# Cleanup Script - Free up disk space
echo "Cleaning up temporary files..."
rm -rf /workspace/LTX_video_training/tmp/*
rm -rf /workspace/.cache/pip
rm -rf /tmp/*
echo "Cleanup complete!"
df -h /workspace
EOF
chmod +x $WORKSPACE_DIR/scripts/cleanup.sh

echo -e "\n${GREEN}Step 9: Downloading Models (Optional)${NC}"
echo "Skipping automatic model download. Use notebooks to download models as needed."

echo -e "\n${GREEN}Step 10: Setting Up Jupyter Kernel${NC}"

# Install kernel
python -m ipykernel install --user --name ltxv_trajectory --display-name "LTX Video Trajectory"

echo -e "\n${GREEN}Step 11: Creating Quick Start Symlinks${NC}"

# Create symlinks in workspace root for easy access
cd /workspace
ln -sf $WORKSPACE_DIR/notebooks notebooks 2>/dev/null || true
ln -sf $WORKSPACE_DIR/output output 2>/dev/null || true
ln -sf $WORKSPACE_DIR/checkpoints checkpoints 2>/dev/null || true

echo -e "\n${GREEN}Setup Complete!${NC}"
echo "=================================="
echo "RunPod Environment Ready"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Open JupyterLab: http://your-runpod-instance:8888"
echo "2. Navigate to notebooks/ directory"
echo "3. Open '01_Quick_Start.ipynb' to begin"
echo ""
echo "Useful commands:"
echo "  - Monitor GPUs: $WORKSPACE_DIR/scripts/monitor_gpu.sh"
echo "  - Monitor training: $WORKSPACE_DIR/scripts/monitor_training.sh"
echo "  - Cleanup: $WORKSPACE_DIR/scripts/cleanup.sh"
echo ""
echo "Workspace: $WORKSPACE_DIR"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"
echo "Disk usage: $(df -h /workspace | tail -1 | awk '{print $3 " / " $2 " (" $5 ")"}')"
echo ""
echo -e "${GREEN}Happy Training! 🚀${NC}"
