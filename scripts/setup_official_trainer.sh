#!/bin/bash
# Setup Official LTX-Video Trainer Inside Our Project
# This keeps everything self-contained in /workspace/LTX_video_training

set -e

echo "========================================="
echo "Setting Up Official LTX-Video Trainer"
echo "========================================="
echo ""

# Ensure we're in the right directory
cd /workspace/LTX_video_training

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Step 1: Cloning Official Trainer${NC}"
if [ -d "LTX-Video-Trainer" ]; then
    echo "Official trainer already exists. Updating..."
    cd LTX-Video-Trainer
    git pull
    cd ..
else
    echo "Cloning official trainer..."
    git clone https://github.com/Lightricks/LTX-Video-Trainer.git
fi

echo ""
echo -e "${GREEN}Step 2: Installing Dependencies${NC}"
# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# Install trainer dependencies
cd LTX-Video-Trainer
echo "Installing trainer dependencies..."
uv pip install --system .
cd ..

echo ""
echo -e "${GREEN}Step 3: Setting Up Data Links${NC}"
mkdir -p LTX-Video-Trainer/data/trajectory_control

# Remove old symlinks if they exist
rm -f LTX-Video-Trainer/data/trajectory_control/videos
rm -f LTX-Video-Trainer/data/trajectory_control/conditioning
rm -f LTX-Video-Trainer/data/trajectory_control/captions

# Create new symlinks
ln -s /workspace/LTX_video_training/dataset/videos LTX-Video-Trainer/data/trajectory_control/videos
ln -s /workspace/LTX_video_training/dataset/trajectories LTX-Video-Trainer/data/trajectory_control/conditioning
ln -s /workspace/LTX_video_training/dataset/captions LTX-Video-Trainer/data/trajectory_control/captions

echo "Linked dataset directories:"
echo "  - videos -> dataset/videos"
echo "  - conditioning -> dataset/trajectories"
echo "  - captions -> dataset/captions"

echo ""
echo -e "${GREEN}Step 4: Creating Training Configuration${NC}"

# Create config file
cat > LTX-Video-Trainer/configs/trajectory_control_h100.yaml << 'EOF'
# Trajectory Control IC-LoRA Configuration for H100
# Custom trajectory-guided video generation using our preprocessed data

# Basic settings
output_dir: /workspace/LTX_video_training/output/trajectory_control_official
seed: 42

# Model settings
model_id: "Lightricks/LTX-Video"
model_variant: "13B-0.9.7"

# Training mode: IC-LoRA for control
training_mode: "ic_lora"

# Dataset settings
dataset:
  type: "video_folder"
  video_dir: "data/trajectory_control/videos"
  conditioning_dir: "data/trajectory_control/conditioning"
  caption_dir: "data/trajectory_control/captions"

  # Video parameters
  resolution: 704
  num_frames: 121
  fps: 30

  # Preprocessing
  center_crop: false
  random_flip: 0.5

# IC-LoRA settings
ic_lora:
  rank: 512  # High rank for H100
  alpha: 512
  dropout: 0.05
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

  # Conditioning
  conditioning_type: "concat"
  conditioning_channels: 3  # RGB trajectory visualization
  conditioning_scale: 1.0

# Training settings
training:
  num_train_epochs: 100
  train_batch_size: 4  # H100 optimized
  gradient_accumulation_steps: 2

  learning_rate: 1.0e-4
  lr_scheduler: "cosine_with_restarts"
  lr_warmup_steps: 500
  lr_num_cycles: 3

  max_grad_norm: 1.0

  # Optimizer
  optimizer: "adamw"
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_weight_decay: 0.01

  # Memory optimizations
  mixed_precision: "bf16"
  gradient_checkpointing: true
  enable_xformers: true

# Checkpointing
checkpointing:
  save_steps: 250
  checkpointing_steps: 250
  keep_last_n_checkpoints: 10
  save_on_epoch_end: true

# Validation
validation:
  enabled: true
  validation_steps: 250
  num_validation_samples: 4

  validation_prompts:
    - "A person walking towards camera in a park"
    - "Camera panning across a beautiful landscape"
    - "A car driving down a winding road"
    - "Autumn leaves falling and swirling"

# Logging
logging:
  report_to: "tensorboard"
  logging_steps: 10
  tensorboard_dir: "/workspace/LTX_video_training/output/trajectory_control_official/tensorboard"

# Acceleration (H100 specific)
acceleration:
  use_torch_compile: true
  torch_compile_mode: "max-autotune"
  use_flash_attention_2: true
EOF

echo "Created config: LTX-Video-Trainer/configs/trajectory_control_h100.yaml"

echo ""
echo -e "${GREEN}Step 5: Checking Dataset${NC}"
VIDEO_COUNT=$(ls -1 dataset/videos/*.mp4 2>/dev/null | wc -l)
TRAJ_COUNT=$(ls -1 dataset/trajectories/*.mp4 2>/dev/null | wc -l)
CAPTION_COUNT=$(ls -1 dataset/captions/*.txt 2>/dev/null | wc -l)

echo "Dataset status:"
echo "  - Videos: $VIDEO_COUNT"
echo "  - Trajectory visualizations: $TRAJ_COUNT"
echo "  - Captions: $CAPTION_COUNT"

if [ $VIDEO_COUNT -eq 0 ]; then
    echo -e "${YELLOW}Warning: No videos found! Run dataset preparation first:${NC}"
    echo "  python scripts/prepare_dataset_single_source.py \\"
    echo "      --input_dir /workspace/LTX_video_training/raw_videos \\"
    echo "      --output_dir /workspace/LTX_video_training/dataset \\"
    echo "      --use_existing_captions"
    exit 1
fi

echo ""
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo "Ready to Train"
echo "========================================="
echo ""
echo "Directory structure:"
echo "  /workspace/LTX_video_training/"
echo "  ├── dataset/                  (your preprocessed data)"
echo "  ├── LTX-Video-Trainer/       (official trainer)"
echo "  │   ├── data/trajectory_control/  (symlinks to dataset)"
echo "  │   └── configs/trajectory_control_h100.yaml"
echo "  └── output/                   (training outputs)"
echo ""
echo "To start training, run:"
echo "  ./scripts/train_with_official.sh"
echo ""
echo "Or manually:"
echo "  cd /workspace/LTX_video_training/LTX-Video-Trainer"
echo "  accelerate launch --mixed_precision=bf16 --num_processes=1 \\"
echo "      scripts/train.py \\"
echo "      configs/trajectory_control_h100.yaml"
echo ""
