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

# Create config file with all required fields
cat > LTX-Video-Trainer/configs/trajectory_control_h100.yaml << 'EOF'
# Trajectory Control IC-LoRA Training Configuration
# Optimized for H100 80GB

# Model configuration
model:
  model_source: "LTXV_13B_097_DEV"
  training_mode: "lora"
  load_checkpoint: null

# LoRA configuration - Memory optimized
lora:
  rank: 128  # Balanced for H100 memory and quality (654M params)
  alpha: 128
  dropout: 0.05
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"
    - "ff.net.0.proj"
    - "ff.net.2"

# Conditioning configuration
conditioning:
  mode: "reference_video"  # Use trajectory visualizations as reference
  first_frame_conditioning_p: 0.9
  reference_latents_dir: "reference_latents"

# Optimization configuration
optimization:
  learning_rate: 1.0e-4
  steps: 5000
  batch_size: 1  # Reduced for memory efficiency
  gradient_accumulation_steps: 8  # Maintain effective batch size of 8
  max_grad_norm: 1.0
  optimizer_type: "adamw"
  scheduler_type: "cosine"
  scheduler_params: {}
  enable_gradient_checkpointing: true

# Acceleration optimization
acceleration:
  mixed_precision_mode: "bf16"
  quantization: null
  load_text_encoder_in_8bit: false
  compile_with_inductor: false  # Disabled for stability
  compilation_mode: "max-autotune"

# Data configuration - REQUIRED FIELD
data:
  preprocessed_data_root: "/workspace/LTX_video_training/preprocessed_official"
  num_dataloader_workers: 16

# Validation configuration
validation:
  prompts:
    - "A person walking towards camera in a park"
    - "Camera panning across a beautiful landscape"
    - "A car driving down a winding road"
    - "Autumn leaves falling and swirling"
  reference_videos:
    - "/workspace/LTX_video_training/dataset/trajectories/video_000000_clip_000_traj.mp4"
    - "/workspace/LTX_video_training/dataset/trajectories/video_000001_clip_000_traj.mp4"
    - "/workspace/LTX_video_training/dataset/trajectories/video_000002_clip_000_traj.mp4"
    - "/workspace/LTX_video_training/dataset/trajectories/video_000003_clip_000_traj.mp4"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  video_dims: [704, 1216, 121]  # [width, height, frames]
  seed: 42
  inference_steps: 50
  interval: 10000  # Disabled (higher than total steps) to avoid OOM
  videos_per_prompt: 1
  guidance_scale: 3.5
  skip_initial_validation: true

# Checkpoint configuration
checkpoints:
  interval: 250  # Save every 250 steps
  keep_last_n: 10

# Flow matching configuration
flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {}

# HuggingFace Hub configuration
hub:
  push_to_hub: false
  hub_model_id: null

# W&B configuration
wandb:
  enabled: false
  project: "ltxv-trajectory-control"
  entity: null
  tags: ["trajectory", "ic-lora", "h100"]
  log_validation_videos: true

# General configuration
seed: 42
output_dir: "/workspace/LTX_video_training/output/trajectory_control_official"
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
