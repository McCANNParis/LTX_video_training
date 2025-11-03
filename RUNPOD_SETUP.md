# RunPod Setup Guide - Realism LoRA Training

Quick guide to set up realism LoRA training on your RunPod container.

## Step 1: Clone/Pull the Repository

```bash
cd /workspace
git clone https://github.com/McCANNParis/LTX_video_training.git
cd LTX_video_training

# Switch to realism training branch
git checkout claude/lora-training-realism-011CUjmk3msVokkZVQwPepWK
git pull
```

## Step 2: Clone Official LTX-Video-Trainer

```bash
cd /workspace/LTX_video_training
git clone https://github.com/Lightricks/LTX-Video-Trainer.git

# Restore the realism config
git checkout LTX-Video-Trainer/configs/realism_lora.yaml
```

## Step 3: Install Dependencies

```bash
cd /workspace/LTX_video_training
./install_dependencies.sh
```

This will install:
- uv package manager
- LTX-Video-Trainer and all Python dependencies
- All required libraries (torch, diffusers, transformers, etc.)

**⏱️ Estimated time:** 5-10 minutes

## Step 4: Prepare Your Dataset

```bash
# Create the dataset folder
mkdir -p /workspace/LTX_video_training/raw_videos

# Upload your videos and captions to raw_videos/
# Each .mp4 must have a matching .txt file with the same name
```

Example structure:
```
raw_videos/
├── video_001.mp4
├── video_001.txt
├── video_002.mp4
├── video_002.txt
└── ...
```

## Step 5: Start Training

```bash
cd /workspace/LTX_video_training

# Option 1: Interactive guide (recommended)
./scripts/quick_start_realism.sh

# Option 2: Manual steps
./scripts/prepare_for_realism_training.sh
export TORCH_COMPILE_DISABLE=1
./scripts/train_realism_lora.sh
```

## Quick Check

Run this anytime to check your setup status:
```bash
cd /workspace/LTX_video_training
./check_setup.sh
```

---

## Troubleshooting

### "No module named 'typer'" or similar errors
Run the installation script again:
```bash
./install_dependencies.sh
```

### Disk space issues
Clear pip cache:
```bash
rm -rf /root/.cache/*
```

### Need to update the code
```bash
cd /workspace/LTX_video_training
git pull
```

---

## Complete Command Reference

```bash
# 1. Initial setup (one time)
cd /workspace
git clone https://github.com/McCANNParis/LTX_video_training.git
cd LTX_video_training
git checkout claude/lora-training-realism-011CUjmk3msVokkZVQwPepWK
git clone https://github.com/Lightricks/LTX-Video-Trainer.git
git checkout LTX-Video-Trainer/configs/realism_lora.yaml
./install_dependencies.sh

# 2. Add your data
mkdir -p raw_videos
# Upload videos and captions

# 3. Train
./scripts/quick_start_realism.sh

# 4. Test (after training)
python scripts/test_realism_lora.py --compare
```

That's it! 🚀
