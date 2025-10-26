# LTX Video Training - Trajectory Control

Production-ready system for training LTX Video 13B with trajectory-guided motion control on RunPod. Optimized for H100 GPUs.

## Quick Start (3 Commands)

```bash
# 1. Setup official trainer
./scripts/setup_official_trainer.sh

# 2. Prepare your dataset (if not already done)
./scripts/prepare_for_official_trainer.sh

# 3. Train!
export TORCH_COMPILE_DISABLE=1 && ./scripts/train_with_official.sh
```

All files stay in `/workspace/LTX_video_training` - completely self-contained!

## What This Does

Trains LTX Video to generate videos conditioned on:
- **Input image**: First frame content
- **Trajectory visualization**: Motion guidance (from reference video)
- **Text prompt**: Semantic description

Result: Generate videos from a single image + desired motion trajectory.

## System Requirements

- **GPU**: H100 80GB (optimized configuration)
- **Storage**: ~50GB for model + dataset
- **Platform**: RunPod (or any Linux with CUDA)

## Dataset Preparation

Your dataset should contain:
```
dataset/
├── videos/           # Your training videos (MP4)
├── trajectories/     # Motion trajectory visualizations (MP4)
└── captions/         # Text descriptions (TXT)
```

Already have 21 preprocessed videos? Skip preparation and go straight to training!

### Preparing New Data

1. Place raw videos in `/workspace/LTX_video_training/raw_videos/`

2. Run preprocessing:
```bash
./scripts/prepare_for_official_trainer.sh
```

This will:
- Extract VAE latents from videos
- Generate trajectory visualizations
- Create caption embeddings
- Set up proper directory structure

**Video Requirements:**
- Format: MP4
- Resolution: 704x1216 (or will be resized)
- Duration: 4-10 seconds
- Frames: 121 frames minimum
- FPS: 30

## Training Configuration

Optimized settings for H100 (in `LTX-Video-Trainer/configs/trajectory_control_h100.yaml`):

```yaml
# Memory-optimized for H100 80GB
lora:
  rank: 128              # Balanced quality/speed (654M params)
  alpha: 128

optimization:
  learning_rate: 1.0e-4
  steps: 5000
  batch_size: 1          # Prevents OOM
  gradient_accumulation_steps: 8  # Effective batch size 8

acceleration:
  mixed_precision_mode: "bf16"
  compile_with_inductor: false    # Disabled for stability
  gradient_checkpointing: true

validation:
  interval: 5000         # Only at end to avoid OOM
  skip_initial_validation: true
```

**Training Speed**: ~106 seconds/step (17.5 hours for 5000 steps)

## Key Scripts

### Setup & Preparation
- `setup_official_trainer.sh` - Clone and install official LTX-Video trainer
- `prepare_for_official_trainer.sh` - Preprocess dataset for training
- `regenerate_captions.sh` - Generate caption embeddings with correct dtype

### Training
- `train_with_official.sh` - Launch training with optimal settings
- `fix_preprocessing_structure.sh` - Fix nested directory structure if needed

### Configuration
- `LTX-Video-Trainer/configs/trajectory_control_h100.yaml` - H100 training config

## Resuming Training

Training automatically resumes from latest checkpoint:

```bash
export TORCH_COMPILE_DISABLE=1 && ./scripts/train_with_official.sh
```

Checkpoints saved every 250 steps in:
```
/workspace/LTX_video_training/output/trajectory_control_official/
```

## Monitoring Training

Watch training progress:
```bash
# Monitor logs
tail -f /workspace/LTX_video_training/logs/train_official_*.log

# Watch GPU usage
watch -n 1 nvidia-smi

# View TensorBoard
tensorboard --logdir /workspace/LTX_video_training/output/trajectory_control_official/tensorboard
```

## Troubleshooting

### Out of Memory (OOM)

If you hit OOM during training:

1. **Reduce LoRA rank** (edit the config file directly):
```bash
# Edit config
vim LTX-Video-Trainer/configs/trajectory_control_h100.yaml

# Change lora.rank from 128 to 64
lora:
  rank: 64   # Lower from 128
  alpha: 64
```

2. **Increase gradient accumulation**:
```yaml
optimization:
  gradient_accumulation_steps: 16  # Higher from 8
```

3. **Disable validation during training**:
```yaml
validation:
  interval: 10000  # Very high number
```

### Validation OOM

Already configured to skip validation until the end. If still having issues:

```bash
# Edit config to disable validation entirely
vim LTX-Video-Trainer/configs/trajectory_control_h100.yaml
# Set: validation.interval: 999999
```

### Caption Embedding Errors

If you see dtype mismatch errors, regenerate embeddings:

```bash
./scripts/regenerate_captions.sh
```

This ensures attention masks are boolean (not int64).

### Training Too Slow

Current setup: ~106 sec/step

To speed up (with quality tradeoff):
- Reduce num_frames: 121 → 91 or 61
- Reduce resolution: 704x1216 → 512x896
- Reduce LoRA rank: 128 → 64

### Nested Directory Structure

If preprocessing creates `preprocessed/dataset/videos/` instead of `preprocessed/videos/`:

```bash
./scripts/fix_preprocessing_structure.sh
```

## Project Structure

```
LTX_video_training/
├── scripts/                    # All helper scripts
│   ├── setup_official_trainer.sh
│   ├── train_with_official.sh
│   ├── prepare_for_official_trainer.sh
│   ├── regenerate_captions.sh
│   └── fix_preprocessing_structure.sh
├── dataset/                    # Your training data
│   ├── videos/
│   ├── trajectories/
│   └── captions/
├── LTX-Video-Trainer/         # Official trainer (auto-cloned)
│   ├── configs/trajectory_control_h100.yaml
│   └── scripts/train.py
├── output/                     # Training outputs & checkpoints
├── logs/                       # Training logs
└── README.md                   # This file
```

## Hardware Tested

- **H100 80GB**: Fully working (batch_size=1, rank=128)
- **Memory usage**: ~71GB peak during training
- **Training speed**: ~106 sec/step

## Production Tips

1. **Start small**: Test with 5-10 videos first
2. **Monitor closely**: Watch first 100 steps for OOM
3. **Save checkpoints**: Every 250 steps (already configured)
4. **Resume friendly**: Training auto-resumes from latest checkpoint
5. **Validate at end**: Skip validation during training to save memory

## Getting Help

**Common Issues:**
- OOM errors: See Troubleshooting section above
- Slow training: Reduce resolution or LoRA rank
- Caption errors: Run `regenerate_captions.sh`
- Directory structure: Run `fix_preprocessing_structure.sh`

**Configuration Issues:**
- Config validation errors: Re-run `./scripts/setup_official_trainer.sh` to regenerate config
- The setup script creates a complete config with all required fields

## Version Info

- **LTX Video**: 13B-0.9.7
- **Training Mode**: IC-LoRA
- **Precision**: bfloat16
- **Optimizer**: AdamW
- **Last Updated**: 2025-10-26

---

**Status**: Production ready, tested on H100 80GB

Successfully trained to step 249 with loss 0.1860 before optimization.
