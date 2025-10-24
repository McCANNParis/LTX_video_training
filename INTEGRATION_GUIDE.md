# Integration Guide: Using Official LTX-Video Trainer for Trajectory Control

**Important**: The skeleton training script in this repo needs to be replaced with the official LTX-Video trainer which already supports IC-LoRA control training.

---

## 🎯 Solution: Use Official Trainer + Our Trajectory Preprocessing

Our trajectory-guided training system should use:
1. **Our preprocessing pipeline** - Extract trajectories and create visualizations
2. **Official LTX-Video trainer** - Train IC-LoRA control model

---

## 📋 Setup Steps

### 1. Install Official LTX-Video Trainer

```bash
cd /workspace
git clone https://github.com/Lightricks/LTX-Video-Trainer.git
cd LTX-Video-Trainer

# Install dependencies
pip install uv
uv pip install .
```

### 2. Keep Our Trajectory Preprocessing

Our preprocessing pipeline is unique and valuable - keep using it:

```bash
# Use our dataset preparation (already done)
cd /workspace/LTX_video_training
python scripts/prepare_dataset_single_source.py \
    --input_dir /workspace/LTX_video_training/raw_videos \
    --output_dir /workspace/LTX_video_training/dataset \
    --use_existing_captions
```

This creates:
- `dataset/videos/` - Original videos (training targets)
- `dataset/trajectories/` - Trajectory visualizations (conditioning)
- `dataset/captions/` - Your custom captions

### 3. Prepare Dataset for Official Trainer

The official trainer expects a specific format. Create a dataset manifest:

```bash
cd /workspace/LTX-Video-Trainer

# Create dataset directory
mkdir -p data/trajectory_control

# Link our processed data
ln -s /workspace/LTX_video_training/dataset/videos data/trajectory_control/videos
ln -s /workspace/LTX_video_training/dataset/trajectories data/trajectory_control/conditioning
ln -s /workspace/LTX_video_training/dataset/captions data/trajectory_control/captions
```

### 4. Create Training Configuration

Create `configs/trajectory_control_h100.yaml`:

```yaml
# Trajectory Control IC-LoRA Configuration
# Based on official depth control example

# Basic settings
output_dir: /workspace/LTX_video_training/output/trajectory_iclora_official
seed: 42

# Model settings
model_id: "Lightricks/LTX-Video"
model_variant: "13B-0.9.7"

# Training mode: IC-LoRA for control
training_mode: "ic_lora"  # In-Context LoRA

# Dataset settings
dataset:
  type: "video_folder"
  video_dir: "data/trajectory_control/videos"
  conditioning_dir: "data/trajectory_control/conditioning"  # Our trajectory visualizations
  caption_dir: "data/trajectory_control/captions"

  # Resolution and frames
  resolution: 704
  num_frames: 121
  fps: 30

  # Data preprocessing
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

  # Conditioning settings
  conditioning_type: "concat"  # Concatenate conditioning with latents
  conditioning_channels: 3  # RGB trajectory visualization
  conditioning_scale: 1.0

# Training settings
training:
  num_train_epochs: 100  # With 21 videos, use epochs
  train_batch_size: 4  # H100 can handle this
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
    - "A person walking towards camera"
    - "Camera panning across landscape"
    - "Car driving down road"
    - "Leaves falling and swirling"

# Logging
logging:
  report_to: "tensorboard"
  logging_steps: 10

  tensorboard_dir: "/workspace/LTX_video_training/output/trajectory_iclora_official/tensorboard"

# Acceleration (H100 specific)
acceleration:
  use_torch_compile: true
  torch_compile_mode: "max-autotune"
  use_flash_attention_2: true
```

### 5. Train with Official Trainer

```bash
cd /workspace/LTX-Video-Trainer

# Launch training
accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=1 \
    src/training/train.py \
    --config configs/trajectory_control_h100.yaml
```

---

## 🔄 Complete Workflow

```bash
# 1. Prepare trajectory dataset (our preprocessing)
cd /workspace/LTX_video_training
python scripts/prepare_dataset_single_source.py \
    --input_dir raw_videos \
    --output_dir dataset \
    --use_existing_captions

# 2. Set up official trainer
cd /workspace
git clone https://github.com/Lightricks/LTX-Video-Trainer.git
cd LTX-Video-Trainer
pip install uv && uv pip install .

# 3. Link our data
mkdir -p data/trajectory_control
ln -s /workspace/LTX_video_training/dataset/videos data/trajectory_control/videos
ln -s /workspace/LTX_video_training/dataset/trajectories data/trajectory_control/conditioning
ln -s /workspace/LTX_video_training/dataset/captions data/trajectory_control/captions

# 4. Create config (copy from above)
# Edit configs/trajectory_control_h100.yaml

# 5. Train!
accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=1 \
    src/training/train.py \
    --config configs/trajectory_control_h100.yaml
```

---

## 📊 What Makes This Work

### Our Contribution:
✅ **Trajectory extraction** from videos
✅ **Trajectory visualization** (multi-channel encoding)
✅ **Single-source approach** (same videos for both)
✅ **Custom caption integration**
✅ **H100-optimized preprocessing**

### Official Trainer Contribution:
✅ **Proper LTX-Video model loading**
✅ **IC-LoRA implementation**
✅ **Training loop and optimization**
✅ **Checkpoint management**
✅ **Validation and metrics**

### Combined Result:
🎯 **Trajectory-guided video generation** with proper LTX-Video integration!

---

## 🎨 After Training

### Use the Trained Model

```python
from ltxvideo import LTXVideoPipeline

# Load base model + your trajectory control LoRA
pipe = LTXVideoPipeline.from_pretrained("Lightricks/LTX-Video")
pipe.load_lora_weights("/workspace/LTX_video_training/output/trajectory_iclora_official/checkpoint-5000")

# Generate video with trajectory control
video = pipe(
    prompt="A person walking in a park",
    control_image=trajectory_visualization,  # Your trajectory viz
    num_frames=121,
    guidance_scale=7.5
).frames
```

---

## 📚 References

- **Official Trainer**: https://github.com/Lightricks/LTX-Video-Trainer
- **IC-LoRA Training Docs**: https://github.com/Lightricks/LTX-Video-Trainer/blob/main/docs/training-modes.md
- **Depth Control Example**: https://huggingface.co/Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7
- **Pose Control Example**: https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7

---

## ⚡ Quick Commands for RunPod

```bash
# Full setup and training
cd /workspace/LTX_video_training && \
python scripts/prepare_dataset_single_source.py \
    --input_dir raw_videos \
    --output_dir dataset \
    --use_existing_captions && \
cd /workspace && \
git clone https://github.com/Lightricks/LTX-Video-Trainer.git && \
cd LTX-Video-Trainer && \
pip install uv && uv pip install . && \
mkdir -p data/trajectory_control && \
ln -s /workspace/LTX_video_training/dataset/videos data/trajectory_control/videos && \
ln -s /workspace/LTX_video_training/dataset/trajectories data/trajectory_control/conditioning && \
ln -s /workspace/LTX_video_training/dataset/captions data/trajectory_control/captions
```

Then create the config file and launch training!

---

## 🎯 Summary

**Don't use our skeleton trainer** - it's incomplete.
**Use the official LTX-Video trainer** - it already supports IC-LoRA!
**Keep using our preprocessing** - trajectory extraction is unique and valuable.

**Result**: Professional trajectory-guided video generation with proper LTX-Video integration! 🚀
