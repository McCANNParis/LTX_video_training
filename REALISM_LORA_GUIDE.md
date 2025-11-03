# Simple Realism LoRA Training Guide

This guide covers training a **simple text-to-video LoRA** to improve realism in LTX-Video generations, **without trajectory conditioning**.

---

## Overview

### What's Different from Trajectory Training?

| Feature | Trajectory Control (IC-LoRA) | Realism LoRA (This Guide) |
|---------|------------------------------|---------------------------|
| **Conditioning** | Reference video + text | Text only |
| **Training Mode** | IC-LoRA | Standard LoRA |
| **Use Case** | Motion control | Realism enhancement |
| **Dataset Requirements** | Videos + trajectories + captions | Videos + captions |
| **Inference** | Needs reference videos | Works with text prompts alone |

### Why Realism LoRA?

- **Simpler setup**: No trajectory visualization needed
- **Text-only inference**: Works like standard text-to-video models
- **Realism focus**: Fine-tune on your high-quality footage to improve visual fidelity
- **More broadly applicable**: Can be used for any text prompt without reference videos

---

## Quick Start

### 1. Setup (One-Time)

```bash
# Setup the official trainer (if not already done)
./scripts/setup_official_trainer.sh
```

### 2. Prepare Your Dataset

Your dataset should have the following structure:

```
raw_videos/           # All your training data in one folder
├── video_001.mp4     # Video file
├── video_001.txt     # Caption file (same name as video)
├── video_002.mp4
├── video_002.txt
├── video_003.mp4
├── video_003.txt
└── ...
```

**Important:** Each `.mp4` video file must have a corresponding `.txt` caption file with the exact same name.

**Video Requirements:**
- Format: MP4
- Resolution: 704x1216 (width x height)
- Duration: 4-10 seconds
- Frames: 121 frames (at 30 FPS ≈ 4 seconds)
- FPS: 30

**Caption Requirements:**
- Format: Plain text files (.txt)
- One caption per video
- Filename must match video exactly (e.g., `video_001.mp4` → `video_001.txt`)
- Placed in the same `raw_videos/` directory
- Content: Descriptive text with focus on realism aspects
  - Good: "A person walking in a park, photorealistic, natural lighting, detailed features"
  - Avoid: "A person walking in a park" (too generic)

### 3. Preprocess Dataset

```bash
./scripts/prepare_for_realism_training.sh
```

This will:
- Create `dataset_realism.json` mapping videos to captions
- Extract VAE latents from videos
- Generate caption embeddings
- Save preprocessed data to `preprocessed_realism/`

**Expected output:**
```
✓ Created dataset_realism.json with N entries
✓ Preprocessing complete!
  preprocessed_realism/.precomputed/
```

### 4. Train

```bash
# Disable torch.compile for stability (recommended)
export TORCH_COMPILE_DISABLE=1

# Start training
./scripts/train_realism_lora.sh
```

**Training Progress:**
- Checkpoints saved every 250 steps
- Total training: 5000 steps
- Time: ~17.5 hours on single H100 (or ~4-5 hours on 4x H100)
- Output: `output/realism_lora/checkpoints/`

**Monitor training:**
```bash
# Watch log
tail -f logs/train_realism_*.log

# Watch GPU usage
watch -n 1 nvidia-smi

# TensorBoard (optional)
tensorboard --logdir output/realism_lora/tensorboard
```

### 5. Test Your LoRA

```bash
# Test with LoRA only
python scripts/test_realism_lora.py \
    --lora_path output/realism_lora/checkpoints/lora_weights_step_05000.safetensors \
    --prompt "A person walking in a park, photorealistic, high detail"

# Compare base model vs LoRA side-by-side
python scripts/test_realism_lora.py \
    --lora_path output/realism_lora/checkpoints/lora_weights_step_05000.safetensors \
    --prompt "A person walking in a park, photorealistic, high detail" \
    --compare
```

---

## Configuration

### Training Config

Location: `LTX-Video-Trainer/configs/realism_lora.yaml`

**Key settings:**

```yaml
# LoRA settings
lora:
  rank: 128           # 654M parameters
  alpha: 128
  dropout: 0.05

# Conditioning - TEXT ONLY
conditioning:
  mode: "text_only"   # No reference videos
  first_frame_conditioning_p: 0.0

# Training
optimization:
  learning_rate: 1.0e-4
  steps: 5000
  batch_size: 4
  gradient_accumulation_steps: 2  # Effective batch = 8

# Hardware
acceleration:
  mixed_precision_mode: "bf16"
  compile_with_inductor: false
```

### Adjusting Training

**For smaller datasets (< 20 videos):**
- Reduce `steps` to 3000-4000
- Lower `learning_rate` to 5.0e-5

**For larger datasets (> 50 videos):**
- Increase `steps` to 7000-10000
- Keep `learning_rate` at 1.0e-4

**For lower memory GPUs:**
- Reduce `batch_size` to 2 or 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable more aggressive gradient checkpointing

**For better quality:**
- Increase `rank` to 256 (more parameters, needs more memory)
- Train for more steps (7000-10000)

---

## Dataset Best Practices

### Video Quality

1. **Resolution**: Native 704x1216 is ideal, but you can resize
2. **Compression**: Use high-quality encoding (H.264 with high bitrate)
3. **Content**: Focus on realistic scenarios matching your use case
4. **Diversity**: Include various lighting, scenes, angles, and subjects
5. **Motion**: Include both static and dynamic content

### Caption Quality

**Good captions include:**
- Visual details: "detailed textures", "natural lighting"
- Camera movement: "camera pans left", "static shot"
- Realism descriptors: "photorealistic", "high detail", "natural"
- Scene description: "outdoor park", "urban street", "indoor office"

**Example good captions:**
```
A person walking towards camera in a park, photorealistic, natural daylight, detailed facial features, smooth motion
Camera slowly panning across a mountain landscape, high detail, realistic textures, golden hour lighting
Close-up of hands typing on keyboard, shallow depth of field, natural office lighting, photorealistic
```

**Avoid:**
- Generic descriptions: "A person walking"
- Overly artistic language: "majestic", "breathtaking"
- Fictional content: "dragon flying", "magic spell"

---

## Inference Best Practices

### Text Prompts

**Effective prompts for realism LoRA:**
```
"A person walking in a park, photorealistic, natural lighting, high detail"
"Camera panning across cityscape, realistic motion, detailed architecture"
"Close-up of a face, natural expression, detailed skin texture, soft lighting"
```

**Tips:**
- Include realism keywords: "photorealistic", "natural", "detailed"
- Describe lighting: "natural lighting", "soft shadows", "golden hour"
- Specify camera motion: "static shot", "slow pan", "dolly forward"
- Use negative prompts: "cartoon, animated, unrealistic, distorted"

### Generation Parameters

```python
# Recommended settings for realism LoRA
guidance_scale = 3.5           # Higher for text-only (vs 1.0 for IC-LoRA)
num_inference_steps = 50       # Good balance of quality/speed
decode_timestep = 0.05         # Reduces green tint
image_cond_noise_scale = 0.025 # Reduces color artifacts
```

### Checkpoint Selection

- **Step 2500-3000**: Good for preliminary tests
- **Step 4000-4500**: Usually best quality
- **Step 5000+**: Risk of overfitting on small datasets

Test multiple checkpoints to find the best one for your use case.

---

## Troubleshooting

### Training Issues

**Out of Memory:**
- Reduce `batch_size` in config
- Ensure `gradient_checkpointing: true`
- Use `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`

**Training too slow:**
- Use multiple GPUs (auto-detected by script)
- Ensure `mixed_precision_mode: "bf16"`
- Check GPU utilization with `nvidia-smi`

**Poor quality results:**
- Train for more steps (7000-10000)
- Check caption quality (add more detail)
- Increase dataset size (50+ videos ideal)
- Try higher LoRA rank (256)

### Inference Issues

**Green tint in output:**
- Already handled by using `decode_timestep=0.05`
- Use `LTXConditionPipeline` (not `LTXPipeline`)
- Ensure VAE is in bfloat16

**LoRA not affecting output:**
- Text-only LoRA should show clear differences
- Try more descriptive prompts with realism keywords
- Ensure you're loading the correct checkpoint
- Check that LoRA keys are loading (script shows count)

**Videos look identical to base:**
- May need longer training
- Check if dataset is too similar to base model's training data
- Increase LoRA rank or learning rate

---

## Advanced Usage

### Multi-GPU Training

Automatically detected and used:
```bash
# Will use all available GPUs
./scripts/train_realism_lora.sh
```

Effective batch size scales: `batch_size × gradient_accumulation × num_gpus`

### Converting for ComfyUI

```bash
python scripts/convert_lora_for_comfyui.py \
    --input output/realism_lora/checkpoints/lora_weights_step_05000.safetensors \
    --output realism_lora_comfyui.safetensors \
    --conversion-type diffusers
```

### Upload to HuggingFace

```bash
python scripts/upload_lora_to_hf.py \
    --lora_path output/realism_lora/checkpoints/lora_weights_step_05000.safetensors \
    --repo_id your-username/ltx-realism-lora \
    --token YOUR_HF_TOKEN
```

---

## Comparison with Trajectory Control

### When to Use Each

**Use Realism LoRA when:**
- You want to improve general video quality/realism
- You need text-only inference
- You don't have trajectory visualizations
- You want simpler workflow

**Use Trajectory Control (IC-LoRA) when:**
- You need precise motion control
- You have reference trajectory videos
- You're building a motion-controlled generation system
- You can provide reference videos at inference time

### Can I Use Both?

**No** - you cannot load both LoRAs simultaneously. Choose one based on your use case:
- Realism LoRA: Better for general quality improvement
- IC-LoRA: Better for motion control applications

---

## File Structure

```
LTX_video_training/
├── REALISM_LORA_GUIDE.md                    # This guide
├── LTX-Video-Trainer/
│   └── configs/
│       └── realism_lora.yaml                # Training config
├── scripts/
│   ├── prepare_for_realism_training.sh      # Dataset preprocessing
│   ├── train_realism_lora.sh                # Training launcher
│   ├── test_realism_lora.py                 # Inference testing
│   └── quick_start_realism.sh               # Interactive quick start
├── raw_videos/                              # Your training data
│   ├── video_001.mp4                        # Video files
│   ├── video_001.txt                        # Caption files (matching names)
│   ├── video_002.mp4
│   ├── video_002.txt
│   └── ...
├── dataset_realism.json                     # Dataset manifest (auto-generated)
├── preprocessed_realism/                    # Preprocessed data (auto-generated)
├── output/
│   └── realism_lora/
│       └── checkpoints/                     # LoRA weights
└── logs/                                    # Training logs
```

---

## Next Steps

1. **Gather high-quality video footage** matching your target use case
2. **Write detailed captions** focusing on realism aspects
3. **Place both in `raw_videos/`** with matching filenames (video_001.mp4 + video_001.txt)
4. **Run the quick start guide**: `./scripts/quick_start_realism.sh`
5. **Test checkpoints** to find the best one
6. **Iterate**: Adjust config based on results and retrain if needed

Good luck with your realism LoRA training! 🎥
