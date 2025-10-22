# Best Practices for Trajectory-Guided Video Training

## Dataset Preparation

### 1. Video Quality Standards

**Resolution**
- Minimum: 1080p (1920×1080)
- Recommended: 4K (3840×2160)
- Training: Downsampled to 704×1216 or 1024×1024

**Frame Rate**
- Source: 30fps or higher
- Training: 24-30fps (standard cinema rates)
- Smooth motion is critical for trajectory extraction

**Bitrate & Compression**
- Use high bitrate (20+ Mbps for 1080p)
- Avoid heavily compressed videos
- H.264 or H.265 codec recommended

**Lighting**
- Well-lit scenes
- Consistent lighting within clips
- Avoid extreme contrast or darkness

**Motion Quality**
- Natural motion blur (not excessive)
- Stable footage (minimal camera shake unless intentional)
- Clear motion patterns

### 2. Dataset Composition

**Size Requirements**
- Minimum: 100 videos (testing only)
- Good: 500-1,000 videos
- Excellent: 2,000+ videos

**Diversity Targets**

```
Motion Types:
├── Camera Motion (30%)
│   ├── Pan (left/right)
│   ├── Tilt (up/down)
│   ├── Dolly (forward/backward)
│   ├── Zoom
│   └── Combined movements
│
├── Object Motion (30%)
│   ├── Translation (moving objects)
│   ├── Rotation
│   ├── Deformation (non-rigid)
│   └── Complex interactions
│
├── Scene Dynamics (20%)
│   ├── Nature (wind, water, leaves)
│   ├── Crowds
│   ├── Animals
│   └── Environmental changes
│
└── Special Cases (20%)
    ├── Occlusions
    ├── Fast motion
    ├── Multi-object scenes
    └── Static scenes (as negatives)
```

**Content Diversity**
- Indoor and outdoor scenes
- Day and night
- Different seasons
- Various subjects (people, vehicles, nature, objects)
- Different camera angles and perspectives

### 3. Data Cleaning

**Remove:**
- Corrupted videos
- Videos with major artifacts
- Extremely short clips (< 2 seconds)
- Duplicate or near-duplicate content
- Videos with text overlays or watermarks

**Fix:**
- Stabilize shaky footage (optional, depends on use case)
- Color correct inconsistent videos
- Trim black frames at start/end

### 4. Caption Quality

**Good Captions Include:**
```
1. Motion description
   - "slow pan left"
   - "person walking forward"
   - "camera zoom in"

2. Scene description
   - "in a park"
   - "urban street"
   - "indoor kitchen"

3. Subject description
   - "a woman in red dress"
   - "black sports car"
   - "autumn trees"

4. Style/lighting
   - "cinematic lighting"
   - "bright daylight"
   - "moody atmosphere"
```

**Example: Good vs Bad**

✅ Good: "Cinematic slow pan left across a misty mountain landscape at sunrise, camera moving smoothly"

❌ Bad: "video123.mp4"

❌ Bad: "Mountain"

## Training Configuration

### 1. Hyperparameter Guidelines

**Learning Rate**
```yaml
# Conservative (recommended for first runs)
learning_rate: 8e-5

# Moderate
learning_rate: 1e-4

# Aggressive (only if you know what you're doing)
learning_rate: 2e-4
```

**LoRA Rank**
```yaml
# Low (faster, less capacity)
rank: 64

# Medium (balanced)
rank: 128

# High (slower, more capacity)
rank: 256

# Very high (for complex motion patterns)
rank: 512
```

**Training Steps**
```yaml
# Quick test
max_train_steps: 1000

# Standard
max_train_steps: 10000

# High quality
max_train_steps: 20000

# Production
max_train_steps: 50000+
```

### 2. Memory Optimization

**For 24GB GPUs (3090, 4090)**
```yaml
model:
  transformer:
    load_in_8bit: true
  vae:
    load_in_8bit: true

optimization:
  train_batch_size: 1
  gradient_accumulation_steps: 16
  use_8bit_adam: true
```

**For 48GB GPUs (A6000)**
```yaml
model:
  transformer:
    load_in_8bit: false
  vae:
    load_in_8bit: true

optimization:
  train_batch_size: 1
  gradient_accumulation_steps: 8
```

**For 80GB GPUs (A100)**
```yaml
model:
  transformer:
    load_in_8bit: false
  vae:
    load_in_8bit: false

optimization:
  train_batch_size: 2
  gradient_accumulation_steps: 4
```

### 3. Progressive Training Strategy

**Stage 1: Coarse Motion (0-2000 steps)**
```yaml
resolution: [512, 512]
num_frames: 61
learning_rate: 1e-4
lora_rank: 128
```
Goal: Learn basic motion patterns

**Stage 2: Medium Fidelity (2000-5000 steps)**
```yaml
resolution: [704, 1216]
num_frames: 91
learning_rate: 5e-5
lora_rank: 256
```
Goal: Refine motion, increase quality

**Stage 3: High Quality (5000-10000 steps)**
```yaml
resolution: [704, 1216]
num_frames: 121
learning_rate: 2e-5
lora_rank: 256
```
Goal: Fine-tune details, maximize quality

### 4. Validation Strategy

**Frequency**
- Validate every 500 steps
- Save best model based on metrics
- Keep last 5 checkpoints

**Validation Prompts**
Create diverse prompts covering:
- Different motion types
- Different subjects
- Different styles
- Edge cases

**Metrics to Track**
```python
# Primary metrics
- loss (should decrease)
- trajectory_fidelity (> 0.7 is good)
- temporal_consistency (> 0.8 is excellent)

# Secondary metrics
- motion_smoothness (higher is better)
- physical_plausibility (> 0.75 is good)
- psnr (if using ground truth)
```

## Training Monitoring

### 1. Early Warning Signs

**Overfitting**
- Training loss decreases but validation loss increases
- Generated videos look identical to training data
- Poor generalization to new prompts

**Fix:**
- Increase dataset size
- Add more augmentation
- Reduce model capacity (lower LoRA rank)
- Add regularization (LoRA dropout)

**Underfitting**
- Both training and validation loss remain high
- Generated videos don't follow trajectories
- Poor motion quality

**Fix:**
- Increase model capacity (higher LoRA rank)
- Train longer
- Increase learning rate (carefully)
- Check dataset quality

**Instability**
- Loss spikes suddenly
- NaN values appear
- Generated videos have artifacts

**Fix:**
- Lower learning rate
- Check for corrupted data
- Increase gradient clipping
- Use gradient checkpointing

### 2. Good Training Curves

**Loss**
```
10000 |
      |  \
 5000 |   \__
      |      \_____
 1000 |           \___
      |               \____
    0 +--------------------
      0   2k   4k   6k   8k   10k
```
Smooth decrease, plateaus after ~8k steps

**Trajectory Fidelity**
```
1.0 |              ________
    |          ___/
0.7 |      ___/
    |   __/
0.5 | _/
    |/
0.0 +--------------------
    0   2k   4k   6k   8k   10k
```
Should reach 0.7+ by 5k steps

## Inference Best Practices

### 1. Generation Parameters

**Guidance Scale**
```python
# Low (more creative, less adherent)
guidance_scale = 3.0

# Medium (balanced)
guidance_scale = 7.5

# High (very adherent to prompt/trajectory)
guidance_scale = 15.0
```

**Inference Steps**
```python
# Fast (lower quality)
num_inference_steps = 25

# Standard
num_inference_steps = 50

# High quality (slower)
num_inference_steps = 100
```

### 2. Trajectory Quality

**For Best Results:**
1. Use high-quality reference videos
2. Match resolution between reference and output
3. Ensure trajectories are smooth (apply physics smoothing)
4. Avoid extremely fast or discontinuous motion

**Trajectory Visualization Types:**
```python
# For general motion
visualization_type = 'multi'  # RGB channels for X, Y, magnitude

# For depth-aware motion
visualization_type = 'depth'

# For simple 2D motion
visualization_type = 'flow'
```

### 3. Prompt Engineering

**Good Prompts:**
```
"A cinematic shot of [subject] [action], [camera motion], [lighting], [style]"

Examples:
- "A cinematic shot of a woman walking towards camera in slow motion,
   smooth dolly forward, golden hour lighting, film grain aesthetic"

- "Aerial drone footage panning right over a coastal city at sunset,
   cinematic composition, high detail"
```

**Prompt Structure:**
1. Style indicator (cinematic, documentary, etc.)
2. Subject description
3. Action/motion description
4. Camera movement
5. Lighting/atmosphere
6. Technical details (optional)

## Troubleshooting

### Issue: Poor Trajectory Following

**Symptoms:**
- Generated videos ignore reference motion
- Motion is random or unrelated to trajectory

**Solutions:**
1. Increase conditioning probability:
   ```yaml
   conditioning_prob: 0.95  # from 0.85
   ```

2. Increase LoRA rank:
   ```yaml
   lora_rank: 512  # from 256
   ```

3. Train longer

4. Check trajectory visualization quality

5. Verify trajectory_fidelity metric during training

### Issue: Temporal Inconsistency

**Symptoms:**
- Flickering between frames
- Objects appear/disappear
- Sudden style changes

**Solutions:**
1. Enable temporal consistency loss:
   ```yaml
   experimental:
     use_temporal_consistency_loss: true
     temporal_consistency_loss_weight: 0.05
   ```

2. Increase num_frames during training

3. Use EMA (exponential moving average):
   ```yaml
   optimization:
     use_ema: true
     ema_decay: 0.9999
   ```

4. Lower learning rate

### Issue: Out of Memory

**Solutions:**
1. Enable quantization:
   ```yaml
   model:
     transformer:
       load_in_8bit: true
   ```

2. Increase gradient accumulation:
   ```yaml
   optimization:
     gradient_accumulation_steps: 16  # or higher
   ```

3. Reduce resolution or num_frames

4. Enable gradient checkpointing:
   ```yaml
   acceleration:
     gradient_checkpointing: true
   ```

### Issue: Slow Training

**Solutions:**
1. Enable PyTorch 2.0 compilation:
   ```yaml
   acceleration:
     enable_torch_compile: true
   ```

2. Use multiple GPUs

3. Optimize dataloader:
   ```yaml
   dataset:
     dataloader_num_workers: 8
     pin_memory: true
     prefetch_factor: 2
   ```

4. Reduce validation frequency

## Production Deployment

### 1. Model Optimization

**Export for Inference:**
```python
# Merge LoRA weights into base model
from peft import PeftModel

base_model = load_base_model()
lora_model = PeftModel.from_pretrained(base_model, "path/to/lora")
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("path/to/merged")
```

**Quantization:**
```python
# INT8 quantization for deployment
from optimum.quanto import quantize, freeze

quantize(model, weights=torch.int8, activations=torch.int8)
freeze(model)
```

### 2. API Deployment

**Basic Flask API:**
```python
from flask import Flask, request, send_file
from trajectory_guided_inference import TrajectoryGuidedVideoGenerator

app = Flask(__name__)
generator = TrajectoryGuidedVideoGenerator(model_path, lora_path)

@app.route('/generate', methods=['POST'])
def generate():
    image = request.files['image']
    trajectory = request.files['trajectory']
    prompt = request.form['prompt']

    output = generator.generate_from_image_and_reference(
        input_image=image,
        reference_video=trajectory,
        prompt=prompt
    )

    return send_file(output, mimetype='video/mp4')
```

### 3. Monitoring

**Track in Production:**
- Generation time
- Memory usage
- Output quality metrics
- User feedback
- Error rates

## Community Resources

### Recommended Reading
- LTX Video paper and documentation
- Trace Anything paper
- LoRA: Low-Rank Adaptation paper
- Diffusion models fundamentals

### Useful Tools
- **Weights & Biases**: Experiment tracking
- **Gradio**: Quick UI prototyping
- **ComfyUI**: Node-based workflow
- **Automatic1111**: Alternative inference

### Datasets
- **Pexels**: Free stock footage
- **Kinetics-700**: Action recognition
- **UCF-101**: Action videos
- **Custom footage**: Always best for specific use cases

---

**Remember**: High-quality data beats complex algorithms. Spend time curating your dataset!
