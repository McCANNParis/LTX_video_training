# Trajectory-Guided LTX Video Training System

A complete system for training LTX Video 13B with LoRA on trajectory-guided motion control, enabling motion-guided img2video generation inspired by Trace Anything.

## Overview

This system enables you to:
- **Train** LTX Video with IC-LoRA to condition on trajectory fields
- **Generate** videos from a single image + desired motion trajectory
- **Transfer** motion from one video to another subject
- **Control** camera and object motion precisely in generated videos

The approach combines:
- **LTX Video 13B**: State-of-the-art video generation model
- **IC-LoRA**: In-Context LoRA for trajectory conditioning
- **Trajectory Fields**: Dense 3D motion representations (inspired by Trace Anything)

## 💡 Single-Source Training Approach

**KEY INSIGHT**: Use the **same high-quality videos** for both trajectory extraction AND training targets!

```
High-Quality Video → [Extract Trajectory] → Trajectory Visualization (conditioning)
                  ↓
                  → Original Video (training target)
```

**Benefits:**
- ✅ **Perfect Alignment**: Trajectory and content perfectly synchronized
- ✅ **Realistic Physics**: Model learns real-world motion → appearance mapping
- ✅ **Simple & Efficient**: One dataset instead of two
- ✅ **Scalable**: Just add more source videos

The model learns: `(first_frame + trajectory_viz + caption) → realistic_video`

**See [SINGLE_SOURCE_APPROACH.md](docs/SINGLE_SOURCE_APPROACH.md) for complete guide**

## 🚀 RunPod Deployment

**NEW**: This system is fully optimized for RunPod!

- **Quick Start**: See [RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md) for 5-minute setup
- **Full Guide**: See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for complete documentation
- **Interactive Launcher**: Use `./scripts/runpod_launcher.sh` for menu-driven interface
- **Jupyter Notebooks**: Ready-to-use notebooks in `notebooks/` directory
- **Optimized Configs**: Configs for 24GB, 48GB, and 80GB GPUs

```bash
# One-command setup on RunPod
cd /workspace && bash <(wget -qO- https://raw.githubusercontent.com/your-repo/LTX_video_training/main/runpod_setup.sh)
```

## Features

✅ **High-Quality Training Pipeline**
- Multi-resolution training with aspect ratio buckets
- IC-LoRA for efficient fine-tuning
- Progressive training strategy support
- Comprehensive data augmentation

✅ **Trajectory Extraction & Visualization**
- Optical flow-based trajectory extraction
- Multiple visualization modes (flow, depth, multi-channel)
- Physics-based smoothing
- Occlusion detection

✅ **Flexible Inference**
- Image + reference video → guided generation
- Image + motion description → text-guided motion
- Motion transfer between videos

✅ **Quality Monitoring**
- Trajectory fidelity metrics
- Temporal consistency evaluation
- Motion smoothness analysis
- Physical plausibility scoring

## Project Structure

```
LTX_video_training/
├── configs/                          # Training configurations
│   └── trajectory_iclora_high_quality.yaml
├── src/
│   ├── preprocessing/               # Trajectory extraction & visualization
│   │   ├── trajectory_extractor.py
│   │   └── trajectory_visualizer.py
│   ├── training/                    # Training scripts
│   │   └── train_trajectory_iclora.py
│   ├── inference/                   # Inference pipeline
│   │   └── trajectory_guided_inference.py
│   └── utils/                       # Utilities
│       └── quality_metrics.py
├── scripts/                         # Helper scripts
│   └── prepare_dataset.py
├── dataset/                         # Dataset directory
│   ├── videos/                      # Processed videos
│   ├── trajectories/                # Trajectory visualizations
│   ├── captions/                    # Text captions
│   └── train.csv                    # Training CSV
├── models/                          # Model cache
├── output/                          # Training outputs
├── checkpoints/                     # Model checkpoints
└── docs/                            # Documentation
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd LTX_video_training

# Create conda environment
conda create -n ltxv_trajectory python=3.10
conda activate ltxv_trajectory

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate peft
pip install opencv-python-headless numpy scipy pandas tqdm
pip install wandb tensorboard  # For logging
```

### 2. Prepare Dataset (Single-Source Approach)

Place your high-quality videos in a directory, then run:

```bash
# Enhanced single-source preparation (recommended)
python scripts/prepare_dataset_single_source.py \
    --input_dir /path/to/raw/videos \
    --output_dir ./dataset \
    --visualization_type multi \
    --overlay_first_frame \
    --min_frames 60 \
    --max_frames 121 \
    --target_fps 30 \
    --resolution 704x1216 \
    --num_workers 4
```

**What This Does:**

From each high-quality video:
1. ✅ Extracts trajectory field (optical flow or Trace Anything)
2. ✅ Creates trajectory visualization (conditioning input)
3. ✅ Saves **original video** as training target
4. ✅ Generates motion-aware caption
5. ✅ Creates perfect training pair: `(first_frame + trajectory) → video`

**Recommended Source Videos:**
- **Size**: 500-2,000 high-quality videos (ONE dataset for both trajectory & target)
- **Quality**: 1080p or 4K source material, minimal compression
- **Content**: Diverse motion types (camera motion, object motion, deformations)
- **Duration**: 3-10 seconds per clip
- **FPS**: 30+ for smooth motion
- **Sources**: Pexels, Pixabay, or your own footage

### 3. Configure Training

Edit `configs/trajectory_iclora_high_quality.yaml` to set:
- Model paths
- Dataset paths
- Training hyperparameters
- Logging configuration

Key settings:
```yaml
model:
  model_name: "Lightricks/LTX-Video"
  model_version: "LTXV_13B_097_DEV"

lora:
  rank: 256  # Higher for complex motion
  alpha: 256

dataset:
  train_csv: "./dataset/train.csv"
  num_frames: 121
  resolution: [704, 1216]

optimization:
  learning_rate: 8e-5
  max_train_steps: 10000
  gradient_accumulation_steps: 8
```

### 4. Train

```bash
# Single GPU
accelerate launch src/training/train_trajectory_iclora.py \
    --config configs/trajectory_iclora_high_quality.yaml

# Multi-GPU
accelerate launch --multi_gpu --num_processes 2 \
    src/training/train_trajectory_iclora.py \
    --config configs/trajectory_iclora_high_quality.yaml
```

**Hardware Requirements:**
- **Minimum**: 1x GPU with 48GB VRAM (A6000, A100)
- **Recommended**: 2x A100 80GB for faster training
- **Budget**: 1x 24GB GPU with INT8 quantization (slower)

**Training Time:**
- 10,000 steps with batch size 1 + grad accumulation 8:
  - On A100: ~24-36 hours
  - On A6000: ~36-48 hours

### 5. Inference

#### Generate from Image + Reference Video

```bash
python src/inference/trajectory_guided_inference.py \
    --model_path ./models/ltx-video-13b \
    --lora_path ./output/trajectory_iclora/final \
    --mode reference \
    --input_image ./examples/input.jpg \
    --reference_video ./examples/reference_motion.mp4 \
    --prompt "A cinematic shot of a person walking through a park" \
    --output ./outputs/generated.mp4 \
    --num_frames 121 \
    --fps 30 \
    --guidance_scale 7.5
```

#### Generate from Image + Motion Description

```bash
python src/inference/trajectory_guided_inference.py \
    --model_path ./models/ltx-video-13b \
    --lora_path ./output/trajectory_iclora/final \
    --mode description \
    --input_image ./examples/input.jpg \
    --motion_description "pan left slowly" \
    --prompt "A beautiful landscape" \
    --output ./outputs/generated_pan.mp4
```

#### Transfer Motion

```bash
python src/inference/trajectory_guided_inference.py \
    --model_path ./models/ltx-video-13b \
    --lora_path ./output/trajectory_iclora/final \
    --mode transfer \
    --input_image ./examples/target_subject.jpg \
    --reference_video ./examples/source_motion.mp4 \
    --prompt "Apply walking motion to target subject" \
    --output ./outputs/motion_transfer.mp4
```

## Training Strategies

### High-Quality Dataset Recommendations

1. **Video Sources**
   - Pexels, Pixabay: Free high-quality stock footage
   - Kinetics-700: Action recognition dataset
   - Custom footage: Film your own diverse motion patterns

2. **Dataset Composition**
   ```
   Human Motion (30%):     Walking, dancing, gestures
   Camera Motion (25%):    Pans, tilts, dolly shots
   Object Motion (20%):    Vehicles, flying objects
   Scene Dynamics (15%):   Nature, crowds, water
   Complex Scenarios (10%): Occlusions, multi-object
   ```

3. **Quality Criteria**
   - Resolution: 1080p minimum
   - FPS: 30+
   - Bitrate: High (minimal compression)
   - Lighting: Good, consistent
   - Motion blur: Natural (not excessive)

### Training Tips

1. **Start Conservative**
   ```yaml
   learning_rate: 8e-5  # Lower is safer
   max_train_steps: 10000
   gradient_accumulation_steps: 8
   ```

2. **Monitor Metrics**
   - Loss should decrease steadily
   - Trajectory fidelity > 0.7 is good
   - Temporal consistency > 0.8 is excellent

3. **Progressive Training** (Optional)
   - Stage 1 (2000 steps): 512×512, 61 frames, coarse motion
   - Stage 2 (3000 steps): 704×1216, 91 frames, medium fidelity
   - Stage 3 (5000 steps): 704×1216, 121 frames, high quality

4. **Validation**
   - Validate every 500 steps
   - Save best model based on LPIPS
   - Review sample videos regularly

## Architecture

### How It Works

1. **Training Phase**
   ```
   Original Video → Trajectory Extraction → Trajectory Visualization
                                                      ↓
   First Frame + Trajectory Viz → IC-LoRA → Generated Video
                                     ↑
                                Text Prompt
   ```

2. **Inference Phase**
   ```
   Input Image + Trajectory Field → Trained IC-LoRA → Generated Video
                                           ↑
                                     Text Prompt
   ```

### IC-LoRA Conditioning

IC-LoRA (In-Context LoRA) enables the model to condition on reference videos:
- **Reference Video**: Trajectory visualization (motion guidance)
- **First Frame**: Input image (content)
- **Text Prompt**: Semantic guidance

The model learns to combine all three to generate videos that:
- Match the input image appearance
- Follow the trajectory motion
- Align with the text description

## Evaluation Metrics

The system tracks multiple quality metrics:

1. **Trajectory Fidelity** (0-1, higher better)
   - Measures how well generated motion matches target trajectory
   - Compares optical flow from generated video to target

2. **Temporal Consistency** (0-1, higher better)
   - Measures smoothness between frames
   - Penalizes sudden jumps or artifacts

3. **Motion Smoothness** (0-1, higher better)
   - Measures motion jerk (derivative of acceleration)
   - Ensures physically plausible motion

4. **Physical Plausibility** (0-1, higher better)
   - Checks brightness consistency
   - Validates motion magnitudes
   - Ensures color coherence

## Troubleshooting

### Out of Memory

```yaml
# In config YAML:
model:
  transformer:
    load_in_8bit: true
  vae:
    load_in_8bit: true

optimization:
  use_8bit_adam: true
  gradient_accumulation_steps: 16  # Increase this
  train_batch_size: 1  # Keep at 1
```

### Poor Trajectory Following

- Increase LoRA rank: `rank: 512`
- Increase conditioning probability: `conditioning_prob: 0.95`
- Train longer: `max_train_steps: 20000`
- Use higher quality trajectory visualizations

### Temporal Inconsistency

- Enable temporal consistency loss in config
- Increase num_frames in training (more temporal context)
- Reduce learning rate
- Use EMA (exponential moving average)

### Overfitting

- Increase dataset size (500+ videos minimum)
- Enable data augmentation
- Add LoRA dropout: `dropout: 0.1`
- Use early stopping based on validation metrics

## Advanced Features

### Custom Trajectory Generation

Implement your own trajectory generation:

```python
from src.preprocessing.trajectory_extractor import TrajectoryExtractor

class CustomTrajectoryGenerator:
    def generate_trajectory(self, description, num_frames, resolution):
        # Create custom trajectory based on description
        # Return: (H, W, T, 3) array
        pass
```

### Multi-Scale Training

Enable in config:

```yaml
dataset:
  enable_resolution_buckets: true
  resolution_buckets:
    - [512, 896]
    - [704, 1216]
    - [896, 512]
```

### Physics-Based Loss

Experimental feature:

```yaml
experimental:
  use_physics_loss: true
  physics_loss_weight: 0.1
```

## Integration with Trace Anything

When Trace Anything becomes available:

1. Update `trajectory_extractor.py`:
```python
def _load_trace_anything_model(self, model_name):
    from trace_anything import TraceAnythingModel
    return TraceAnythingModel.from_pretrained(model_name)
```

2. Use real trajectory fields instead of optical flow:
```python
trajectory_data = extractor.extract_from_video(video_path)
# trajectory_data now contains true 3D trajectory fields
```

Benefits:
- More accurate 3D motion representation
- Better occlusion handling
- Goal-conditioned motion planning

## Citation

If you use this system, please cite:

```bibtex
@software{ltxv_trajectory_training,
  title={Trajectory-Guided LTX Video Training System},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## License

This project is for research and educational purposes. Please respect the licenses of:
- LTX Video (Lightricks)
- Trace Anything (when available)
- Other dependencies

## Acknowledgments

- **Lightricks** for LTX Video
- **Trace Anything** authors for trajectory field inspiration
- **Hugging Face** for Diffusers and PEFT libraries

## Support

For issues and questions:
- GitHub Issues: [Report here]
- Discord: [Join community]
- Email: [your-email]

---

**Status**: ✅ Complete system ready for training

**Last Updated**: 2025-10-22
