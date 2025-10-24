# Single-Source Training Approach

## Overview

The **Single-Source Approach** is the recommended method for preparing training data for trajectory-guided video generation. Instead of using separate datasets for trajectories and target videos, **you use the same high-quality videos for both**.

## Why Single-Source?

### ✅ Advantages

1. **Perfect Alignment**: Trajectory and content are perfectly synchronized
2. **Realistic Physics**: Model learns real-world motion → appearance mapping
3. **Efficiency**: No need for separate datasets
4. **Consistency**: Same quality standards throughout
5. **Scalability**: Easy to expand - just add more source videos
6. **Lower Data Requirements**: One dataset instead of two

### ❌ Alternative (NOT Recommended)

Some might think you need:
- **Dataset A**: Videos with motion patterns (for trajectory extraction)
- **Dataset B**: High-quality realistic videos (for training targets)

**This is NOT necessary and actually worse** because:
- Misalignment between trajectory and visual content
- More data to collect and manage
- Potential physics inconsistencies

## How It Works

### The Pipeline

```
Single High-Quality Video (e.g., person_walking.mp4)
            ↓
    [Processing Pipeline]
            ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
Trajectory Field              Original Video
(extracted via                (kept as-is)
optical flow or
Trace Anything)
    ↓                               ↓
[Visualization]               [Training Target]
    ↓                               ↓
trajectory_viz.mp4     →    person_walking.mp4
(RGB color-coded)           (realistic HD footage)
```

### Training Process

**Input to Model:**
- First frame from original video
- Trajectory visualization (color-coded motion)
- Text caption (auto-generated)

**Expected Output:**
- Full original video

**What Model Learns:**
```
(first_frame + trajectory_visualization + caption) → realistic_video
```

The model learns to:
1. **Interpret trajectory patterns** from the visualization
2. **Generate realistic motion** that matches the pattern
3. **Preserve visual quality** from the high-quality training targets

## Implementation

### Step 1: Collect High-Quality Source Videos

**Requirements:**
- Resolution: 1080p or higher
- FPS: 30+
- Duration: 3-10 seconds per clip
- Quality: Minimal compression, good lighting
- Content: Diverse motion patterns

**Sources:**
- Pexels: https://www.pexels.com/videos/
- Pixabay: https://pixabay.com/videos/
- Your own footage (best option!)

**Recommended Quantity:**
- Minimum: 100 videos (testing)
- Good: 500 videos
- Excellent: 1000-2000 videos

### Step 2: Run Single-Source Preparation

```bash
cd /workspace/LTX_video_training

# Use the enhanced single-source script
python scripts/prepare_dataset_single_source.py \
    --input_dir /workspace/LTX_video_training/raw_videos \
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

For each input video:
1. ✅ Loads high-quality frames
2. ✅ Extracts trajectory field (optical flow or Trace Anything)
3. ✅ Creates trajectory visualization
4. ✅ Saves ORIGINAL video as training target
5. ✅ Generates motion-aware caption
6. ✅ Creates training pair

### Step 3: Verify Output

```
dataset/
├── videos/                     # Original videos (training targets)
│   ├── video_000000_clip_000.mp4
│   ├── video_000001_clip_000.mp4
│   └── ...
├── trajectories/               # Trajectory visualizations (conditioning)
│   ├── video_000000_clip_000_traj.mp4
│   ├── video_000001_clip_000_traj.mp4
│   └── ...
├── captions/                   # Auto-generated captions
│   ├── video_000000_clip_000.txt
│   └── ...
└── train.csv                   # Training pairs
```

### Step 4: Train

```bash
./scripts/train.sh configs/trajectory_iclora_high_quality.yaml 1
```

The model learns from the training pairs in `train.csv`:
```csv
video_path,trajectory_path,caption_path
/path/videos/clip_001.mp4,/path/trajectories/clip_001_traj.mp4,/path/captions/clip_001.txt
```

## Trajectory Visualization Strategies

### Option 1: Multi-Channel (Recommended)

Encodes trajectory in RGB channels:
- **R channel**: Horizontal motion (X)
- **G channel**: Vertical motion (Y)
- **B channel**: Magnitude or depth (Z)

```python
visualization_type='multi'
```

### Option 2: Flow Visualization

HSV color-coded optical flow:
- **Hue**: Motion direction
- **Saturation**: Full
- **Value**: Motion magnitude

```python
visualization_type='flow'
```

### Option 3: Overlay on First Frame

Overlays trajectory on first frame (helps model preserve structure):
```python
overlay_first_frame=True
overlay_alpha=0.3  # 30% first frame, 70% trajectory
```

**Advantages:**
- Provides spatial context
- Helps model preserve appearance
- Better for complex scenes

## Motion Analysis & Captioning

The system automatically analyzes motion and generates captions:

### Motion Statistics Computed

```python
{
    'camera_motion_score': 0.75,      # 0-1, higher = more camera motion
    'object_motion_score': 0.25,       # 0-1, higher = more object motion
    'camera_type': 'pan left',         # Camera motion classification
    'motion_description': 'slow motion',  # Speed classification
    'motion_magnitude': 3.5,           # Average motion magnitude
    'num_occlusions': 25,              # Count of occlusion events
    'flow_x': -2.3,                    # Mean horizontal flow
    'flow_y': 0.5                      # Mean vertical flow
}
```

### Auto-Generated Captions

Examples:
- `"Camera pan left, slow motion, realistic footage"`
- `"Medium speed motion, camera static, realistic footage"`
- `"Fast motion with camera zoom in, realistic footage"`
- `"Camera pan right and tilt up, moderate motion, with occlusions, realistic footage"`

### Custom Captions

For better results, manually enhance captions:

```txt
# Instead of:
Camera pan left, slow motion, realistic footage

# Use:
Cinematic slow pan left across a misty mountain landscape at sunrise, smooth camera movement, natural lighting, film grain aesthetic

# Or:
A woman in red dress walking towards camera in slow motion, elegant movement, soft bokeh background, golden hour lighting
```

## Quality Optimization

### For Best Training Results

**Source Videos Should Have:**
- ✅ Clear, well-defined motion
- ✅ Good lighting and contrast
- ✅ Minimal motion blur (some is OK)
- ✅ Stable footage (unless intentional camera shake)
- ✅ High bitrate / low compression

**Trajectory Visualization Should:**
- ✅ Show clear motion patterns
- ✅ Be visually distinct from background
- ✅ Maintain temporal consistency
- ✅ Highlight important motion regions

### Data Distribution

Aim for diversity in your source videos:

```
Motion Types:
├── Camera Motion (30%)
│   ├── Pans (left, right)
│   ├── Tilts (up, down)
│   ├── Dollys (forward, backward)
│   ├── Zooms (in, out)
│   └── Combined movements
│
├── Object Motion (30%)
│   ├── People walking/running
│   ├── Vehicles moving
│   ├── Objects falling/flying
│   └── Deformations (cloth, water)
│
├── Scene Dynamics (20%)
│   ├── Nature (wind, water, leaves)
│   ├── Crowds
│   └── Animals
│
└── Complex Scenarios (20%)
    ├── Occlusions
    ├── Multi-object interactions
    ├── Fast action
    └── Lighting changes
```

## Comparison with Two-Dataset Approach

| Aspect | Single-Source | Two-Dataset |
|--------|--------------|-------------|
| **Alignment** | Perfect | May have misalignment |
| **Data Collection** | 500-2000 videos | 1000-4000 videos |
| **Processing Time** | 1× | 2× |
| **Consistency** | Guaranteed | Requires careful curation |
| **Physics** | Real-world | May be inconsistent |
| **Complexity** | Simple | More complex |
| **Recommended** | ✅ Yes | ❌ No |

## Advanced: Two-Stage Training (Optional)

For even better results, you can do progressive training:

### Stage 1: Diverse Motion (Any Quality)
```bash
# Train on diverse motion patterns (can use lower quality)
# Focus: Learn to follow trajectories
python scripts/train.sh configs/stage1_motion_learning.yaml
```

### Stage 2: High-Quality Refinement
```bash
# Fine-tune on high-quality videos
# Focus: Learn visual fidelity
python scripts/train.sh configs/stage2_quality_refinement.yaml \
    --resume_from_checkpoint ./output/stage1/final
```

**Benefits:**
- Stage 1: Broad motion understanding
- Stage 2: High visual quality

**Trade-off:**
- More complex workflow
- Longer total training time
- Usually not necessary

## Example Workflow

### Complete End-to-End Example

```bash
# 1. Collect source videos
mkdir -p /workspace/LTX_video_training/raw_videos
# Upload your HD videos here

# 2. Prepare dataset (single-source)
python scripts/prepare_dataset_single_source.py \
    --input_dir /workspace/LTX_video_training/raw_videos \
    --output_dir ./dataset \
    --visualization_type multi \
    --overlay_first_frame \
    --resolution 704x1216

# 3. Verify dataset
python scripts/verify_dataset.py --dataset_dir ./dataset

# 4. Train
./scripts/train.sh configs/trajectory_iclora_high_quality.yaml 1

# 5. Monitor
watch -n 1 nvidia-smi
tail -f logs/*.log

# 6. After training, test inference
python src/inference/trajectory_guided_inference.py \
    --model_path ./models/ltx-video-13b \
    --lora_path ./output/trajectory_iclora/final \
    --mode reference \
    --input_image examples/test_image.jpg \
    --reference_video dataset/trajectories/video_000000_clip_000_traj.mp4 \
    --prompt "A cinematic shot" \
    --output result.mp4
```

## Troubleshooting

### Issue: Poor Motion Following

**Problem**: Generated videos don't follow trajectories well

**Solutions:**
1. Increase trajectory visibility in visualization:
   ```python
   overlay_alpha=0.1  # Less first frame, more trajectory
   ```

2. Use clearer visualization type:
   ```python
   visualization_type='flow'  # Instead of 'multi'
   ```

3. Ensure source videos have clear motion:
   - Avoid static or nearly-static videos
   - Check trajectory visualizations look clear

### Issue: Low Visual Quality

**Problem**: Generated videos look unrealistic

**Solutions:**
1. Use higher quality source videos (1080p+)
2. Increase training steps
3. Use larger LoRA rank
4. Check if source videos are properly saved (not re-compressed)

### Issue: Temporal Inconsistency

**Problem**: Flickering or inconsistent frames

**Solutions:**
1. Enable temporal consistency loss in config
2. Use first-frame overlay:
   ```python
   overlay_first_frame=True
   ```
3. Ensure source videos are smooth (not jittery)

## Best Practices

### ✅ Do

- Use high-quality source videos (1080p+, 30fps+)
- Enable first-frame overlay for structural guidance
- Generate diverse motion types
- Use multi-channel visualization for rich encoding
- Manually enhance auto-generated captions
- Validate dataset before training

### ❌ Don't

- Mix low and high quality videos
- Use heavily compressed source videos
- Skip motion analysis
- Ignore failed preprocessing (check logs)
- Train without validating dataset first

## Summary

The **Single-Source Approach** is:
- ✅ Simpler to implement
- ✅ More efficient
- ✅ Better aligned
- ✅ Physically consistent
- ✅ Easier to scale

**Bottom Line**: Use the same high-quality videos for both trajectory extraction and training targets. This is the recommended approach for trajectory-guided video generation with LTX Video.

---

For questions or issues, see:
- `README.md`: Main documentation
- `docs/BEST_PRACTICES.md`: Training best practices
- `RUNPOD_DEPLOYMENT.md`: RunPod-specific guide
