# Video Resolution and Length Requirements for Training

## Short Answer

**NO, videos don't need to be the same size!** The system supports:

✅ **Different resolutions** using **aspect ratio bucketing**
✅ **Different frame counts** with proper configuration
✅ **Mixed aspect ratios** (16:9, 9:16, 1:1, 4:3, etc.)

## Detailed Explanation

### Spatial Dimensions (Width × Height)

#### ❌ Old Approach: Fixed Resolution
```
All videos → Resize to 704×1216 → Training
Problems:
- Distorts aspect ratios
- Loses quality on downsizing
- Wastes quality on upsizing
- Forces square content into widescreen
```

#### ✅ Modern Approach: Resolution Bucketing
```
Videos of different sizes → Group by aspect ratio → Training
Benefits:
- Preserves aspect ratios
- Better quality utilization
- Supports multiple output formats
- More flexible dataset
```

### How Resolution Bucketing Works

**Concept**: Group videos with similar aspect ratios into "buckets" and train on each bucket separately.

```python
# Your config already has this!
enable_resolution_buckets: true
resolution_buckets:
  - [512, 896]    # ~0.57 ratio (9:16, vertical/portrait)
  - [704, 1216]   # ~0.58 ratio (16:9, cinematic)
  - [896, 512]    # ~1.75 ratio (16:9, horizontal)
  - [768, 768]    # 1.0 ratio (1:1, square)
  - [640, 640]    # 1.0 ratio (1:1, smaller square)
```

**During Training:**
1. Videos are assigned to nearest bucket based on aspect ratio
2. Each batch contains videos from same bucket
3. All videos in batch are resized to bucket's resolution
4. Model learns to handle multiple resolutions

### Temporal Dimension (Number of Frames)

#### Two Approaches:

**1. Fixed Frame Count (Simpler, Recommended)**
```yaml
num_frames: 121  # All videos processed to 121 frames
```

**How it works:**
- Videos **longer than 121 frames**: Split into multiple clips
- Videos **shorter than 121 frames**: Pad with last frame or skip
- Each training sample has exactly 121 frames

**Pros:**
- Simpler training loop
- Consistent memory usage
- Easier to debug

**Cons:**
- May waste some video content
- Padding can introduce artifacts

**2. Temporal Bucketing (Advanced)**
```yaml
enable_temporal_buckets: true
temporal_buckets:
  - 61   # ~2 seconds at 30fps
  - 91   # ~3 seconds at 30fps
  - 121  # ~4 seconds at 30fps
```

**How it works:**
- Videos assigned to nearest temporal bucket
- Batches contain videos with same frame count
- Model learns multiple durations

**Pros:**
- More efficient use of video content
- Can generate different lengths at inference
- Less padding needed

**Cons:**
- More complex training
- Higher memory variance
- Requires careful batch construction

## Best Practices

### For Your Dataset Preparation

#### ✅ Recommended: Use Resolution Buckets + Fixed Frames

**Configuration:**
```yaml
# In your training config
dataset:
  num_frames: 121  # Fixed
  enable_resolution_buckets: true
  resolution_buckets:
    - [512, 896]    # Vertical (TikTok, Instagram Stories)
    - [704, 1216]   # Cinematic 16:9 (YouTube, Films)
    - [896, 512]    # Horizontal 16:9
    - [768, 768]    # Square (Instagram)
```

**Your source videos can be:**
- 1920×1080 (Full HD 16:9) → Assigned to [704, 1216]
- 1080×1920 (Vertical HD) → Assigned to [512, 896]
- 3840×2160 (4K 16:9) → Assigned to [704, 1216]
- 2160×3840 (4K Vertical) → Assigned to [512, 896]
- 1080×1080 (Square) → Assigned to [768, 768]

**Your source videos duration:**
- 2 seconds (60 frames) → Padded to 121 frames OR skipped
- 4 seconds (120 frames) → Used as single clip (121 frames)
- 10 seconds (300 frames) → Split into 2-3 clips

### Resolution Bucket Assignment Logic

```python
def assign_to_bucket(video_width, video_height, buckets):
    """
    Assign video to nearest resolution bucket
    """
    video_ratio = video_width / video_height

    best_bucket = None
    min_diff = float('inf')

    for bucket_h, bucket_w in buckets:
        bucket_ratio = bucket_w / bucket_h
        diff = abs(video_ratio - bucket_ratio)

        if diff < min_diff:
            min_diff = diff
            best_bucket = (bucket_h, bucket_w)

    return best_bucket

# Examples:
assign_to_bucket(1920, 1080, buckets)  # → [704, 1216] (16:9)
assign_to_bucket(1080, 1920, buckets)  # → [512, 896] (9:16)
assign_to_bucket(1080, 1080, buckets)  # → [768, 768] (1:1)
assign_to_bucket(1280, 720, buckets)   # → [704, 1216] (16:9)
```

### What Happens During Preprocessing

**Spatial Processing:**
```python
# 1. Load video
video = load_video("1920x1080_clip.mp4")  # Original resolution

# 2. Assign to bucket
bucket = assign_to_bucket(1920, 1080, buckets)  # → [704, 1216]

# 3. Resize to bucket
video_resized = resize(video, target_size=(1216, 704))  # Note: (W, H)

# 4. Save
save_video(video_resized, "dataset/videos/clip_001.mp4")
```

**Temporal Processing:**
```python
# 1. Load video
video = load_video("clip.mp4")  # 300 frames

# 2. Check length
if len(video) > max_frames:  # 300 > 121
    # Split into clips
    clips = split_video(video, max_frames=121, overlap=10)
    # → clip_001 (frames 0-121), clip_002 (frames 111-232), etc.

elif len(video) < min_frames:  # < 60
    # Skip or pad
    if len(video) < 60:
        skip()  # Too short
    else:
        video = pad_to_length(video, target=121)

else:
    # Use as-is
    video = video[:121]  # Trim to exact length
```

## Common Scenarios

### Scenario 1: Mixed Aspect Ratios

**Your videos:**
```
video_001.mp4: 1920×1080 (16:9)
video_002.mp4: 1080×1920 (9:16)
video_003.mp4: 1080×1080 (1:1)
video_004.mp4: 3840×2160 (16:9)
video_005.mp4: 1280×720  (16:9)
```

**After processing:**
```
videos/
  video_001_clip_000.mp4: 1216×704  (bucket [704, 1216])
  video_002_clip_000.mp4: 512×896   (bucket [512, 896])
  video_003_clip_000.mp4: 768×768   (bucket [768, 768])
  video_004_clip_000.mp4: 1216×704  (bucket [704, 1216])
  video_005_clip_000.mp4: 1216×704  (bucket [704, 1216])
```

**Result:** ✅ All videos usable, aspect ratios preserved!

### Scenario 2: Different Durations

**Your videos:**
```
video_001.mp4: 3 seconds (90 frames)
video_002.mp4: 5 seconds (150 frames)
video_003.mp4: 10 seconds (300 frames)
video_004.mp4: 2 seconds (60 frames)
```

**After processing (with num_frames=121):**
```
videos/
  video_001_clip_000.mp4: 90 → 121 frames (padded)
  video_002_clip_000.mp4: 121 frames (trimmed)
  video_003_clip_000.mp4: Split into 3 clips (121 each)
  video_004_clip_000.mp4: 60 frames (skipped, too short)
```

**Result:** ✅ Multiple clips from long videos, consistent length!

### Scenario 3: Extreme Aspect Ratios

**Your video:**
```
ultra_wide.mp4: 2560×1080 (21:9, ultra-wide)
```

**Processing:**
```
1. Calculate ratio: 2560/1080 = 2.37
2. Find nearest bucket:
   - [704, 1216] ratio = 1.73
   - [896, 512] ratio = 1.75  ← Closest!
3. Assign to [896, 512]
4. Resize with aspect ratio preservation:
   - May crop sides slightly to fit 16:9
   - Or add letterboxing
```

**Options:**
- **Crop**: Lose some edges but preserve central content
- **Letterbox**: Keep all content but add black bars
- **Stretch**: Distort (NOT recommended)

## Configuration Examples

### Configuration 1: Standard Multi-Aspect (Recommended)

```yaml
dataset:
  num_frames: 121
  enable_resolution_buckets: true
  resolution_buckets:
    - [512, 896]    # Portrait
    - [704, 1216]   # Landscape (16:9)
    - [768, 768]    # Square

  # How to handle videos
  min_frames: 60    # Skip shorter
  max_frames: 121   # Split longer
```

**Use case:** General purpose, supports social media content

### Configuration 2: Cinematic Only

```yaml
dataset:
  num_frames: 121
  enable_resolution_buckets: true
  resolution_buckets:
    - [640, 1136]   # 16:9 small
    - [704, 1216]   # 16:9 medium
    - [768, 1360]   # 16:9 large
```

**Use case:** Film/cinematic content only

### Configuration 3: Fixed Single Resolution (Simple)

```yaml
dataset:
  num_frames: 121
  enable_resolution_buckets: false
  resolution: [704, 1216]  # All videos → this size
```

**Use case:** All source videos same aspect ratio, simplest approach

### Configuration 4: Multi-Duration + Multi-Resolution (Advanced)

```yaml
dataset:
  enable_resolution_buckets: true
  resolution_buckets:
    - [704, 1216]
    - [768, 768]

  enable_temporal_buckets: true
  temporal_buckets:
    - 61
    - 91
    - 121
```

**Use case:** Maximum flexibility, complex training

## Memory Considerations

### GPU Memory Usage by Configuration

**Formula:**
```
Memory = batch_size × num_frames × height × width × channels × bytes_per_value
```

**Examples (with batch_size=1, bf16):**

| Resolution | Frames | Memory | Notes |
|------------|--------|--------|-------|
| 512×896 | 61 | ~8GB | Small, fast |
| 704×1216 | 91 | ~18GB | Medium |
| 704×1216 | 121 | ~24GB | Standard |
| 768×1360 | 121 | ~30GB | Large |
| 1024×1024 | 121 | ~30GB | Square large |

**With resolution bucketing:**
- Memory usage varies per bucket
- Smaller buckets train faster
- Larger buckets need more VRAM

**Recommendation:**
- 24GB GPU: Use buckets up to 704×1216, 121 frames
- 48GB GPU: Use buckets up to 768×1360, 121 frames
- 80GB GPU: Use buckets up to 1024×1024, 121 frames

## Dataset Preparation Script

Your `prepare_dataset_single_source.py` already handles this!

```python
# The script automatically:
1. Detects video resolution
2. Assigns to appropriate bucket (if bucketing enabled)
3. Resizes to bucket resolution
4. Handles frame count (pad/split/trim)
5. Saves processed videos
```

**What you need to do:**
```bash
# Just run the script with your source videos
python scripts/prepare_dataset_single_source.py \
    --input_dir /workspace/LTX_video_training/raw_videos \  # Mix of resolutions OK!
    --output_dir ./dataset \
    --resolution 704x1216 \  # Default bucket (if bucketing disabled)
    --min_frames 60 \
    --max_frames 121
```

**The script handles:**
- Videos of any resolution → Resized appropriately
- Videos of any duration → Split/padded/trimmed
- Different aspect ratios → Assigned to buckets
- Different framerates → Resampled to target FPS

## Inference Behavior

**During inference** (after training):

```python
# You can generate at any resolution the model was trained on!

# Portrait video
generator.generate(
    image=input_image,
    trajectory=trajectory,
    output_size=(512, 896),  # Vertical
    num_frames=121
)

# Landscape video
generator.generate(
    image=input_image,
    trajectory=trajectory,
    output_size=(704, 1216),  # Horizontal
    num_frames=121
)

# Square video
generator.generate(
    image=input_image,
    trajectory=trajectory,
    output_size=(768, 768),  # Square
    num_frames=121
)
```

**Model learns to generate at all bucketed resolutions!**

## Recommendations

### ✅ Best Practices

1. **Use resolution bucketing** (enable_resolution_buckets: true)
   - Supports multiple aspect ratios
   - Better quality than forcing single size
   - More flexible at inference

2. **Fixed frame count** (num_frames: 121)
   - Simpler than temporal bucketing
   - Consistent memory usage
   - Easier to debug

3. **Source video requirements:**
   - **Resolution**: Any HD or higher (1080p+)
   - **Duration**: 2-10 seconds ideal
   - **Aspect ratio**: Any (16:9, 9:16, 1:1, 4:3 all OK)
   - **FPS**: 24-60 (will be resampled to 30)

4. **Bucket design:**
   - Cover common aspect ratios (16:9, 9:16, 1:1)
   - Keep resolutions reasonable for your GPU
   - 3-5 buckets is typical

### ❌ What to Avoid

1. **Forcing all videos to same size** without bucketing
   - Distorts aspect ratios
   - Lower quality

2. **Too many buckets** (>7-8)
   - Fragmentizes dataset
   - Each bucket gets less training data

3. **Extreme resolution differences** in same bucket
   - 480p and 4K shouldn't be same bucket
   - Pre-filter very low quality

4. **Videos shorter than min_frames**
   - Will be skipped or heavily padded
   - Filter out <2 seconds

## Summary

### Key Points

✅ **Spatial (W×H)**: Videos can be different sizes - use bucketing!
✅ **Temporal (frames)**: Use fixed count (121) for simplicity
✅ **Aspect ratios**: Mix freely - will be assigned to buckets
✅ **Source quality**: Higher is better (1080p+)
✅ **Duration range**: 2-10 seconds ideal

### Quick Decision Tree

```
Q: Do I have videos with different aspect ratios?
├─ YES → Use resolution bucketing (recommended)
└─ NO  → Use fixed resolution (simpler)

Q: Do I have videos with very different durations?
├─ YES → Set appropriate min/max frames, script will split
└─ NO  → Set num_frames to match your video length

Q: What GPU do I have?
├─ 24GB → buckets up to 704×1216, 121 frames
├─ 48GB → buckets up to 768×1360, 121 frames
└─ 80GB → buckets up to 1024×1024, 121 frames
```

### Your Current Config Already Supports This!

Check `configs/trajectory_iclora_high_quality.yaml`:

```yaml
dataset:
  enable_resolution_buckets: true
  resolution_buckets:
    - [512, 896]   # Portrait
    - [704, 1216]  # Landscape
    - [896, 512]   # Horizontal
    - [768, 768]   # Square
  num_frames: 121
```

**You're all set!** Just prepare your dataset with mixed resolutions and the system will handle it automatically.

---

**Bottom line**: You do **NOT** need videos of the same size. The resolution bucketing system handles different resolutions intelligently, preserving aspect ratios and quality!
