# Quick Start: Trajectory-Guided Training

Everything happens in `/workspace/LTX_video_training` - no external directories needed!

---

## 🚀 Two-Command Setup

### 1. Setup Official Trainer (One Time)

```bash
cd /workspace/LTX_video_training
./scripts/setup_official_trainer.sh
```

This will:
- Clone official LTX-Video trainer into `LTX-Video-Trainer/`
- Install all dependencies
- Link your preprocessed dataset
- Create H100-optimized training config
- Keep everything in `/workspace/LTX_video_training`

### 2. Start Training

```bash
cd /workspace/LTX_video_training
./scripts/train_with_official.sh
```

That's it! Training will start immediately.

---

## 📂 Directory Structure

Everything stays in one place:

```
/workspace/LTX_video_training/
├── raw_videos/                    # Your source videos + captions
│   ├── video1.mp4
│   ├── video1.txt
│   └── ...
│
├── dataset/                       # Preprocessed by our scripts
│   ├── videos/                   # Processed videos
│   ├── trajectories/             # Trajectory visualizations
│   ├── captions/                 # Caption files
│   └── train.csv
│
├── LTX-Video-Trainer/            # Official trainer (auto-cloned)
│   ├── data/trajectory_control/  # → Symlinks to ../dataset/
│   ├── configs/
│   │   └── trajectory_control_h100.yaml
│   └── src/training/
│
├── output/                        # Training outputs
│   └── trajectory_control_official/
│       ├── checkpoint-250/
│       ├── checkpoint-500/
│       └── ...
│
├── logs/                          # Training logs
│   └── train_official_*.log
│
└── scripts/                       # Helper scripts
    ├── setup_official_trainer.sh
    └── train_with_official.sh
```

---

## 📋 Complete Workflow

### If Starting Fresh:

```bash
cd /workspace/LTX_video_training

# 1. Put your videos + captions in raw_videos/
# (You've already done this!)

# 2. Prepare dataset (already done!)
python scripts/prepare_dataset_single_source.py \
    --input_dir raw_videos \
    --output_dir dataset \
    --use_existing_captions

# 3. Setup official trainer
./scripts/setup_official_trainer.sh

# 4. Train!
./scripts/train_with_official.sh
```

### If Dataset Already Prepared:

```bash
cd /workspace/LTX_video_training

# Just these two commands:
./scripts/setup_official_trainer.sh
./scripts/train_with_official.sh
```

---

## 📊 Monitor Training

### Watch Logs:
```bash
tail -f /workspace/LTX_video_training/logs/train_official_*.log
```

### Watch GPU:
```bash
watch -n 1 nvidia-smi
```

### TensorBoard:
```bash
tensorboard --logdir /workspace/LTX_video_training/output/trajectory_control_official/tensorboard
```

---

## ⚙️ Configuration

The config is at: `LTX-Video-Trainer/configs/trajectory_control_h100.yaml`

### Key Settings for Your 21 Videos:

```yaml
training:
  num_train_epochs: 100        # Train for 100 epochs
  train_batch_size: 4          # H100 can handle this
  gradient_accumulation_steps: 2

ic_lora:
  rank: 512                    # High rank for quality
  alpha: 512

checkpointing:
  save_steps: 250             # Save every 250 steps
  keep_last_n_checkpoints: 10 # Keep last 10
```

### To Modify:

```bash
nano /workspace/LTX_video_training/LTX-Video-Trainer/configs/trajectory_control_h100.yaml
```

---

## 🔄 Resume Training

If training stops, just run the train script again:

```bash
cd /workspace/LTX_video_training
./scripts/train_with_official.sh
```

The official trainer will automatically resume from the last checkpoint!

---

## 📦 Checkpoints

Checkpoints are saved to:
```
/workspace/LTX_video_training/output/trajectory_control_official/
├── checkpoint-250/
├── checkpoint-500/
├── checkpoint-750/
└── ...
```

Each checkpoint contains:
- LoRA weights
- Training state
- Optimizer state
- Configuration

---

## 🎨 After Training

### Use Your Trained Model:

```python
from ltxvideo import LTXVideoPipeline

# Load base model
pipe = LTXVideoPipeline.from_pretrained("Lightricks/LTX-Video")

# Load your trajectory control LoRA
pipe.load_lora_weights(
    "/workspace/LTX_video_training/output/trajectory_control_official/checkpoint-5000"
)

# Generate with trajectory control
video = pipe(
    prompt="A person walking in a park",
    control_image=trajectory_visualization,
    num_frames=121,
    guidance_scale=7.5
).frames
```

---

## 🛠️ Troubleshooting

### "Official trainer not found"
```bash
./scripts/setup_official_trainer.sh
```

### "No videos found in dataset"
```bash
python scripts/prepare_dataset_single_source.py \
    --input_dir raw_videos \
    --output_dir dataset \
    --use_existing_captions
```

### Check dataset:
```bash
ls -la /workspace/LTX_video_training/dataset/videos/
ls -la /workspace/LTX_video_training/dataset/trajectories/
ls -la /workspace/LTX_video_training/dataset/captions/
```

### Out of Memory:
Edit config, reduce batch size:
```yaml
training:
  train_batch_size: 2          # Reduce from 4
  gradient_accumulation_steps: 4  # Increase to compensate
```

---

## ⚡ Expected Training Time

**With H100 and 21 videos:**

- **5,000 steps**: ~6-8 hours (recommended for small dataset)
- **10,000 steps**: ~12-16 hours
- **Steps per epoch**: ~21 videos / 4 batch = ~5 steps/epoch
- **100 epochs**: ~500 steps total

**Recommendation**: Train for 2000-5000 steps, monitor validation loss.

---

## 📚 What's Happening?

1. **Your preprocessing** (unique value):
   - Extracts trajectories from videos
   - Creates multi-channel visualizations
   - Preserves your custom captions

2. **Official trainer** (proven implementation):
   - Loads LTX-Video model properly
   - Trains IC-LoRA control adapter
   - Handles checkpointing and validation

3. **Result**:
   - Trajectory-guided video generation
   - Professional quality
   - Production-ready

---

## 🎯 Summary

**All in one directory** → `/workspace/LTX_video_training`
**Two scripts** → `setup_official_trainer.sh` + `train_with_official.sh`
**One config** → `LTX-Video-Trainer/configs/trajectory_control_h100.yaml`
**Simple** → Everything automated!

**Ready to train!** 🚀
