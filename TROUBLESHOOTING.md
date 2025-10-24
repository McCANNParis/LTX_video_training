# Troubleshooting Guide

Common issues and solutions for LTX Video Trajectory-Guided Training.

---

## Installation Issues

### ImportError: T5Tokenizer requires the SentencePiece library

**Error:**
```
ImportError: T5Tokenizer requires the SentencePiece library but it was not found in your environment.
```

**Cause:** Missing `sentencepiece` dependency required by LTX-Video's T5 tokenizer.

**Solution:**
```bash
pip install sentencepiece protobuf
```

**Prevention:** Use the requirements file:
```bash
pip install -r requirements.txt
# or for RunPod
pip install -r runpod_requirements.txt
```

---

### ModuleNotFoundError: No module named 'src'

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Cause:** Scripts not running from project root directory.

**Solution:**
```bash
# Always run scripts from project root
cd /workspace/LTX_video_training
python scripts/prepare_dataset_single_source.py ...
```

The scripts now use dynamic path detection, so this should be rare. If it persists:
```bash
export PYTHONPATH=/workspace/LTX_video_training:$PYTHONPATH
```

---

### CUDA Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions by GPU Size:**

**24GB GPU (RTX 4090, RTX 3090):**
```bash
# Use the 24GB optimized config
python src/training/train_trajectory_iclora.py \
    --config configs/trajectory_iclora_runpod_24gb.yaml
```

Edit config if still failing:
```yaml
optimization:
  train_batch_size: 1
  gradient_accumulation_steps: 16  # Increase this

model:
  transformer:
    load_in_8bit: true  # Enable 8-bit quantization
```

**48GB GPU (A6000, A40):**
```bash
# Use high quality config but adjust batch size
# Edit configs/trajectory_iclora_high_quality.yaml
optimization:
  train_batch_size: 2
  gradient_accumulation_steps: 4
```

**80GB GPU (H100, A100):**
```bash
# Use H100 optimized config
python src/training/train_trajectory_iclora.py \
    --config configs/trajectory_iclora_h100.yaml
```

---

## Dataset Issues

### No videos found

**Error:**
```
INFO:__main__:Found 0 source videos
ERROR:__main__:No video files found!
```

**Solutions:**
1. **Check path:** Verify videos are in correct directory
   ```bash
   ls /workspace/LTX_video_training/raw_videos/
   ```

2. **Check file extensions:** Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
   ```bash
   # Rename if needed
   for f in *.MP4; do mv "$f" "${f%.MP4}.mp4"; done
   ```

3. **Check permissions:**
   ```bash
   chmod 644 /workspace/LTX_video_training/raw_videos/*.mp4
   ```

---

### Video too short or invalid

**Warning:**
```
WARNING:__main__:Video too short or invalid: video.mp4
```

**Cause:** Video has fewer than minimum required frames (default: 60 frames).

**Solutions:**
1. **Adjust minimum frames:**
   ```bash
   python scripts/prepare_dataset_single_source.py \
       --input_dir /workspace/LTX_video_training/raw_videos \
       --output_dir /workspace/LTX_video_training/dataset \
       --min_frames 30  # Lower minimum
   ```

2. **Check video integrity:**
   ```bash
   ffprobe video.mp4
   ```

3. **Re-encode if corrupted:**
   ```bash
   ffmpeg -i corrupted.mp4 -c:v libx264 -crc copy fixed.mp4
   ```

---

### Caption files not found

**Issue:** Using existing captions but they're not being detected.

**Solutions:**
1. **Check naming:** Caption files must match video names
   ```
   video1.mp4 → video1.txt
   video2.mp4 → video2.txt
   ```

2. **Use correct flag:**
   ```bash
   # Captions alongside videos
   python scripts/prepare_dataset_single_source.py \
       --input_dir /workspace/LTX_video_training/raw_videos \
       --output_dir /workspace/LTX_video_training/dataset \
       --use_existing_captions

   # Captions in separate directory
   python scripts/prepare_dataset_single_source.py \
       --input_dir /workspace/LTX_video_training/raw_videos \
       --output_dir /workspace/LTX_video_training/dataset \
       --captions_dir /workspace/LTX_video_training/captions
   ```

3. **Check file encoding:**
   ```bash
   file video1.txt  # Should show: UTF-8 Unicode text
   ```

---

## Training Issues

### Training stops without error

**Issue:** Training stops but no error message.

**Causes & Solutions:**

1. **Disk full:**
   ```bash
   df -h /workspace
   # If full, clean up:
   ./scripts/cleanup.sh
   ```

2. **RunPod timeout:** Check if pod stopped
   ```bash
   # On RunPod dashboard, restart pod
   # Training will resume from last checkpoint if configured
   ```

3. **Check logs:**
   ```bash
   tail -100 /workspace/LTX_video_training/logs/train_h100_*.log
   ```

---

### Loss is NaN

**Issue:**
```
Step 50/10000 | Loss: nan
```

**Solutions:**
1. **Lower learning rate:**
   ```yaml
   optimization:
     learning_rate: 5e-5  # Reduce from 1e-4
   ```

2. **Increase gradient clipping:**
   ```yaml
   optimization:
     max_grad_norm: 0.5  # Tighten clipping
   ```

3. **Check data:**
   ```bash
   # Verify dataset isn't corrupted
   python scripts/validate_dataset.py
   ```

4. **Try different optimizer:**
   ```yaml
   optimization:
     use_8bit_adam: false  # Try standard Adam
   ```

---

### Training very slow

**Expected speeds:**
- **H100**: 3-4 seconds/step (batch size 4)
- **A100**: 5-7 seconds/step (batch size 2)
- **RTX 4090**: 10-15 seconds/step (batch size 1)

**If slower:**

1. **Enable optimizations:**
   ```yaml
   acceleration:
     enable_torch_compile: true  # PyTorch 2.0+
     use_flash_attention_2: true
   ```

2. **Check dataloader:**
   ```yaml
   dataset:
     dataloader_num_workers: 16  # Increase workers
     prefetch_factor: 4
   ```

3. **Monitor GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   # GPU utilization should be 90-100%
   ```

4. **Check I/O:**
   ```bash
   iotop
   # If I/O wait is high, increase workers or use faster storage
   ```

---

## Inference Issues

### Generated videos have artifacts

**Solutions:**
1. **Use higher quality checkpoint:**
   - Don't use the earliest checkpoints
   - Try checkpoint from 2000-5000 steps
   - Use EMA weights if available

2. **Adjust inference parameters:**
   ```python
   generator.generate(
       num_inference_steps=100,  # Increase steps
       guidance_scale=7.5,  # Try 6.0-9.0
   )
   ```

3. **Check trajectory quality:**
   - Trajectory visualization should be smooth
   - No sudden jumps or discontinuities

---

### Motion doesn't follow trajectory

**Causes:**
1. **Undertrained model:** Train longer (5000+ steps)
2. **Low conditioning scale:** Increase in config
   ```yaml
   conditioning:
     reference_conditioning_scale: 1.5  # Increase from 1.0
   ```
3. **Poor trajectory extraction:** Try different visualization type

---

## RunPod Specific Issues

### SSH connection lost

**Solution:**
```bash
# Training continues in background
# Reconnect and check logs:
tail -f /workspace/LTX_video_training/logs/train_h100_*.log
```

---

### Jupyter notebook kernel dies

**Causes:**
1. **Out of memory:** Restart with smaller batch
2. **Timeout:** Increase timeout in notebook settings

**Solution:**
```bash
# Check what's running
nvidia-smi
ps aux | grep python

# Kill if needed
pkill -f train_trajectory_iclora.py
```

---

### Can't download checkpoints

**Solutions:**
1. **Compress first:**
   ```bash
   cd /workspace/LTX_video_training/output
   tar -czf checkpoints.tar.gz trajectory_iclora_h100/checkpoint-5000/
   ```

2. **Use RunPod CLI:**
   ```bash
   runpod download <pod-id>:/workspace/LTX_video_training/output
   ```

3. **Upload to cloud:**
   ```bash
   # AWS S3
   pip install awscli
   aws s3 cp checkpoints.tar.gz s3://your-bucket/

   # Hugging Face Hub
   huggingface-cli upload your-username/model-name checkpoints.tar.gz
   ```

---

## Getting More Help

### Gather Information

When reporting issues, include:
```bash
# System info
nvidia-smi
python --version
pip list | grep -E "torch|transformers|diffusers"

# Training info
cat /workspace/LTX_video_training/logs/train_h100_*.log | tail -100

# Config
cat /workspace/LTX_video_training/configs/trajectory_iclora_h100.yaml

# Dataset info
wc -l /workspace/LTX_video_training/dataset/train.csv
ls -lh /workspace/LTX_video_training/dataset/videos/ | head
```

### Where to Get Help

1. **GitHub Issues**: [Your repo URL]
2. **RunPod Discord**: https://discord.gg/runpod
3. **Check logs first**: Most issues have clear error messages

---

## Common Warnings (Safe to Ignore)

### Running pip as root
```
WARNING: Running pip as the 'root' user...
```
**Safe to ignore on RunPod containers.**

---

### FutureWarning: weights_only
```
FutureWarning: You are using `torch.load` with `weights_only=False`
```
**Safe to ignore - will be fixed in future library updates.**

---

### Some weights not initialized
```
Some weights of the model checkpoint were not used...
```
**Expected when using LoRA - only LoRA weights are being used.**

---

## Still Having Issues?

1. **Pull latest code:**
   ```bash
   cd /workspace/LTX_video_training
   git pull origin main
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Try fresh start:**
   ```bash
   # Backup your data first!
   mv /workspace/LTX_video_training /workspace/LTX_video_training.backup
   git clone <repo-url> /workspace/LTX_video_training
   # Copy your data back
   cp -r /workspace/LTX_video_training.backup/raw_videos /workspace/LTX_video_training/
   ```
