# 🚀 RunPod Deployment Guide

Complete guide for deploying and using the Trajectory-Guided LTX Video Training System on RunPod.

## 📋 Prerequisites

### RunPod Account Setup
1. Create account at [runpod.io](https://www.runpod.io/)
2. Add payment method
3. Recommended GPU: A100 80GB, A6000 48GB, or RTX 4090 24GB

### Recommended Pod Configuration

**For Training:**
- **GPU**: 1x A100 80GB (optimal) or 1x A6000 48GB (good)
- **Disk**: 200GB+ (persistent storage recommended)
- **Template**: PyTorch 2.4.0

**For Inference Only:**
- **GPU**: 1x RTX 4090 24GB (sufficient with quantization)
- **Disk**: 100GB+
- **Template**: PyTorch 2.4.0

## 🎬 Quick Start (5 Minutes)

### Step 1: Deploy RunPod Pod

1. Go to [runpod.io/console/gpu-cloud](https://www.runpod.io/console/gpu-cloud)
2. Click "Deploy" on a GPU
3. Select **PyTorch 2.4.0** template
4. Set disk space (200GB+ recommended)
5. Click "Deploy On-Demand" or "Deploy Spot" (cheaper)

### Step 2: Connect to JupyterLab

1. Wait for pod to be "Running" (30-60 seconds)
2. Click "Connect" → "Start Jupyter Lab"
3. Copy the JupyterLab URL (format: `https://your-pod-id.runpod.net/`)
4. Open URL in browser

### Step 3: Run Setup Script

Open a Terminal in JupyterLab and run:

```bash
cd /workspace

# Download setup script
wget https://raw.githubusercontent.com/your-repo/LTX_video_training/main/runpod_setup.sh

# Make executable
chmod +x runpod_setup.sh

# Run setup
./runpod_setup.sh
```

Setup will take 5-10 minutes and will:
- Install all dependencies
- Download the code
- Configure the environment
- Set up Jupyter kernels
- Create helper scripts

### Step 4: Start Using Notebooks

1. In JupyterLab, navigate to `/workspace/LTX_video_training/notebooks/`
2. Open `01_Quick_Start.ipynb`
3. Follow the notebook instructions

## 📂 File Organization on RunPod

```
/workspace/
├── LTX_video_training/          # Main project
│   ├── src/                     # Source code
│   ├── scripts/                 # Helper scripts
│   ├── configs/                 # Training configs
│   ├── notebooks/               # Jupyter notebooks
│   ├── dataset/                 # Processed dataset
│   ├── models/                  # Downloaded models
│   ├── output/                  # Training outputs
│   └── checkpoints/             # Model checkpoints
├── raw_videos/                  # Your uploaded videos
├── .cache/                      # HuggingFace cache
└── wandb/                       # W&B logs
```

## 💾 Data Management

### Uploading Videos

**Option 1: JupyterLab Upload** (Small files)
1. In JupyterLab, click Upload button
2. Select video files
3. Upload to `/workspace/raw_videos/`

**Option 2: RunPod CLI** (Large datasets)
```bash
# Install RunPod CLI on your local machine
pip install runpod

# Upload files
runpod send /local/path/to/videos pod-id:/workspace/raw_videos/
```

**Option 3: wget/curl** (From URLs)
```bash
cd /workspace/raw_videos
wget https://example.com/video.mp4
```

**Option 4: Google Drive/Dropbox**
```bash
# Install gdown for Google Drive
pip install gdown

# Download from Google Drive
gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID
```

### Downloading Results

**Option 1: JupyterLab Download**
- Right-click file → Download

**Option 2: RunPod CLI**
```bash
# Download from pod to local machine
runpod receive pod-id:/workspace/LTX_video_training/output/ /local/path/
```

**Option 3: Upload to Cloud Storage**
```python
# In notebook or script
import boto3  # AWS S3
# or
from google.cloud import storage  # Google Cloud Storage
```

## 🎓 Training Workflow

### 1. Prepare Dataset (30 minutes - 2 hours)

```bash
cd /workspace/LTX_video_training

# Upload videos to /workspace/raw_videos/ first

# Then run preparation
python scripts/prepare_dataset.py \
    --input_dir /workspace/raw_videos \
    --output_dir ./dataset \
    --num_workers 4
```

Or use the notebook: `02_Dataset_Preparation.ipynb`

### 2. Configure Training (5 minutes)

Edit `configs/trajectory_iclora_high_quality.yaml`:

```yaml
# For A100 80GB
model:
  transformer:
    load_in_8bit: false
optimization:
  train_batch_size: 2
  gradient_accumulation_steps: 4

# For A6000 48GB
model:
  transformer:
    load_in_8bit: true
optimization:
  train_batch_size: 1
  gradient_accumulation_steps: 8

# For RTX 4090 24GB
model:
  transformer:
    load_in_8bit: true
  vae:
    load_in_8bit: true
optimization:
  train_batch_size: 1
  gradient_accumulation_steps: 16
```

### 3. Start Training (24-48 hours)

**Terminal Method:**
```bash
cd /workspace/LTX_video_training
./scripts/train.sh configs/trajectory_iclora_high_quality.yaml 1
```

**Notebook Method:**
- Open `03_Training.ipynb`
- Follow cells step by step

### 4. Monitor Training

**Option 1: Terminal**
```bash
# GPU usage
./scripts/monitor_gpu.sh

# Training logs
./scripts/monitor_training.sh
```

**Option 2: Weights & Biases**
- Configure W&B in config YAML
- View training at wandb.ai

**Option 3: TensorBoard**
```bash
# Start TensorBoard
tensorboard --logdir /workspace/LTX_video_training/logs --port 6006

# Then connect via RunPod TCP port
```

### 5. Generate Videos (Inference)

Use notebook `04_Inference.ipynb` or:

```bash
python src/inference/trajectory_guided_inference.py \
    --model_path ./models/ltx-video-13b \
    --lora_path ./output/trajectory_iclora/final \
    --mode reference \
    --input_image examples/input.jpg \
    --reference_video examples/reference.mp4 \
    --prompt "cinematic shot" \
    --output result.mp4
```

## 💰 Cost Estimation

### Training Costs (10,000 steps)

| GPU | $/hour | Training Time | Total Cost |
|-----|--------|---------------|------------|
| A100 80GB (On-Demand) | $2.89 | 24 hours | ~$70 |
| A100 80GB (Spot) | $1.39 | 24 hours | ~$33 |
| A6000 48GB (On-Demand) | $0.79 | 36 hours | ~$28 |
| A6000 48GB (Spot) | $0.39 | 36 hours | ~$14 |
| RTX 4090 24GB (Spot) | $0.39 | 48 hours | ~$19 |

*Prices as of 2025, subject to change*

### Cost Optimization Tips

1. **Use Spot Instances** (50-70% cheaper)
   - May be interrupted, but training resumes from checkpoint
   - Best for non-urgent training

2. **Stop Pod When Not Training**
   - Only pay when pod is running
   - Persistent storage keeps your data

3. **Use Smaller Dataset for Testing**
   - Test with 100 videos first
   - Scale up once pipeline works

4. **Batch Process Inference**
   - Generate multiple videos in one session
   - Minimize pod rental time

## 🔧 Troubleshooting

### Pod Won't Start

**Issue**: Pod stuck in "Provisioning"
- **Solution**: Try different GPU type or region

**Issue**: Out of capacity
- **Solution**: Use Spot instance or try different time

### Out of Memory During Training

**Symptoms**: CUDA OOM error

**Solutions**:
```yaml
# In config YAML, reduce batch size
optimization:
  train_batch_size: 1
  gradient_accumulation_steps: 16  # Increase this

# Enable quantization
model:
  transformer:
    load_in_8bit: true
```

### Slow Training

**Issue**: Training much slower than expected

**Check**:
```bash
# Verify GPU is being used
nvidia-smi

# Check GPU utilization (should be >80%)
watch -n 1 nvidia-smi
```

**Solutions**:
- Increase `dataloader_num_workers` in config
- Enable `torch_compile` if using PyTorch 2.0+
- Check disk I/O (SSD vs HDD)

### Connection Issues

**Issue**: JupyterLab disconnects frequently

**Solutions**:
- Check internet connection
- Use SSH instead of web interface
- Save work frequently

### Disk Space Full

```bash
# Check disk usage
df -h /workspace

# Clean up
./scripts/cleanup.sh

# Remove old checkpoints
rm -rf /workspace/LTX_video_training/checkpoints/checkpoint-*
# (keep only the latest)
```

## 🔐 Security Best Practices

### API Keys and Secrets

**Never commit sensitive data!**

Create `.env` file:
```bash
# /workspace/LTX_video_training/.env
WANDB_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

Load in scripts:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Secure Downloads

Only download models from trusted sources:
- HuggingFace official repos
- Verified publishers

## 📊 Monitoring Resources

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# Log GPU usage
nvidia-smi dmon -s mu -d 10 > gpu_log.txt
```

### Disk Monitoring

```bash
# Overall disk usage
df -h /workspace

# Directory sizes
du -sh /workspace/*

# Find large files
du -h /workspace | sort -rh | head -20
```

### Process Monitoring

```bash
# Training process
ps aux | grep python

# Kill training if needed
pkill -f train_trajectory
```

## 🔄 Pause and Resume

### Save Checkpoint Before Stopping

Training auto-saves every N steps (configured in YAML).

To manually save:
```python
# In training script/notebook
# Checkpoint is automatically saved
# Resume with: --resume_from_checkpoint latest
```

### Resume Training

```bash
cd /workspace/LTX_video_training

./scripts/train.sh configs/trajectory_iclora_high_quality.yaml 1
# Automatically resumes from latest checkpoint if available
```

Or specify checkpoint:
```bash
accelerate launch src/training/train_trajectory_iclora.py \
    --config configs/trajectory_iclora_high_quality.yaml \
    --resume_from_checkpoint ./checkpoints/checkpoint-5000
```

## 📦 Exporting Trained Model

### Option 1: Download Directly

```bash
# Compress trained model
cd /workspace/LTX_video_training/output
tar -czf trained_lora.tar.gz trajectory_iclora/

# Download via JupyterLab or RunPod CLI
```

### Option 2: Upload to HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/workspace/LTX_video_training/output/trajectory_iclora/final",
    repo_id="your-username/ltxv-trajectory-lora",
    repo_type="model"
)
```

### Option 3: Cloud Storage

```bash
# AWS S3
aws s3 sync /workspace/LTX_video_training/output/ s3://your-bucket/ltxv-training/

# Google Cloud Storage
gsutil -m rsync -r /workspace/LTX_video_training/output/ gs://your-bucket/ltxv-training/
```

## 🆘 Support and Resources

### Getting Help

1. **Check Logs**
   ```bash
   cat /workspace/LTX_video_training/logs/*.log
   ```

2. **RunPod Discord**: [discord.gg/runpod](https://discord.gg/runpod)

3. **Project Issues**: GitHub Issues (your repo)

4. **W&B Support**: If using Weights & Biases

### Useful Links

- RunPod Docs: https://docs.runpod.io/
- PyTorch Docs: https://pytorch.org/docs/
- Diffusers Docs: https://huggingface.co/docs/diffusers/
- LTX Video: https://github.com/Lightricks/LTX-Video

## 🎯 Quick Reference Commands

```bash
# Setup
./runpod_setup.sh

# Prepare dataset
python scripts/prepare_dataset.py --input_dir /workspace/raw_videos --output_dir ./dataset

# Train
./scripts/train.sh configs/trajectory_iclora_high_quality.yaml 1

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor training
tail -f logs/*.log

# Cleanup
./scripts/cleanup.sh

# Check disk
df -h /workspace

# Inference
python src/inference/trajectory_guided_inference.py --mode reference --input_image input.jpg --reference_video ref.mp4 --output result.mp4
```

## ✅ Pre-Flight Checklist

Before starting training:

- [ ] Pod is running with sufficient GPU memory
- [ ] Disk space > 200GB free
- [ ] Videos uploaded to `/workspace/raw_videos/`
- [ ] Setup script completed successfully
- [ ] Dataset preparation finished
- [ ] Training config reviewed and customized
- [ ] W&B or TensorBoard configured (optional)
- [ ] Checkpointing enabled in config
- [ ] Budget allocated for training time

---

**Happy Training on RunPod! 🚀**

For questions or issues, check the main README.md or open an issue on GitHub.
