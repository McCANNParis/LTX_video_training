# ⚡ RunPod Quick Start - 5 Minute Setup

Get started with Trajectory-Guided LTX Video Training on RunPod in 5 minutes.

## 1️⃣ Deploy Pod (1 minute)

1. Go to [runpod.io](https://runpod.io)
2. Click **"Deploy"** on a GPU:
   - **Recommended**: A100 80GB or A6000 48GB
   - **Budget**: RTX 4090 24GB (use 24GB config)
3. Select template: **PyTorch 2.4.0**
4. Disk space: **200GB+**
5. Click **"Deploy"**

## 2️⃣ Connect (30 seconds)

1. Wait for pod status = **"Running"**
2. Click **"Connect"** → **"Start Jupyter Lab"**
3. Open JupyterLab URL in browser

## 3️⃣ Setup (2 minutes)

**In JupyterLab, open Terminal and run:**

```bash
cd /workspace
wget https://raw.githubusercontent.com/your-repo/LTX_video_training/main/runpod_setup.sh
chmod +x runpod_setup.sh
./runpod_setup.sh
```

Wait for setup to complete (~2 minutes).

## 4️⃣ Upload Videos (varies)

**Option A: JupyterLab Upload** (for small datasets)
- Click Upload button in JupyterLab
- Upload to `/workspace/raw_videos/`

**Option B: Command line** (for URLs)
```bash
cd /workspace/raw_videos
wget https://example.com/your-video.mp4
```

## 5️⃣ Start Training (1 minute)

**Option A: Interactive Launcher**
```bash
cd /workspace/LTX_video_training
./scripts/runpod_launcher.sh
```
Then follow the menu.

**Option B: Jupyter Notebooks**
1. Navigate to `/workspace/LTX_video_training/notebooks/`
2. Open `01_Quick_Start.ipynb`
3. Run cells

## 📊 Monitor Training

**GPU Usage:**
```bash
watch -n 1 nvidia-smi
```

**Training Logs:**
```bash
cd /workspace/LTX_video_training
tail -f logs/*.log
```

**Weights & Biases:**
- Configure in config YAML
- View at wandb.ai

## 💾 Download Results

**After training completes:**

```bash
cd /workspace/LTX_video_training/output
tar -czf trained_model.tar.gz trajectory_iclora/
# Then download via JupyterLab
```

## 🚨 Troubleshooting

### Out of Memory?
Edit config: `/workspace/LTX_video_training/configs/trajectory_iclora_runpod_24gb.yaml`

```yaml
optimization:
  train_batch_size: 1
  gradient_accumulation_steps: 16  # Increase this
```

### Disk Full?
```bash
cd /workspace/LTX_video_training
./scripts/cleanup.sh
```

### Training Stopped?
It resumes automatically from last checkpoint. Just run train command again.

## 📚 Full Documentation

- **Main README**: `/workspace/LTX_video_training/README.md`
- **RunPod Guide**: `/workspace/LTX_video_training/RUNPOD_DEPLOYMENT.md`
- **Best Practices**: `/workspace/LTX_video_training/docs/BEST_PRACTICES.md`

## 🎯 Recommended Workflow

1. **Test with small dataset** (10-20 videos)
   - Use `trajectory_iclora_runpod_24gb.yaml`
   - Train for 1000 steps
   - Validate inference works

2. **Scale to full dataset** (500+ videos)
   - Use `trajectory_iclora_high_quality.yaml`
   - Train for 10,000 steps
   - Monitor quality metrics

3. **Deploy** trained model
   - Download LoRA weights
   - Use for inference
   - Share on HuggingFace Hub

## 💰 Cost Estimate

**Training 10K steps:**
- A100 80GB (Spot): ~$33 (24 hours)
- A6000 48GB (Spot): ~$14 (36 hours)
- RTX 4090 (Spot): ~$19 (48 hours)

**Tips to save money:**
- Use Spot instances
- Stop pod when not training
- Start with small test run

## ⚡ Quick Commands

```bash
# Setup
cd /workspace && ./runpod_setup.sh

# Interactive launcher
cd /workspace/LTX_video_training && ./scripts/runpod_launcher.sh

# Prepare dataset
python scripts/prepare_dataset.py --input_dir /workspace/raw_videos --output_dir ./dataset

# Train (24GB GPU)
./scripts/train.sh configs/trajectory_iclora_runpod_24gb.yaml 1

# Monitor GPU
watch -n 1 nvidia-smi

# Cleanup
./scripts/cleanup.sh
```

## 🆘 Need Help?

1. Check `/workspace/LTX_video_training/logs/`
2. Review `RUNPOD_DEPLOYMENT.md`
3. RunPod Discord: [discord.gg/runpod](https://discord.gg/runpod)
4. GitHub Issues: [your-repo-url]

---

**That's it! You're ready to train.** 🚀

For detailed guidance, open the Jupyter notebooks or read the full documentation.
