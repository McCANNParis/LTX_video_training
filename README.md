# LTX Video - Realism LoRA Training

Train a LoRA to improve video realism in LTX-Video 13B using your own footage. Simple text-to-video training without trajectory conditioning.

---

## 🎯 What This Does

Train a LoRA (Low-Rank Adaptation) to fine-tune LTX-Video 13B on your realistic video footage. The trained LoRA will:
- ✅ Improve visual realism and quality
- ✅ Work with standard text prompts (no reference videos needed)
- ✅ Maintain the base model's capabilities while adapting to your style
- ✅ Be compatible with ComfyUI and other diffusion pipelines

**Training Mode:** Text-to-Video LoRA (no trajectory conditioning)

---

## 🚀 Quick Start

### **On RunPod:**

```bash
# 1. Clone repository
cd /workspace
git clone https://github.com/McCANNParis/LTX_video_training.git
cd LTX_video_training

# 2. Switch to realism training branch
git checkout claude/lora-training-realism-011CUjmk3msVokkZVQwPepWK

# 3. Clone official trainer
git clone https://github.com/Lightricks/LTX-Video-Trainer.git
git checkout LTX-Video-Trainer/configs/realism_lora.yaml

# 4. Install dependencies
./install_dependencies.sh

# 5. Add your training data
mkdir -p raw_videos
# Upload your .mp4 and .txt files to raw_videos/

# 6. Start training!
./scripts/quick_start_realism.sh
```

📚 **See [RUNPOD_SETUP.md](RUNPOD_SETUP.md) for detailed setup instructions**

---

## 📂 Dataset Structure

```
raw_videos/
├── video_001.mp4       # Your video file
├── video_001.txt       # Caption (same name as video!)
├── video_002.mp4
├── video_002.txt
└── ...
```

**Requirements:**
- Each `.mp4` must have a matching `.txt` caption file
- Videos: 704x1216 resolution, 121 frames (~4 seconds at 30fps), MP4 format
- Captions: Include realism keywords like "photorealistic", "natural lighting", "detailed"

**Example caption:**
```
A person walking in a park, photorealistic, natural lighting, high detail, smooth motion
```

---

## 📋 Training Steps

### **Option 1: Interactive Guide (Recommended)**
```bash
cd /workspace/LTX_video_training
./scripts/quick_start_realism.sh
```

### **Option 2: Manual Steps**
```bash
# 1. Preprocess dataset
./scripts/prepare_for_realism_training.sh

# 2. Train (disable torch compile for stability)
export TORCH_COMPILE_DISABLE=1
./scripts/train_realism_lora.sh

# 3. Test your LoRA
python scripts/test_realism_lora.py --compare
```

### **Check Status Anytime:**
```bash
./check_setup.sh
```

---

## ⚙️ Training Configuration

**Default settings** (optimized for H100):
- **LoRA Rank:** 128 (654M parameters)
- **Training Steps:** 5000
- **Batch Size:** 4 (effective batch = 8 with gradient accumulation)
- **Learning Rate:** 1.0e-4
- **Precision:** bfloat16
- **Hardware:** Auto-detects and uses all available GPUs

**Training Time Estimates:**
- 1x H100: ~17.5 hours
- 2x H100: ~8-9 hours
- 4x H100: ~4-5 hours

**Config file:** `LTX-Video-Trainer/configs/realism_lora.yaml`

---

## 🧪 Testing Your LoRA

```bash
# Test with your LoRA
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

## 🔧 Utilities

### **Convert for ComfyUI**
```bash
python scripts/convert_lora_for_comfyui.py \
    --input output/realism_lora/checkpoints/lora_weights_step_05000.safetensors \
    --output realism_lora_comfyui.safetensors \
    --conversion-type diffusers
```

### **Upload to HuggingFace**
```bash
python scripts/upload_lora_to_hf.py \
    --lora_path output/realism_lora/checkpoints/lora_weights_step_05000.safetensors \
    --repo_id your-username/ltx-realism-lora \
    --token YOUR_HF_TOKEN
```

---

## 💡 Caption Writing Tips

**Good captions include:**
- Visual details: "detailed textures", "natural lighting", "high detail"
- Camera work: "camera pans left", "static shot", "slow zoom"
- Realism keywords: "photorealistic", "natural", "realistic motion"
- Scene description: "outdoor park", "urban street", "indoor office"

**Examples:**
```
Close-up of hands typing on keyboard, shallow depth of field, natural office lighting, photorealistic
Camera slowly panning across mountain landscape, high detail, realistic textures, golden hour lighting
Person walking towards camera in a park, natural daylight, detailed facial features, smooth motion
```

---

## 📚 Documentation

- **[REALISM_LORA_GUIDE.md](REALISM_LORA_GUIDE.md)** - Complete training guide, best practices, troubleshooting
- **[RUNPOD_SETUP.md](RUNPOD_SETUP.md)** - Detailed RunPod setup instructions
- **`./check_setup.sh`** - Check your current setup status

---

## 🛠️ System Requirements

- **GPU:** H100 or H200 (single or multi-GPU)
  - Single H100 80GB minimum
  - Multi-GPU automatically detected and used
- **Storage:** ~50GB for model + dependencies
- **Platform:** RunPod, Linux with CUDA

---

## 📁 Repository Structure

```
LTX_video_training/
├── README.md                           # This file
├── REALISM_LORA_GUIDE.md               # Complete training guide
├── RUNPOD_SETUP.md                     # RunPod setup instructions
├── install_dependencies.sh             # Dependency installer
├── check_setup.sh                      # Setup checker
├── LTX-Video-Trainer/
│   └── configs/
│       └── realism_lora.yaml           # Training configuration
├── scripts/
│   ├── prepare_for_realism_training.sh # Dataset preprocessing
│   ├── train_realism_lora.sh           # Training launcher
│   ├── test_realism_lora.py            # LoRA testing
│   ├── quick_start_realism.sh          # Interactive quick start
│   ├── convert_lora_for_comfyui.py     # Format conversion
│   └── upload_lora_to_hf.py            # HuggingFace upload
├── raw_videos/                         # Your training data (create this)
├── output/
│   └── realism_lora/
│       └── checkpoints/                # Trained LoRA weights
└── logs/                               # Training logs
```

---

## 🤝 Support

- **Issues:** https://github.com/McCANNParis/LTX_video_training/issues
- **Official LTX-Video:** https://github.com/Lightricks/LTX-Video
- **Official Trainer:** https://github.com/Lightricks/LTX-Video-Trainer

---

## 📝 License

This training code is provided as-is. The LTX-Video model and official trainer have their own licenses - please review them before use.

---

**Ready to improve your video generation? Start training! 🎥**
