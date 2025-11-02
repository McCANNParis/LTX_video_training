# Standard Text-to-Video LoRA Training

Train a LoRA to improve realism **WITHOUT** trajectory conditioning.

## Why Standard LoRA Instead of IC-LoRA?

**Advantages:**
- ✅ Works with simple text prompts (no reference videos needed)
- ✅ Easier to test and use
- ✅ Better compatibility with diffusers
- ✅ Focuses on improving realism and quality
- ✅ No green tint issues

**What You Trained Before:**
- IC-LoRA for trajectory control (requires reference videos)
- Difficult to test with diffusers
- Text-only prompts don't activate it properly

## Quick Start

### 1. Preprocess Your Data (if not already done)

If you already preprocessed for trajectory control, **you can reuse it**:

```bash
# Check if preprocessed data exists
ls /workspace/LTX_video_training/preprocessed_official/

# If not, preprocess:
cd /workspace/LTX_video_training/LTX-Video-Trainer
python -m ltxv_trainer.cli preprocess \
    --config ../configs/standard_text_lora_h100.yaml
```

### 2. Start Training

```bash
cd /workspace/LTX_video_training
bash scripts/train_standard_lora.sh
```

**Or manually:**

```bash
cd /workspace/LTX_video_training/LTX-Video-Trainer

python -m ltxv_trainer.cli train \
    --config ../configs/standard_text_lora_h100.yaml
```

### 3. Monitor Progress

```bash
# Watch training log
tail -f /workspace/LTX_video_training/output/standard_text_lora/train.log

# Check checkpoints
ls -lh /workspace/LTX_video_training/output/standard_text_lora/checkpoints/
```

## Training Configuration

**Model:**
- LTX-Video 13B (0.9.7 dev)
- LoRA rank: 128 (654M trainable parameters)

**Training:**
- Steps: 10,000 (vs 5,000 for IC-LoRA)
- Batch size: 1 (effective 8 with gradient accumulation)
- Learning rate: 1e-4 with cosine schedule
- Warmup: 500 steps

**Optimization:**
- Mixed precision: BF16
- Gradient checkpointing: enabled
- H100 optimized

**Data:**
- Same 42 videos from your dataset
- ~17 valid samples
- Text-to-video (NO trajectory conditioning)

## Expected Training Time

On H100 80GB:
- **10,000 steps**: ~12-15 hours
- Checkpoints every 500 steps
- ~20 checkpoints total

## Testing Your LoRA

### During Training

Check latest checkpoint:

```bash
ls -lt /workspace/LTX_video_training/output/standard_text_lora/checkpoints/ | head -5
```

### After Training

Test with proper inference (no green tint):

```bash
cd /workspace/LTX_video_training

# Test at step 5000
python scripts/test_with_proper_guidance.py \
    --lora_path output/standard_text_lora/checkpoints/lora_weights_step_05000.safetensors \
    --prompt "A cinematic landscape with beautiful lighting and smooth camera movement" \
    --guidance_scale 3.5 \
    --seed 42

# Compare base vs LoRA
python scripts/test_with_proper_guidance.py \
    --lora_path output/standard_text_lora/checkpoints/lora_weights_step_10000.safetensors \
    --prompt "A person walking through a park with natural motion" \
    --seed 42
```

## Resume Training

If training stops, resume from latest checkpoint:

```bash
cd /workspace/LTX_video_training/LTX-Video-Trainer

# Find latest checkpoint
LATEST=$(ls -t ../output/standard_text_lora/checkpoints/checkpoint-step-* | head -1)

# Resume
python -m ltxv_trainer.cli train \
    --config ../configs/standard_text_lora_h100.yaml \
    --resume_from_checkpoint $LATEST
```

## Upload to HuggingFace

```bash
cd /workspace/LTX_video_training

python scripts/upload_lora_to_hf.py \
    --lora_path output/standard_text_lora/checkpoints/lora_weights_step_10000.safetensors \
    --repo_name "ltx-video-realism-lora"
```

## Key Differences from IC-LoRA Training

| Feature | IC-LoRA (Previous) | Standard LoRA (New) |
|---------|-------------------|---------------------|
| **Conditioning** | Reference videos (trajectory) | Text only |
| **Testing** | Needs reference videos | Works with text prompts |
| **Compatibility** | Limited (diffusers issues) | Full (works everywhere) |
| **Use Case** | Motion transfer | Realism improvement |
| **Steps** | 5,000 | 10,000 |
| **Green Tint** | Issue with testing | Fixed with proper settings |

## Monitoring Training

### Check Loss

```bash
grep "loss:" /workspace/LTX_video_training/output/standard_text_lora/train.log | tail -20
```

### Check GPU Usage

```bash
nvidia-smi
```

### Check Disk Space

```bash
df -h /workspace
```

## Troubleshooting

### Out of Memory

Reduce batch size in config:

```yaml
optimization:
  batch_size: 1
  gradient_accumulation_steps: 4  # Reduce from 8
```

### Training Too Slow

- Check GPU utilization: `nvidia-smi`
- Reduce `num_dataloader_workers` to 8
- Disable gradient checkpointing (uses more VRAM)

### Poor Results

- Train longer (15,000-20,000 steps)
- Increase LoRA rank to 256
- Lower learning rate to 5e-5
- Add more diverse training data

## Expected Results

After 10,000 steps:
- Better motion coherence
- Improved lighting and realism
- More consistent textures
- Smoother camera movements
- No trajectory control (use text prompts only)

## Next Steps After Training

1. **Test extensively** with different prompts
2. **Compare** base model vs LoRA outputs
3. **Upload to HuggingFace** for sharing
4. **Fine-tune further** if needed (continue training)
5. **Use in production** with proper inference settings

## Questions?

Check the main README or:
- Official LTX-Video docs: https://github.com/Lightricks/LTX-Video
- Diffusers docs: https://huggingface.co/docs/diffusers
