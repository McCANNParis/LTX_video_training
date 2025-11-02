# Testing and Improving Your LTX-Video LoRA

## Understanding Your LoRA Type

**CRITICAL**: Your LoRA was trained with **IC-LoRA (In-Context LoRA)** for **trajectory control**.

This means:
- ✓ **Designed to work WITH reference videos** - transfers motion from reference to generated video
- ✗ **NOT designed for text-only generation** - text-only prompts won't activate the LoRA properly

## Quick Diagnosis: Is Your LoRA Working?

### Test 1: Base Model vs LoRA Comparison

This test checks if the LoRA is affecting the output at all.

```bash
cd /workspace/LTX_video_training

# Generate the same prompt with and without LoRA
python scripts/compare_base_vs_lora.py \
    --lora_path output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors \
    --prompt "A cinematic shot of a landscape, camera panning left slowly" \
    --output_base comparison_base.mp4 \
    --output_lora comparison_lora.mp4 \
    --seed 42
```

**Expected Result**:
- If videos look **identical**: LoRA is not activating (see Problem 1 below)
- If videos look **different**: LoRA is working, but may need better testing method (see Test 2)

### Test 2: IC-LoRA with Reference Video (RECOMMENDED)

This is how your LoRA was **designed** to be used.

```bash
# Test with one of your training videos as reference
python scripts/test_with_reference.py \
    --lora_path output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors \
    --reference_video dataset/videos/H3GHijO3Y4YzBOKNtHOU.mp4 \
    --prompt "A beautiful mountain landscape with dramatic lighting" \
    --output test_with_reference.mp4
```

**Note**: This script doesn't exist yet - see "Creating Reference Video Test Script" below.

## Common Problems and Solutions

### Problem 1: "LoRA not affecting output" (videos look identical)

**Symptoms**:
- Base model and LoRA outputs look the same
- 960 keys loaded but no visible effect

**Possible Causes**:
1. **IC-LoRA needs reference videos** (most likely)
   - You're testing text-only but trained for trajectory control
   - Solution: Use Test 2 with reference videos

2. **LoRA strength too low**
   - Current: Using 1.0 strength by default
   - Solution: Already at max, not the issue

3. **Training didn't converge**
   - Final loss: 0.0965 is reasonable
   - Steps: 5000 may need more
   - Solution: See "Improvement Strategies" below

### Problem 2: "Bad quality even with reference videos"

**Symptoms**:
- LoRA does affect output but quality is poor
- Blurry, inconsistent, or wrong motion

**Possible Causes**:
1. **Insufficient training steps**
   - Current: 5000 steps
   - Recommended: 10,000-20,000 steps for complex datasets

2. **Dataset issues**
   - Training data: 42 videos, only 17 valid samples used
   - Solution: Need more diverse, high-quality training data

3. **Learning rate too high/low**
   - Current: 1e-4
   - Try: 5e-5 or 2e-4

4. **LoRA rank too low**
   - Current: rank 128 (654M params)
   - Try: rank 256 for more capacity

### Problem 3: "Reference video test not working"

The official LTX-Video-Trainer's IC-LoRA inference is complex. You may need to:

1. **Check trainer documentation**:
   ```bash
   cd /workspace/LTX_video_training/LTX-Video-Trainer
   ls scripts/infer*.py
   cat scripts/infer.py | head -50
   ```

2. **Use image-to-video mode** as proxy:
   - Extract first frame from reference video
   - Use as conditioning image
   - See if motion follows reference

## Improvement Strategies

### Strategy 1: Continue Training (Quick)

Your training stopped at 5000 steps. Resume for longer:

```bash
cd /workspace/LTX_video_training/LTX-Video-Trainer

# Resume from checkpoint
python -m ltxv_trainer.cli train \
    --resume_from_checkpoint ../output/trajectory_control_official/checkpoints/checkpoint-step-4999 \
    --config ../configs/trajectory_control_h100.yaml
```

**Recommended**: Train to 10,000-15,000 steps total.

### Strategy 2: Improve Dataset (Medium Effort)

Your dataset has 42 videos but only 17 valid samples. Issues:
- Videos may be too short/long
- Resolution mismatches
- Quality issues

**Actions**:
1. **Review validation logs**:
   ```bash
   grep "Skipping" output/trajectory_control_official/train.log
   ```

2. **Check video stats**:
   ```bash
   cd /workspace/LTX_video_training
   python scripts/analyze_dataset.py dataset/videos/
   ```

3. **Add more diverse data**:
   - Aim for 100+ videos
   - Vary camera motions: pan, tilt, zoom, dolly
   - Consistent quality and resolution

### Strategy 3: Retrain with Text-Only LoRA (Complete Redo)

If you want text-only generation (no reference videos):

```bash
cd /workspace/LTX_video_training

# Create new config without IC-LoRA
cp configs/trajectory_control_h100.yaml configs/text_only_h100.yaml

# Edit config to disable trajectory conditioning
# Then train from scratch
```

**Trade-off**: Loses trajectory control capability but works with text-only prompts.

### Strategy 4: Hyperparameter Tuning

Create new config with adjusted parameters:

```yaml
# configs/improved_h100.yaml
lora:
  rank: 256  # Increased from 128
  alpha: 256

optimization:
  learning_rate: 5.0e-5  # Decreased from 1e-4
  steps: 15000  # Increased from 5000

  # Add learning rate warmup
  warmup_steps: 1000

  # Add gradient clipping
  max_grad_norm: 1.0

# Add regular checkpointing
checkpoint:
  save_interval: 1000
```

## Testing Checklist

Before considering your LoRA "done":

- [ ] Test 1: Base vs LoRA comparison shows visible difference
- [ ] Test 2: Reference video properly transfers motion
- [ ] Test 3: Multiple reference videos work consistently
- [ ] Test 4: Quality matches or exceeds base model
- [ ] Test 5: Training videos used as reference work well
- [ ] Test 6: New reference videos (not in training) work reasonably

## Expected Timeline

| Task | Time | GPU Cost (H100) |
|------|------|-----------------|
| Base vs LoRA test | 20 min | $0.60 |
| Reference video test | 30 min | $0.90 |
| Continue training 5k steps | 6 hours | $18 |
| Retrain with better config | 12 hours | $36 |
| Gather 100+ video dataset | 2-4 hours | Manual work |

## Quick Start Commands

```bash
# 1. Test if LoRA is working at all
cd /workspace/LTX_video_training
python scripts/compare_base_vs_lora.py \
    --lora_path output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors \
    --prompt "A scenic mountain landscape, camera slowly panning right" \
    --seed 42

# 2. Check training logs for issues
tail -100 output/trajectory_control_official/train.log | grep -i "loss\|error\|skip"

# 3. Resume training for more steps
cd LTX-Video-Trainer
python -m ltxv_trainer.cli train \
    --resume_from_checkpoint ../output/trajectory_control_official/checkpoints/checkpoint-step-4999 \
    --config ../configs/trajectory_control_h100.yaml
```

## Need More Help?

1. **Share comparison videos**: Upload base vs LoRA outputs so we can diagnose
2. **Check training logs**: Look for warnings or skipped samples
3. **Verify training data**: Ensure videos are diverse and high quality
4. **Consider training mode**: IC-LoRA vs text-only based on your use case

## Summary

Your LoRA is **IC-LoRA for trajectory control**:
- ✓ Requires reference videos to work properly
- ✗ Won't work well with text-only prompts
- → Test with actual training videos first
- → If quality is bad, continue training or improve dataset
