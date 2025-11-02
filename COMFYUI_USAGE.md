# Using Your Trained LoRA in ComfyUI

Your LoRA was trained with the official LTX-Video-Trainer, which uses a different key format than ComfyUI expects. Use the conversion script to make it compatible.

## Quick Start

### Step 1: Convert Your LoRA

On RunPod or your local machine with the checkpoint:

```bash
cd /workspace/LTX_video_training

# Install safetensors if needed
pip install safetensors

# Create all format variants (recommended - try each one)
python scripts/convert_lora_for_comfyui.py \
    output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors \
    --type all
```

This creates 4 variants in `output/trajectory_control_official/checkpoints/converted/`:
- `lora_weights_step_04750_comfyui_diffusers.safetensors`
- `lora_weights_step_04750_comfyui_comfy_standard.safetensors`
- `lora_weights_step_04750_comfyui_no_prefix.safetensors`
- `lora_weights_step_04750_comfyui_kohya.safetensors`

### Step 2: Copy to ComfyUI

```bash
# Copy all variants to ComfyUI
cp output/trajectory_control_official/checkpoints/converted/*.safetensors \
   /path/to/ComfyUI/models/loras/
```

### Step 3: Test in ComfyUI

1. Open ComfyUI
2. Add a "Load LoRA" node
3. Try each converted file:
   - Start with `diffusers` format
   - If that doesn't work, try `comfy_standard`
   - Then `no_prefix`, then `kohya`
4. Set LoRA strength: **0.8 to 1.0**
5. Connect to your LTX-Video model

## Advanced Usage

### Convert Specific Format

```bash
# Just diffusers format (most common)
python scripts/convert_lora_for_comfyui.py \
    output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors \
    -o my_lora_comfyui.safetensors \
    --type diffusers
```

### Inspect LoRA Structure

```bash
# See what keys are in your LoRA
python scripts/convert_lora_for_comfyui.py \
    output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors \
    --inspect-only
```

## Conversion Types Explained

| Type | Description | When to Use |
|------|-------------|-------------|
| **diffusers** | Standard diffusers format: `lora_A` → `lora_down`, `lora_B` → `lora_up` | Try this first - most ComfyUI nodes expect this |
| **comfy_standard** | ComfyUI standard with prefix removed | If diffusers fails |
| **no_prefix** | No "transformer." prefix | For some custom ComfyUI nodes |
| **kohya** | Kohya-ss style format | If using Kohya-based workflows |

## Troubleshooting

### LoRA Keys Still Not Loading

If all formats fail, the issue might be:

1. **ComfyUI LTX-Video version mismatch**
   - Update your ComfyUI LTX-Video nodes to latest version
   - Check: https://github.com/Lightricks/LTX-Video

2. **Model architecture mismatch**
   - Ensure you're loading LTX-Video **13B** model in ComfyUI
   - Your LoRA was trained on 13B, not 5B or other variants

3. **Use official inference instead**
   ```bash
   cd /workspace/LTX_video_training/LTX-Video-Trainer
   python scripts/infer.py \
       --model_path Lightricks/LTX-Video \
       --lora_path ../output/trajectory_control_official/checkpoints/lora_weights_step_04750.safetensors \
       --prompt "Your prompt here" \
       --output_path output.mp4
   ```

### Checking Which Format Worked

In ComfyUI console, look for:
- ✓ **Good**: "LoRA applied with X keys"
- ✗ **Bad**: "lora key not loaded: ..."

If you still see "key not loaded" messages, try the next format.

## Example ComfyUI Workflow

```
[Load Checkpoint: LTX-Video 13B]
           ↓
[Load LoRA: lora_weights_step_04750_comfyui_diffusers.safetensors]
  strength: 0.9
           ↓
[CLIPTextEncode: "Camera panning across landscape"]
           ↓
[LTXVideoSampler]
  steps: 50
  cfg: 3.5
  frames: 121
           ↓
[Save Video]
```

## Using Trajectory Conditioning

Your LoRA was trained for **trajectory-guided generation**. To use it properly:

1. **With reference video** (in ComfyUI with IC-LoRA support):
   - Provide a reference video showing desired motion
   - LoRA will transfer that motion to your generated content

2. **With text descriptions** (fallback):
   - Use motion-heavy prompts: "camera panning left slowly"
   - The LoRA learned motion patterns from your training data

## File Sizes

Converted files will be the same size as original (~2.5 GB for rank 128).

## Need Help?

If conversion isn't working:
1. Share which format you tried in ComfyUI
2. Share the console error messages
3. Verify you're using LTX-Video 13B model
4. Try the official inference script as fallback
