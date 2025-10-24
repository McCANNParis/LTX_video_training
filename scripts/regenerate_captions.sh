#!/bin/bash
# Regenerate caption embeddings only (without re-encoding videos)

set -e

echo "========================================="
echo "Regenerating Caption Embeddings"
echo "========================================="
echo ""

cd /workspace/LTX_video_training

# Check if dataset.json exists
if [ ! -f "dataset.json" ]; then
    echo "Error: dataset.json not found!"
    exit 1
fi

# Create conditions directory if it doesn't exist
mkdir -p preprocessed_official/conditions

echo "Creating caption embedding script..."

# Create a Python script to generate embeddings
python3 << 'PYTHON_SCRIPT'
import json
import torch
from pathlib import Path
from transformers import T5EncoderModel, T5TokenizerFast
from tqdm import tqdm

print("Loading dataset.json...")
with open('dataset.json', 'r') as f:
    dataset = json.load(f)

print(f"Found {len(dataset)} entries")

print("\nLoading T5 text encoder...")
tokenizer = T5TokenizerFast.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16,
).to("cuda")

print("Text encoder loaded!")

output_dir = Path("preprocessed_official/conditions")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nProcessing {len(dataset)} captions...")

for entry in tqdm(dataset, desc="Generating embeddings"):
    caption = entry['caption']
    video_path = entry['video']

    # Extract video filename without extension
    video_name = Path(video_path).stem  # e.g., "video_000000_clip_000"
    output_path = output_dir / f"{video_name}.pt"

    # Skip if already exists
    if output_path.exists():
        continue

    # Tokenize
    text_inputs = tokenizer(
        caption,
        padding="max_length",
        max_length=120,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )[0]

    # Save embeddings with correct key names expected by trainer
    torch.save(
        {
            'prompt_embeds': text_embeddings.cpu(),
            'prompt_attention_mask': text_inputs.attention_mask.cpu(),
        },
        output_path
    )

print(f"\n✓ Caption embeddings saved to {output_dir}")
print(f"  Generated {len(list(output_dir.glob('*.pt')))} embedding files")

PYTHON_SCRIPT

echo ""
echo "========================================="
echo "Caption Embeddings Regenerated!"
echo "========================================="
echo ""
echo "Conditions: $(find preprocessed_official/conditions -name '*.pt' 2>/dev/null | wc -l) files"
echo ""
echo "Now you can train with:"
echo "  ./scripts/train_with_official.sh"
echo ""
