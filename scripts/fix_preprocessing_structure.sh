#!/bin/bash
# Fix preprocessing directory structure
# Move files from nested dataset/videos/ structure to flat structure

set -e

echo "========================================="
echo "Fixing Preprocessing Directory Structure"
echo "========================================="
echo ""

cd /workspace/LTX_video_training/preprocessed_official

# Fix latents structure
echo "Flattening latents directory..."
if [ -d "latents/dataset/videos" ]; then
    mv latents/dataset/videos/* latents/ 2>/dev/null || true
    rm -rf latents/dataset
    echo "✓ Latents flattened"
else
    echo "  Latents already flat"
fi

# Fix reference_latents structure
echo "Flattening reference_latents directory..."
if [ -d "reference_latents/dataset/videos" ]; then
    mv reference_latents/dataset/videos/* reference_latents/ 2>/dev/null || true
    rm -rf reference_latents/dataset
    echo "✓ Reference latents flattened"
else
    echo "  Reference latents already flat"
fi

# Check conditions
echo ""
echo "Checking conditions..."
COND_COUNT=$(find conditions -name "*.pt" 2>/dev/null | wc -l)

if [ "$COND_COUNT" -eq 0 ]; then
    echo "⚠ WARNING: No condition files found!"
    echo "  Need to regenerate caption embeddings"
    echo ""
    echo "  The preprocessing may have failed to save text embeddings."
    echo "  This could be due to:"
    echo "    - Disk space issues during preprocessing"
    echo "    - Incorrect output directory settings"
    echo "    - Caption embedding save errors"
    echo ""
    echo "  Solutions:"
    echo "    1. Check preprocessing logs for errors"
    echo "    2. Re-run preprocessing with more disk space"
    echo "    3. Check if conditions were saved elsewhere"
else
    echo "✓ Found $COND_COUNT condition files"
    # Flatten conditions if needed
    if [ -d "conditions/dataset/videos" ]; then
        mv conditions/dataset/videos/* conditions/ 2>/dev/null || true
        rm -rf conditions/dataset
        echo "✓ Conditions flattened"
    fi
fi

echo ""
echo "Final structure:"
echo "  Latents: $(find latents -name '*.pt' 2>/dev/null | wc -l) files"
echo "  Reference latents: $(find reference_latents -name '*.pt' 2>/dev/null | wc -l) files"
echo "  Conditions: $(find conditions -name '*.pt' 2>/dev/null | wc -l) files"
echo ""
