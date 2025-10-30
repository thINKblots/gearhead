#!/bin/bash
# ROCm inference wrapper script
# Sets proper environment variables for AMD GPU inference

set -e

# Activate virtual environment if it exists
if [ -f "rocm/bin/activate" ]; then
    source rocm/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "env/bin/activate" ]; then
    source env/bin/activate
fi

# Find Python executable
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null || echo "python3")

# Detect GPU architecture
GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1)

# Set environment variables for ROCm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "ROCm inference environment configured"
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo ""

# Find model path
if [ -d "outputs/small_model_rocm/final_model" ]; then
    MODEL_PATH="outputs/small_model_rocm/final_model"
elif [ -d "outputs/final_model" ]; then
    MODEL_PATH="outputs/final_model"
else
    echo "Error: No trained model found"
    echo "Please run 'make train-small-rocm' first"
    exit 1
fi

# Run inference
exec $PYTHON scripts/inference.py --model "$MODEL_PATH" --tokenizer tokenizer/tokenizer.json "$@"
