#!/bin/bash
# Pre-training wrapper for ROCm (AMD GPUs)
# Configures environment for optimal ROCm pre-training

set -e

# Detect GPU architecture
GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1)

echo "================================"
echo "ROCm Pre-training Configuration"
echo "================================"
echo "GPU Architecture: $GPU_ARCH"

# Workaround for RDNA2 (gfx103x) embedding layer issues
if [[ "$GPU_ARCH" == gfx103* ]]; then
    echo "Detected RDNA2 GPU - applying compatibility workarounds..."
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export PYTORCH_ROCM_ARCH=gfx1030
fi

# ROCm environment variables
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "HIP Device: $HIP_VISIBLE_DEVICES"
echo "================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run pre-training with Python from venv
exec python scripts/pretrain.py "$@"
