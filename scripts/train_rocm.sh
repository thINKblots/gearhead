#!/bin/bash
# ROCm training wrapper script
# Sets proper environment variables for AMD GPU training

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
echo "Detected GPU architecture: $GPU_ARCH"

# Set environment variables for ROCm
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-$GPU_ARCH}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Workaround for RDNA2 (gfx103x) embedding layer issues
# Force use of compatible GPU target for embedding operations
if [[ "$GPU_ARCH" == gfx103* ]]; then
    echo "Detected RDNA2 GPU - applying embedding layer workaround..."
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    # Use ROCm 5.7 compatible kernels
    export PYTORCH_ROCM_ARCH=gfx1030
fi

# For debugging (uncomment if needed)
# export AMD_SERIALIZE_KERNEL=3
# export TORCH_USE_HIP_DSA=1
# export HIP_LAUNCH_BLOCKING=1

# Select the discrete GPU (GPU 0 - RX 6750 XT, not the integrated GPU)
export CUDA_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

# Enable TF32 for better performance
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

echo "Environment configured for ROCm training"
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo ""

# Run training
exec $PYTHON scripts/train.py "$@"
