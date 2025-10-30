#!/bin/bash
# Quick test to verify ROCm training setup before full training run

set -e

echo "=========================================="
echo "ROCm Training Setup Test"
echo "=========================================="
echo ""

# Activate venv
if [ -f "rocm/bin/activate" ]; then
    source rocm/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found"
    exit 1
fi

# Check PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__} with ROCm {torch.version.hip}')"

# Check GPU
python -c "import torch; assert torch.cuda.is_available(), 'GPU not available'; print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')"

# Set environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "✓ Environment configured (HSA_OVERRIDE_GFX_VERSION=10.3.0)"
echo ""

# Test model forward pass
echo "Testing model forward pass..."
python -c "
import sys
sys.path.insert(0, 'src')
import torch
from gearhead.model.gearhead_model import GearheadConfig, GearheadModel

config = GearheadConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2, num_attention_heads=4)
model = GearheadModel(config).cuda()
test_input = torch.randint(0, 100, (2, 10)).cuda()
logits, _ = model(test_input)
print(f'✓ Model forward pass successful! Output shape: {logits.shape}')
"

echo ""
echo "=========================================="
echo "All tests passed! Ready for training."
echo "Run: make train-small-rocm"
echo "=========================================="
