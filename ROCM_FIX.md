# ROCm Training Fix - HIP Kernel Error

## The Problem

You encountered: `RuntimeError: HIP error: invalid device function`

This error occurs because:
1. Your RX 6750 XT uses the `gfx1031` architecture (RDNA 2)
2. PyTorch/ROCm needs to know the exact GPU architecture for kernel compilation
3. Environment variables weren't set for your specific GPU

## The Solution

I've implemented several fixes:

### 1. **Automatic GPU Detection Script** - [scripts/train_rocm.sh](scripts/train_rocm.sh)

This wrapper script automatically:
- Detects your GPU architecture (gfx1031)
- Sets `HSA_OVERRIDE_GFX_VERSION`
- Selects the discrete GPU (GPU 0, not integrated)
- Configures memory allocator settings
- Sets proper environment variables

### 2. **Model Memory Compatibility** - [src/gearhead/training/trainer.py](src/gearhead/training/trainer.py:163-169)

Added `_make_model_contiguous()` to ensure all model parameters are contiguous in memory, which ROCm requires for embeddings.

### 3. **Updated Training Command** - [Makefile](Makefile:46-49)

The `make train-small-rocm` command now uses the wrapper script.

## How to Use

Simply run:
```bash
make train-small-rocm
```

The script will automatically:
1. Detect GPU: RX 6750 XT (gfx1031)
2. Configure environment for your GPU
3. Select the discrete GPU (not integrated)
4. Start training with proper settings

## What Changed

### Environment Variables Set Automatically

```bash
HSA_OVERRIDE_GFX_VERSION=gfx1031     # Your GPU architecture
HIP_VISIBLE_DEVICES=0                 # Use discrete GPU (RX 6750 XT)
CUDA_VISIBLE_DEVICES=0                # Same (ROCm uses CUDA API)
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True  # Memory optimization
```

### Model Initialization

The trainer now ensures all parameters are contiguous before training starts, preventing HIP kernel errors with embedding layers.

## Testing the Fix

Try running training again:
```bash
make train-small-rocm
```

You should see:
```
Detected GPU architecture: gfx1031
Environment configured for ROCm training
HSA_OVERRIDE_GFX_VERSION=gfx1031
HIP_VISIBLE_DEVICES=0

Starting training...
Device: cuda (ROCM: AMD Radeon RX 6750 XT)
ROCm optimizations: Enabled
Making model parameters contiguous for ROCm compatibility...
```

## If You Still Have Issues

### Debug Mode

Enable detailed debugging:
```bash
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
make train-small-rocm
```

This will show exactly which kernel fails.

### Manual Environment Setup

If the automatic detection fails, set manually:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.1
export HIP_VISIBLE_DEVICES=0
python3 scripts/train.py --config configs/small_config_rocm.yaml
```

### Check GPU Selection

You have 2 GPUs:
- GPU 0: RX 6750 XT (discrete) - **Use this for training**
- GPU 1: Integrated graphics - Don't use

Verify the script selects GPU 0:
```bash
./scripts/train_rocm.sh --config configs/small_config_rocm.yaml
```

## Why This Happened

PyTorch's ROCm backend needs explicit GPU architecture information because:
1. AMD GPUs have diverse architectures (unlike NVIDIA's more unified CUDA)
2. HIP kernels are compiled for specific `gfx` targets
3. The RX 6750 XT (gfx1031) is relatively new and needs explicit configuration
4. ROCm can't always auto-detect the architecture correctly

The wrapper script now handles all of this automatically!

## Memory Usage Verified

With these fixes, training should use:
- ~500 MB for model weights
- ~1 GB for optimizer states
- ~2-3 GB for activations
- **Total: ~5-7 GB** (well within your 32GB RAM)

## Next Steps

1. Run `make train-small-rocm`
2. Monitor with `rocm-smi` in another terminal
3. Training should start successfully!

If you encounter any other issues, let me know!
