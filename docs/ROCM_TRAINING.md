# Training with ROCm (AMD GPUs)

This guide explains how to train Gearhead models on AMD GPUs using ROCm.

## Prerequisites

### 1. Install ROCm

First, install ROCm on your system. Follow the official AMD ROCm installation guide:
- https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html

Verify ROCm installation:
```bash
rocm-smi
```

You should see your AMD GPU listed with temperature, usage, and memory information.

### 2. Check Supported GPUs

ROCm supports:
- **RDNA 3**: RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT, RX 7600
- **RDNA 2**: RX 6950 XT, RX 6900 XT, RX 6800 XT, RX 6800, RX 6700 XT
- **CDNA 2**: MI250X, MI250, MI210
- **CDNA**: MI100
- And more...

## Installation

### Option 1: Using Make (Recommended)

```bash
# Install ROCm dependencies
make install-rocm
```

This will:
1. Install PyTorch with ROCm 6.0 support
2. Install all other dependencies
3. Install the gearhead package in development mode

### Option 2: Manual Installation

```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install other dependencies
pip install -r requirements-rocm.txt

# Install gearhead
pip install -e .
```

## Verify Installation

Check that PyTorch can see your AMD GPU:

```bash
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Or use the built-in detection:

```bash
make detect-gpu
```

## Training

### Quick Start

Train the small model with ROCm optimizations:

```bash
make train-small-rocm
```

This uses the memory-optimized configuration in `configs/small_config_rocm.yaml`.

### Manual Training

```bash
python scripts/train.py --config configs/small_config_rocm.yaml
```

## ROCm-Specific Optimizations

The ROCm configuration includes several memory and performance optimizations:

### 1. Reduced Memory Footprint

```yaml
batch_size: 4                    # Smaller batches
gradient_accumulation_steps: 8   # Maintain effective batch size
max_seq_length: 512              # Shorter sequences
```

### 2. Mixed Precision Training

```yaml
fp16: true  # Uses AMD's mixed precision acceleration
```

### 3. ROCm-Specific Features

The trainer automatically detects ROCm and applies optimizations:
- TF32 tensor core acceleration (on CDNA2+ GPUs)
- Memory allocator tuning for HIP
- Efficient gradient accumulation

## Memory Usage

With the ROCm-optimized config on 32GB RAM:

| Configuration | Approximate Memory Usage |
|--------------|-------------------------|
| Model weights | ~500 MB (125M params) |
| Optimizer states | ~1 GB |
| Gradients | ~500 MB |
| Activations (batch=4, seq=512) | ~2-3 GB |
| Dataset | ~1-2 GB (depends on size) |
| **Total** | **~5-7 GB** |

This leaves plenty of headroom on a 32GB system.

## Troubleshooting

### Issue: "HIP error: invalid device function"

This error occurs when the GPU architecture isn't properly configured.

**Solution**: The training script now automatically detects and configures your GPU. If you still see this error:

1. Check your GPU architecture:
   ```bash
   rocm-smi --showproductname
   ```

2. The training script automatically sets `HSA_OVERRIDE_GFX_VERSION` for your GPU
3. If issues persist, manually set it:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.1  # For RX 6750 XT (gfx1031)
   make train-small-rocm
   ```

### Issue: "ROCm not detected"

**Solution**: Make sure you installed PyTorch with ROCm support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### Issue: "Out of memory" errors

**Solutions**:
1. Reduce batch size further:
   ```yaml
   batch_size: 2
   gradient_accumulation_steps: 16
   ```

2. Reduce sequence length:
   ```yaml
   max_seq_length: 256
   ```

3. Monitor GPU memory:
   ```bash
   watch -n 1 rocm-smi
   ```

### Issue: Slow training performance

**Solutions**:
1. Ensure ROCm optimizations are enabled:
   ```yaml
   rocm_optimize: true
   ```

2. Check GPU utilization with `rocm-smi`
3. Verify PCIe bandwidth (should be Gen4 x16 for best performance)
4. Make sure you're using fp16:
   ```yaml
   fp16: true
   ```

## Performance Tips

### 1. PCIe Configuration

For best performance, ensure your GPU is in a PCIe 4.0 x16 slot:
```bash
lspci -vv | grep -A 10 VGA
```

### 2. CPU Pinning

For multi-GPU setups, pin processes to NUMA nodes:
```bash
numactl --cpunodebind=0 python scripts/train.py --config configs/small_config_rocm.yaml
```

### 3. Monitor Training

Use `rocm-smi` to monitor:
```bash
# In another terminal
watch -n 1 'rocm-smi --showmeminfo vram --showuse'
```

### 4. Logging

Enable W&B logging to track metrics:
```yaml
wandb: true
wandb_project: "gearhead"
wandb_run_name: "small-model-rocm-run1"
```

## Configuration Files

- **Standard**: `configs/small_config.yaml` - Original configuration
- **ROCm optimized**: `configs/small_config_rocm.yaml` - Memory-optimized for 32GB systems

## Comparing CUDA vs ROCm

The same code works for both NVIDIA (CUDA) and AMD (ROCm) GPUs. The trainer automatically detects your hardware and applies appropriate optimizations.

To train on NVIDIA GPUs:
```bash
make train-small
```

To train on AMD GPUs:
```bash
make train-small-rocm
```

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [AMD GPU Tuning Guide](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Performance-tuning)
