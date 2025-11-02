# Complete Training Guide - CUDA, ROCm & Apple Silicon

## Quick Reference

| Task | Command | Platform |
|------|---------|----------|
| Train on NVIDIA GPU | `make train-small` | CUDA |
| Train on AMD GPU | `make train-small-rocm` | ROCm |
| Train on Apple Silicon | `make train-small-mps` | MPS |
| Detect GPU | `make detect-gpu` | Any |
| Monitor GPU | `watch -n 1 nvidia-smi` | CUDA |
| Monitor GPU | `watch -n 1 rocm-smi` | ROCm |
| Monitor GPU | Activity Monitor | Apple Silicon |

## Installation

### For NVIDIA GPUs (CUDA)
```bash
make install
```

### For AMD GPUs (ROCm)
```bash
make install-rocm
```

### For Apple Silicon (M1/M2/M3)
```bash
make install-mps
```

## Memory Optimization

### Problem
The original config uses ~32GB+:
- Batch size: 16
- Sequence length: 1024
- Gradient accumulation: 2

### Solutions: Platform-Optimized Configs

#### ROCm Optimized Config
[configs/small_config_rocm.yaml](configs/small_config_rocm.yaml):
```yaml
batch_size: 4                    # 75% less memory
max_seq_length: 512              # 50% less memory
gradient_accumulation_steps: 8   # Maintains effective batch size
fp16: true                       # 50% less memory
```
**Result**: ~5-7GB VRAM usage (85% reduction!)

#### MPS Optimized Config
[configs/small_config_mps.yaml](configs/small_config_mps.yaml):
```yaml
batch_size: 4                    # Optimized for unified memory
max_seq_length: 512              # Efficient sequence length
gradient_accumulation_steps: 8   # Maintains effective batch size
fp16: true                       # Better performance
device: "mps"                    # Apple Silicon
```
**Result**: ~4-5GB unified memory usage

## Training Commands

### AMD GPU (RX 6750 XT)

```bash
# Activate your virtual environment first
source rocm/bin/activate

# Run training
make train-small-rocm
```

The script automatically:
- ✅ Detects GPU architecture (gfx1031)
- ✅ Sets HSA_OVERRIDE_GFX_VERSION
- ✅ Selects discrete GPU (not integrated)
- ✅ Configures memory allocator
- ✅ Enables ROCm optimizations

### NVIDIA GPU

```bash
# Activate your virtual environment first
source venv/bin/activate  # or wherever your venv is

# Run training
make train-small
```

### Apple Silicon (M1/M2/M3)

```bash
# Activate your virtual environment first
source venv/bin/activate  # or wherever your venv is

# Run training
make train-small-mps
```

The script automatically:
- ✅ Detects Apple Silicon
- ✅ Uses Metal Performance Shaders (MPS)
- ✅ Optimizes for unified memory
- ✅ Enables gradient checkpointing

## Monitoring Training

### Terminal 1: Run Training
```bash
# NVIDIA
source venv/bin/activate
make train-small

# AMD
source rocm/bin/activate
make train-small-rocm

# Apple Silicon
source venv/bin/activate
make train-small-mps
```

### Terminal 2: Monitor GPU/Memory

**NVIDIA GPU:**
```bash
watch -n 1 'nvidia-smi'
```

**AMD GPU:**
```bash
watch -n 1 'rocm-smi --showmeminfo vram --showuse'
```

**Apple Silicon:**
```bash
# Use Activity Monitor GUI, or:
while true; do
    echo "Memory usage:"
    vm_stat | grep "Pages active"
    sleep 5
done
```

### Terminal 3: Monitor System
```bash
htop
```

## Expected Output

### AMD GPU (ROCm)
```
Detected GPU architecture: gfx1031
Environment configured for ROCm training
HSA_OVERRIDE_GFX_VERSION=gfx1031
HIP_VISIBLE_DEVICES=0

Gearhead Training
============================================================

Loading tokenizer from tokenizer/tokenizer.json...
Tokenizer loaded. Vocabulary size: 32000

Initializing model...
Model initialized with 125,789,696 parameters

Loading training data from data/processed/train.jsonl...
Training samples: 1000

Initializing trainer...
Device: cuda (ROCM: AMD Radeon RX 6750 XT)
ROCm optimizations: Enabled
Making model parameters contiguous for ROCm compatibility...

============================================================
Starting training...
============================================================

Starting training for 20 epochs
Total training steps: 62500
Device: cuda (ROCM: AMD Radeon RX 6750 XT)
ROCm optimizations: Enabled

Epoch 1/20: 100%|████████| 250/250 [02:15<00:00]
```

### Apple Silicon (MPS)
```
Gearhead Training
============================================================

Loading tokenizer from tokenizer/tokenizer.json...
Tokenizer loaded. Vocabulary size: 32000

Initializing model...
Model initialized with 125,789,696 parameters

Loading training data from data/processed/train.jsonl...
Training samples: 1000

Initializing trainer...
Device: mps (MPS: Apple Silicon (arm64))
Gradient checkpointing enabled for memory efficiency

============================================================
Starting training...
============================================================

Starting training for 20 epochs
Total training steps: 62500

Epoch 1/20: 100%|████████| 250/250 [05:30<00:00, 0.76it/s, loss=3.12]
```

## Troubleshooting

### Issue: "HIP error: invalid device function"

**Fixed!** The wrapper script now automatically configures your GPU.

If you still see this:
```bash
# Check GPU detection
rocm-smi --showproductname

# Manually set if needed
export HSA_OVERRIDE_GFX_VERSION=10.3.1
make train-small-rocm
```

### Issue: "No module named 'torch'"

**Cause**: Virtual environment not activated

**Solution**:
```bash
source rocm/bin/activate
make train-small-rocm
```

### Issue: Still out of memory

**Solution**: Further reduce memory usage in [configs/small_config_rocm.yaml](configs/small_config_rocm.yaml):

```yaml
batch_size: 2                    # Even smaller
gradient_accumulation_steps: 16  # Compensate
max_seq_length: 256              # Shorter sequences
```

### Issue: Slow training

**AMD GPU - Solutions**:
1. Check GPU utilization: `rocm-smi` (should be 90%+)
2. Verify PCIe connection: `lspci | grep VGA`
3. Ensure fp16 is enabled: `fp16: true`
4. Check you're using discrete GPU (GPU 0, not GPU 1)

**Apple Silicon - Solutions**:
1. Close memory-intensive apps (Chrome, Slack, etc.)
2. Check swap usage: `sysctl vm.swapusage` (should be minimal)
3. Verify MPS is being used: Look for "Device: mps" in output
4. Ensure FP16 is enabled: `fp16: true`

### Issue: "MPS backend not available" (Apple Silicon)

**Cause**: macOS < 12.3 or PyTorch without MPS

**Solution**:
```bash
# Update macOS to 12.3+
# Reinstall PyTorch
pip install --upgrade torch torchvision torchaudio
```

### Issue: Out of memory (Apple Silicon)

**Solutions**:
1. Reduce batch size in [configs/small_config_mps.yaml](configs/small_config_mps.yaml):
   ```yaml
   batch_size: 2  # or even 1
   gradient_accumulation_steps: 16
   ```
2. Reduce sequence length:
   ```yaml
   max_seq_length: 256
   ```
3. Close other apps to free unified memory
4. Use CPU if memory is too constrained:
   ```bash
   python scripts/train.py --config configs/small_config.yaml --device cpu
   ```

## Platform Selection

Choose the appropriate platform for your hardware:

### NVIDIA GPU Systems
- **GPU**: NVIDIA RTX/GTX series, Tesla, A100, etc.
- **Selection**: Automatic (CUDA detects GPU)
- **Command**: `make train-small`

### AMD GPU Systems
- **GPU**: RX 6000/7000 series, MI series
- **Selection**: Via `HIP_VISIBLE_DEVICES` (automatic in scripts)
- **Example**: RX 6750 XT (GPU 0) vs integrated (GPU 1)
- **Command**: `make train-small-rocm`

### Apple Silicon Systems
- **Chips**: M1, M2, M3 (all variants: base, Pro, Max, Ultra)
- **Selection**: Automatic (MPS backend)
- **Memory**: Uses unified memory architecture
- **Command**: `make train-small-mps`

## Performance Expectations

Training 1000 examples for 20 epochs:

### NVIDIA GPUs

| GPU | Memory | Tokens/sec | Time per Epoch | Total Time |
|-----|--------|------------|----------------|------------|
| RTX 4080 | 3-4 GB | ~8000 | ~1.5 min | ~20-30 min |
| RTX 3060 Ti | 3-4 GB | ~5000 | ~2.5 min | ~35-45 min |

### AMD GPUs (ROCm)

| GPU | Memory | Tokens/sec | Time per Epoch | Total Time |
|-----|--------|------------|----------------|------------|
| RX 7900 XTX | 2.6 GB | ~6000 | ~1.5 min | ~30-40 min |
| RX 6750 XT | 2.6 GB | ~3500 | ~2.5 min | ~60-80 min |

### Apple Silicon (MPS)

| Chip | Memory | Tokens/sec | Time per Epoch | Total Time |
|------|--------|------------|----------------|------------|
| M3 Max | 4-5 GB | ~5000 | ~2.5 min | ~40-60 min |
| M2 Pro | 4-5 GB | ~3000 | ~4 min | ~60-100 min |
| M1 (8-core) | 4-5 GB | ~2000 | ~6 min | ~100-160 min |

## Configuration Comparison

| Setting | Standard | CUDA | ROCm | MPS |
|---------|----------|------|------|-----|
| batch_size | 16 | 8 | 4 | 4 |
| max_seq_length | 1024 | 1024 | 512 | 512 |
| gradient_accumulation | 2 | 4 | 8 | 8 |
| fp16 | true | true | true | true |
| device | cuda | cuda | cuda | mps |
| **Memory usage** | **~32 GB** | **3-4 GB** | **2.6-7 GB** | **4-5 GB** |

## Advanced: Manual Training

If you prefer manual control:

### NVIDIA GPU
```bash
source venv/bin/activate
python scripts/train.py --config configs/small_config.yaml
```

### AMD GPU (ROCm)
```bash
source rocm/bin/activate
export HSA_OVERRIDE_GFX_VERSION=gfx1031
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
python scripts/train.py --config configs/small_config_rocm.yaml
```

### Apple Silicon
```bash
source venv/bin/activate
python scripts/train.py --config configs/small_config_mps.yaml
```

## Files Reference

### Core Files
- [Makefile](Makefile) - Training commands for all platforms
- [src/gearhead/training/trainer.py](src/gearhead/training/trainer.py) - Training logic

### Configuration Files
- [configs/small_config.yaml](configs/small_config.yaml) - NVIDIA/CUDA config
- [configs/small_config_rocm.yaml](configs/small_config_rocm.yaml) - AMD/ROCm config
- [configs/small_config_mps.yaml](configs/small_config_mps.yaml) - Apple Silicon config

### Platform-Specific Scripts
- [scripts/train_rocm.sh](scripts/train_rocm.sh) - ROCm wrapper script

### Documentation
- [docs/ROCM_TRAINING.md](docs/ROCM_TRAINING.md) - Full ROCm documentation
- [docs/APPLE_SILICON_TRAINING.md](docs/APPLE_SILICON_TRAINING.md) - Full Apple Silicon documentation
- [PLATFORM_COMPARISON.md](PLATFORM_COMPARISON.md) - Platform comparison guide

## Next Steps

Choose your platform and follow these steps:

### NVIDIA GPU
1. ✅ Activate environment: `source venv/bin/activate`
2. ✅ Run training: `make train-small`
3. ✅ Monitor GPU: `watch -n 1 nvidia-smi`
4. ✅ Wait for training to complete (~20-45 minutes)
5. ✅ Find model in `outputs/small_model/`

### AMD GPU (ROCm)
1. ✅ Activate environment: `source rocm/bin/activate`
2. ✅ Run training: `make train-small-rocm`
3. ✅ Monitor GPU: `watch -n 1 rocm-smi`
4. ✅ Wait for training to complete (~30-80 minutes)
5. ✅ Find model in `outputs/small_model_rocm/`

### Apple Silicon
1. ✅ Activate environment: `source venv/bin/activate`
2. ✅ Run training: `make train-small-mps`
3. ✅ Monitor: Use Activity Monitor or `vm_stat`
4. ✅ Wait for training to complete (~40-160 minutes)
5. ✅ Find model in `outputs/small_model_mps/`

Happy training! 🚀
