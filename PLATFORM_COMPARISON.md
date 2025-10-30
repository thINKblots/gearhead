# Training Platform Comparison

Choose the best option for your hardware.

## Three Training Options

### 1. NVIDIA GPUs (CUDA) - Fastest
**Best for**: Production training, large models, fastest performance

| Aspect | Details |
|--------|---------|
| **Hardware** | RTX 3060+, RTX 4000 series, A100, etc. |
| **Speed** | ⭐⭐⭐⭐⭐ (2-3x faster than others) |
| **Setup** | ⭐⭐⭐ (CUDA drivers needed) |
| **Cost** | $$$ (GPUs expensive) |
| **Memory** | Dedicated VRAM (8GB+) |

**Install**:
```bash
make install
```

**Train**:
```bash
make train-small
```

---

### 2. AMD GPUs (ROCm) - Good Alternative
**Best for**: AMD GPU owners, cost-effective training

| Aspect | Details |
|--------|---------|
| **Hardware** | RX 6000/7000 series, MI series |
| **Speed** | ⭐⭐⭐⭐ (1.5-2x faster than MPS) |
| **Setup** | ⭐⭐ (ROCm installation required) |
| **Cost** | $$ (Cheaper than NVIDIA) |
| **Memory** | Dedicated VRAM (8GB+) |

**Install**:
```bash
make install-rocm
```

**Train**:
```bash
make train-small-rocm
```

**Tested on**: RX 6750 XT (12GB) - Uses only 2.6GB peak

---

### 3. Apple Silicon (MPS) - Easiest Setup
**Best for**: Mac users, development, small-medium models

| Aspect | Details |
|--------|---------|
| **Hardware** | M1, M2, M3 (all variants) |
| **Speed** | ⭐⭐⭐ (Good for small models) |
| **Setup** | ⭐⭐⭐⭐⭐ (Easiest - just install PyTorch) |
| **Cost** | $ (If you already have a Mac) |
| **Memory** | Unified memory (8GB+) |

**Install**:
```bash
make install-mps
```

**Train**:
```bash
make train-small-mps
```

**Works on**: MacBook Air, MacBook Pro, Mac mini, Mac Studio, iMac

---

## Performance Comparison

Training 1000 examples for 20 epochs:

| Platform | Hardware Example | Time | Speed | Memory |
|----------|-----------------|------|-------|--------|
| **NVIDIA (CUDA)** | RTX 4080 | ~20-30 min | ~8000 tok/s | 3-4 GB VRAM |
| **NVIDIA (CUDA)** | RTX 3060 Ti | ~35-45 min | ~5000 tok/s | 3-4 GB VRAM |
| **AMD (ROCm)** | RX 6750 XT | ~60-80 min | ~3500 tok/s | 2.6 GB VRAM |
| **AMD (ROCm)** | RX 7900 XTX | ~30-40 min | ~6000 tok/s | 2.6 GB VRAM |
| **Apple (MPS)** | M1 (8-core) | ~100-160 min | ~2000 tok/s | 4-5 GB unified |
| **Apple (MPS)** | M2 Pro | ~60-100 min | ~3000 tok/s | 4-5 GB unified |
| **Apple (MPS)** | M3 Max | ~40-60 min | ~5000 tok/s | 4-5 GB unified |
| **CPU** | Any | ~8-12 hours | ~200 tok/s | 8 GB RAM |

## Memory Requirements

### Minimum Configuration

| Platform | Min VRAM/RAM | Batch Size | Seq Length |
|----------|--------------|------------|------------|
| NVIDIA | 8 GB | 4 | 512 |
| AMD | 8 GB | 4 | 512 |
| Apple Silicon | 8 GB unified | 2 | 256 |
| CPU | 16 GB RAM | 2 | 256 |

### Recommended Configuration

| Platform | Recommended | Batch Size | Seq Length |
|----------|-------------|------------|------------|
| NVIDIA | 12 GB+ | 8 | 1024 |
| AMD | 12 GB+ | 4 | 512 |
| Apple Silicon | 16 GB+ unified | 4 | 512 |
| CPU | 32 GB+ RAM | 4 | 512 |

## Decision Guide

### Choose NVIDIA (CUDA) if:
- ✅ You have an NVIDIA GPU
- ✅ You want fastest training
- ✅ Training large models (350M+ params)
- ✅ Budget allows for GPU purchase

### Choose AMD (ROCm) if:
- ✅ You have an AMD GPU (RX 6000/7000)
- ✅ Good price/performance balance
- ✅ Linux user
- ✅ Want to support open-source GPU stack

### Choose Apple Silicon (MPS) if:
- ✅ You have a Mac (M1/M2/M3)
- ✅ Easiest setup is priority
- ✅ Training small-medium models
- ✅ Low noise/power consumption important
- ✅ Development and testing

### Choose CPU if:
- ✅ No GPU available
- ✅ Very small models only
- ✅ Time is not critical
- ❌ Not recommended for production

## Platform-Specific Features

### NVIDIA (CUDA)
- ✅ Best ecosystem support
- ✅ Mature tooling
- ✅ Fastest performance
- ✅ Best for large models
- ❌ Expensive hardware
- ❌ Driver setup can be tricky

### AMD (ROCm)
- ✅ Good price/performance
- ✅ Open-source friendly
- ✅ Growing ecosystem
- ✅ Optimized for this project
- ❌ Some compatibility issues (now fixed!)
- ❌ Smaller community than NVIDIA

### Apple Silicon (MPS)
- ✅ Zero setup (just works)
- ✅ Unified memory advantage
- ✅ Silent operation
- ✅ Low power consumption
- ✅ Great for mobile Macs
- ❌ Slower than dedicated GPUs
- ❌ macOS only

## Quick Start by Platform

### I have an NVIDIA GPU
```bash
make install
make generate-data
make train-small
```

### I have an AMD GPU
```bash
make install-rocm
make generate-data
make train-small-rocm
```

### I have a Mac (M1/M2/M3)
```bash
make install-mps
make generate-data
make train-small-mps
```

## Can I Use Multiple Platforms?

Yes! The same codebase works on all platforms:

```bash
# Train on AMD GPU
make train-small-rocm

# Resume on NVIDIA GPU
python scripts/train.py --config configs/small_config.yaml \
    --resume outputs/small_model_rocm/checkpoint-1000

# Inference on Mac
python scripts/inference.py --model outputs/final_model \
    --tokenizer tokenizer/tokenizer.json --device mps
```

## Documentation by Platform

- **NVIDIA**: Standard PyTorch documentation applies
- **AMD**: [docs/ROCM_TRAINING.md](docs/ROCM_TRAINING.md)
- **Apple**: [docs/APPLE_SILICON_TRAINING.md](docs/APPLE_SILICON_TRAINING.md)

## Summary

**Best overall**: NVIDIA (if budget allows)
**Best value**: AMD ROCm (RX 6000/7000 series)
**Easiest setup**: Apple Silicon (M1/M2/M3)

All three platforms work great with this project! Choose based on what you have or can afford.
