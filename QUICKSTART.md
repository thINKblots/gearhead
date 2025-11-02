# Gearhead Quick Start Guide

Complete workflow from setup to using your trained model.

## Step-by-Step Workflow

### Step 1: Choose Your Platform

**NVIDIA GPU:**
```bash
make install
```

**AMD GPU (ROCm):**
```bash
make install-rocm
```

**Apple Silicon (M1/M2/M3):**
```bash
make install-mps
```

### Step 2: Prepare Data ⏳

```bash
make prepare-data
```

Creates training data and tokenizer (~2 minutes)

### Step 3: Train Model 🚂

**NVIDIA GPU (20-45 min):**
```bash
make train-small
```

**AMD GPU (30-80 min):**
```bash
make train-small-rocm
```

**Apple Silicon (40-160 min):**
```bash
make train-small-mps
```

### Step 4: Use Model 🎯

```bash
make infer  # Interactive mode
```

## Quick Command Reference

| Platform | Install | Train | Time |
|----------|---------|-------|------|
| **NVIDIA** | `make install` | `make train-small` | 20-45 min |
| **AMD** | `make install-rocm` | `make train-small-rocm` | 30-80 min |
| **Apple** | `make install-mps` | `make train-small-mps` | 40-160 min |

## Full Guide

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete documentation and [PLATFORM_COMPARISON.md](PLATFORM_COMPARISON.md) to choose the best platform for your hardware.
