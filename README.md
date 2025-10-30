# Gearhead - Equipment Diagnostic Language Model

Small language model optimized for heavy equipment diagnostics. **Supports NVIDIA, AMD, and Apple Silicon GPUs.**

## 🚀 Three Training Options

| Platform | Command | Best For |
|----------|---------|----------|
| **NVIDIA GPUs** | `make train-small` | Fastest (RTX 3060+, A100, etc.) |
| **AMD GPUs** | `make train-small-rocm` | Great value (RX 6000/7000 series) |
| **Apple Silicon** | `make train-small-mps` | Easiest setup (M1/M2/M3) |

See [PLATFORM_COMPARISON.md](PLATFORM_COMPARISON.md) for detailed comparison.

## Quick Start

### 1. Choose Your Platform

**NVIDIA GPU**:
```bash
make install
```

**AMD GPU** (RX 6000/7000):
```bash
make install-rocm
```

**Apple Silicon** (M1/M2/M3):
```bash
make install-mps
```

### 2. Generate Sample Data

```bash
make generate-data
```

⚠️ Creates 1000 synthetic examples for testing. For production, you need real diagnostic data - see [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)

### 3. Train Model

**NVIDIA**: `make train-small`
**AMD**: `make train-small-rocm`
**Apple**: `make train-small-mps`

### 4. Use Model

```bash
make infer  # Interactive mode
```

## What's Included

✅ Multi-platform support (NVIDIA, AMD, Apple)
✅ Memory optimizations (2.6GB vs 32GB+ originally)
✅ Gradient checkpointing
✅ ROCm fixes for AMD GPUs
✅ MPS support for Apple Silicon
✅ Automated training pipeline
✅ Interactive inference mode

## ⚠️ Important: Data Requirements

The sample data is **synthetic and minimal** - good for testing, not production.

For a useful model, you need **1,000+ real diagnostic scenarios**.

**See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for details**

## Documentation

### Getting Started
- **[PLATFORM_COMPARISON.md](PLATFORM_COMPARISON.md)** - Choose your platform ⭐
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)** - How to get real data ⭐
- **[PDF_TRAINING_GUIDE.md](PDF_TRAINING_GUIDE.md)** - Extract training data from PDF manuals ⭐

### Platform-Specific
- **[docs/ROCM_TRAINING.md](docs/ROCM_TRAINING.md)** - AMD GPU guide
- **[docs/APPLE_SILICON_TRAINING.md](docs/APPLE_SILICON_TRAINING.md)** - Apple Silicon guide

### Training & Usage
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training guide
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - How to use trained models

## Performance

| Platform | Example Hardware | Training Time* | Memory |
|----------|------------------|----------------|--------|
| NVIDIA | RTX 4080 | ~20-30 min | 3-4 GB VRAM |
| AMD | RX 6750 XT | ~60-80 min | 2.6 GB VRAM |
| Apple | M3 Max | ~40-60 min | 4-5 GB unified |

*1000 examples, 20 epochs

## Tested Hardware

- ✅ AMD RX 6750 XT (12GB) - ROCm 6.0
- ✅ NVIDIA GPUs with CUDA 11.8+
- ✅ Apple M1/M2/M3 (all variants)

## Make Commands

```bash
make install-mps       # Install for Apple Silicon
make install-rocm      # Install for AMD GPU
make generate-data     # Generate 1000 sample examples
make train-small-mps   # Train on Apple Silicon
make train-small-rocm  # Train on AMD GPU
make infer             # Run inference
make detect-gpu        # Detect GPU type
```

## Usage Example

```python
from gearhead.data import GearheadTokenizer
from gearhead.inference import DiagnosticEngine

tokenizer = GearheadTokenizer("tokenizer/tokenizer.json")
engine = DiagnosticEngine(
    model_path="outputs/small_model_mps/final_model",
    tokenizer=tokenizer
)

result = engine.diagnose(
    equipment="Caterpillar 320 Excavator",
    symptom="Engine loses power under load",
    error_codes=["P0087", "SPN 157"]
)

print(result['probable_cause'])
print(result['solution'])
```

## License

[Add your license]

## Contributing

Issues and PRs welcome!
