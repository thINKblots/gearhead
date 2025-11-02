# Inference Quick Start

## After Training Completes

Your model location depends on which platform you trained on:
- **NVIDIA**: `outputs/small_model/final_model/`
- **AMD**: `outputs/small_model_rocm/final_model/`
- **Apple Silicon**: `outputs/small_model_mps/final_model/`

## Three Ways to Use It

### 1. Interactive Mode 🎯 Easiest
```bash
# Replace with your model path
python scripts/inference.py \
  --model outputs/small_model_mps/final_model \
  --tokenizer tokenizer/tokenizer.json
```

Then type your scenarios!

### 2. Command Line
```bash
python scripts/inference.py \
  --model outputs/small_model_mps/final_model \
  --tokenizer tokenizer/tokenizer.json \
  --equipment "Caterpillar 320" \
  --symptom "Engine loses power" \
  --error-codes P0087
```

### 3. Python Code
```python
from gearhead.data import GearheadTokenizer
from gearhead.inference import DiagnosticEngine

tokenizer = GearheadTokenizer("tokenizer/tokenizer.json")
engine = DiagnosticEngine(
    model_path="outputs/small_model_mps/final_model",
    tokenizer=tokenizer,
    device="mps"  # Use "cuda" for NVIDIA/AMD, "mps" for Apple, "cpu" for CPU
)

result = engine.diagnose(
    equipment="Caterpillar 320",
    symptom="Engine loses power",
    error_codes=["P0087"]
)

print(result['probable_cause'])
print(result['solution'])
```

## Device Selection

Choose the appropriate device for your hardware:

**NVIDIA GPU** (~150-300 tokens/sec):
```python
engine = DiagnosticEngine(..., device="cuda")
```

**AMD GPU** (~100-200 tokens/sec):
```python
engine = DiagnosticEngine(..., device="cuda")
```

**Apple Silicon** (~100-200 tokens/sec):
```python
engine = DiagnosticEngine(..., device="mps")
```

**CPU** (~10-30 tokens/sec, most compatible):
```python
engine = DiagnosticEngine(..., device="cpu")
```

## Full Guide

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete documentation.
