# Inference Quick Start

## After Training Completes

Your model is at: `outputs/small_model_rocm/final_model/`

## Three Ways to Use It

### 1. Interactive Mode 🎯 Easiest
```bash
python scripts/inference.py \
  --model outputs/small_model_rocm/final_model \
  --tokenizer tokenizer/tokenizer.json
```

Then type your scenarios!

### 2. Command Line
```bash
python scripts/inference.py \
  --model outputs/small_model_rocm/final_model \
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
    model_path="outputs/small_model_rocm/final_model",
    tokenizer=tokenizer
)

result = engine.diagnose(
    equipment="Caterpillar 320",
    symptom="Engine loses power",
    error_codes=["P0087"]
)

print(result['probable_cause'])
print(result['solution'])
```

## GPU vs CPU

**GPU** (faster, ~100-200 tokens/sec):
```python
engine = DiagnosticEngine(..., device="cuda")
```

**CPU** (slower, ~10-30 tokens/sec, more compatible):
```python
engine = DiagnosticEngine(..., device="cpu")
```

## Full Guide

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete documentation.
