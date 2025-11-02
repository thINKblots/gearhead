# How to Use Your Gearhead Model

Complete guide for using your trained Gearhead diagnostic model.

## Quick Start

After training completes, your model is saved in one of these locations depending on your platform:
- **NVIDIA**: `outputs/small_model/final_model/`
- **AMD**: `outputs/small_model_rocm/final_model/`
- **Apple Silicon**: `outputs/small_model_mps/final_model/`

### 1. Interactive Mode (Easiest)

```bash
# Replace with your model path
python scripts/inference.py \
  --model outputs/small_model_mps/final_model \
  --tokenizer tokenizer/tokenizer.json
```

Then enter diagnostic scenarios interactively:
```
Equipment type/model: Caterpillar 320 Excavator
Symptom/problem: Engine starts but loses power under load
Error codes (comma separated, or press Enter): P0087, SPN 157

Analyzing...

DIAGNOSTIC REPORT
======================================================================
Equipment: Caterpillar 320 Excavator
Symptom: Engine starts but loses power under load
Error Codes: P0087, SPN 157

Probable Cause:
Low fuel pressure due to contaminated fuel filter or air in fuel system

Recommended Solution:
Replace fuel filter and check fuel lines for air leaks. Bleed fuel system.
======================================================================
```

### 2. Single Diagnosis (Command Line)

```bash
python scripts/inference.py \
  --model outputs/small_model_rocm/final_model \
  --tokenizer tokenizer/tokenizer.json \
  --equipment "Caterpillar 320 Excavator" \
  --symptom "Engine loses power under load" \
  --error-codes P0087 SPN157 \
  --output result.json
```

### 3. Batch Processing (Multiple Scenarios)

Create a scenarios file `my_cases.json`:
```json
[
  {
    "equipment": "Caterpillar 320 Excavator",
    "symptom": "Engine starts but loses power under load",
    "error_codes": ["P0087", "SPN 157"]
  },
  {
    "equipment": "John Deere 410 Backhoe",
    "symptom": "Hydraulic system slow response",
    "error_codes": ["HY001"]
  }
]
```

Run batch processing:
```bash
python scripts/inference.py \
  --model outputs/small_model_rocm/final_model \
  --tokenizer tokenizer/tokenizer.json \
  --batch-file my_cases.json \
  --output results.json
```

## Python API Usage

### Basic Example

```python
import sys
sys.path.insert(0, 'src')

from gearhead.data import GearheadTokenizer
from gearhead.inference import DiagnosticEngine

# Load the model
tokenizer = GearheadTokenizer("tokenizer/tokenizer.json")
engine = DiagnosticEngine(
    model_path="outputs/small_model_rocm/final_model",
    tokenizer=tokenizer
)

# Run diagnosis
result = engine.diagnose(
    equipment="Caterpillar 320 Excavator",
    symptom="Engine loses power under load",
    error_codes=["P0087", "SPN 157"]
)

print(f"Cause: {result['probable_cause']}")
print(f"Solution: {result['solution']}")
```

### Advanced Generation Parameters

```python
result = engine.diagnose(
    equipment="Caterpillar 320",
    symptom="Engine loses power",
    error_codes=["P0087"],
    max_length=500,       # Maximum tokens to generate
    temperature=0.7,      # Lower = more focused, higher = more creative
    top_k=50,            # Top-k sampling
    top_p=0.9            # Nucleus sampling threshold
)
```

**Temperature guidelines**:
- `0.3-0.5`: Conservative, factual diagnostics
- `0.7-0.8`: Balanced (recommended)
- `0.9-1.0`: More exploratory suggestions

### Batch Processing in Python

```python
scenarios = [
    {
        "equipment": "Caterpillar 320",
        "symptom": "Engine loses power",
        "error_codes": ["P0087"]
    },
    {
        "equipment": "John Deere 410",
        "symptom": "Hydraulic slow",
        "error_codes": ["HY001"]
    }
]

results = engine.batch_diagnose(
    scenarios=scenarios,
    max_length=500,
    temperature=0.7
)

for i, result in enumerate(results):
    print(f"\nCase {i+1}:")
    print(f"  Cause: {result['probable_cause']}")
    print(f"  Solution: {result['solution']}")
```

## Integration Examples

### Web API Integration

```python
from flask import Flask, request, jsonify
from gearhead.data import GearheadTokenizer
from gearhead.inference import DiagnosticEngine

app = Flask(__name__)

# Load model at startup
tokenizer = GearheadTokenizer("tokenizer/tokenizer.json")
engine = DiagnosticEngine(
    model_path="outputs/small_model_rocm/final_model",
    tokenizer=tokenizer
)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    data = request.json

    result = engine.diagnose(
        equipment=data['equipment'],
        symptom=data['symptom'],
        error_codes=data.get('error_codes', [])
    )

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Test the API:
```bash
curl -X POST http://localhost:5000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "equipment": "Caterpillar 320",
    "symptom": "Engine loses power",
    "error_codes": ["P0087"]
  }'
```

### Command Line Tool

Create a simple CLI tool `diagnose.sh`:
```bash
#!/bin/bash
python scripts/inference.py \
  --model outputs/small_model_rocm/final_model \
  --tokenizer tokenizer/tokenizer.json \
  --equipment "$1" \
  --symptom "$2" \
  --error-codes "${@:3}"
```

Usage:
```bash
chmod +x diagnose.sh
./diagnose.sh "Caterpillar 320" "Engine loses power" P0087 SPN157
```

## Model Location

After training with `make train-small-rocm`, your model is saved at:

```
outputs/small_model_rocm/
├── checkpoint-500/      # Intermediate checkpoints
├── checkpoint-1000/
├── checkpoint-1500/
└── final_model/        # Final trained model ← Use this!
    ├── config.pt       # Model configuration
    ├── model.pt        # Model weights
    └── trainer_state.pt # Training state
```

## Inference Device Selection

### NVIDIA GPU (Fastest)
```python
engine = DiagnosticEngine(
    model_path="outputs/small_model/final_model",
    tokenizer=tokenizer,
    device="cuda"  # Uses NVIDIA GPU
)
```

### AMD GPU (ROCm)
```python
engine = DiagnosticEngine(
    model_path="outputs/small_model_rocm/final_model",
    tokenizer=tokenizer,
    device="cuda"  # ROCm also uses "cuda" device string
)
```

### Apple Silicon (MPS)
```python
engine = DiagnosticEngine(
    model_path="outputs/small_model_mps/final_model",
    tokenizer=tokenizer,
    device="mps"  # Uses Metal Performance Shaders
)
```

### CPU (Most Compatible)
```python
engine = DiagnosticEngine(
    model_path="outputs/small_model/final_model",
    tokenizer=tokenizer,
    device="cpu"  # Uses CPU
)
```

**Performance**:
- NVIDIA GPU: ~150-300 tokens/sec
- AMD GPU: ~100-200 tokens/sec
- Apple Silicon: ~100-200 tokens/sec
- CPU: ~10-30 tokens/sec

## Example Scenarios

### 1. Hydraulic Issues
```python
result = engine.diagnose(
    equipment="John Deere 410 Backhoe",
    symptom="Hydraulic system responds slowly, especially when warm",
    error_codes=["HY001", "HY025"]
)
```

### 2. Electrical Problems
```python
result = engine.diagnose(
    equipment="Komatsu PC200 Excavator",
    symptom="Intermittent electrical failures, lights flickering",
    error_codes=["E101"]
)
```

### 3. Engine Diagnostics
```python
result = engine.diagnose(
    equipment="Caterpillar 320 Excavator",
    symptom="Engine hard to start in cold weather, white smoke on startup",
    error_codes=["P0380", "P0382"]
)
```

## Output Format

The `diagnose()` method returns a dictionary:

```python
{
    "equipment": "Caterpillar 320 Excavator",
    "symptom": "Engine loses power under load",
    "error_codes": ["P0087", "SPN 157"],
    "probable_cause": "Low fuel pressure due to contaminated fuel filter",
    "solution": "Replace fuel filter and inspect fuel lines for blockages"
}
```

## Tips for Best Results

### 1. Be Specific
❌ Bad: "broken"
✅ Good: "Engine loses power under load after 30 minutes of operation"

### 2. Include Error Codes
Error codes significantly improve diagnostic accuracy:
```python
# Without error codes - less accurate
result = engine.diagnose(
    equipment="Caterpillar 320",
    symptom="Engine problem"
)

# With error codes - more accurate
result = engine.diagnose(
    equipment="Caterpillar 320",
    symptom="Engine problem",
    error_codes=["P0087", "SPN 157"]
)
```

### 3. Use Standard Equipment Names
The model works best with equipment names it was trained on. Use manufacturer + model:
- ✅ "Caterpillar 320 Excavator"
- ✅ "John Deere 410 Backhoe"
- ✅ "Komatsu PC200"

### 4. Adjust Temperature for Use Case
- **Critical repairs** (safety-critical): temperature=0.3
- **General diagnostics**: temperature=0.7 (default)
- **Brainstorming possibilities**: temperature=0.9

## Troubleshooting

### Model Not Found
```
Error: Model not found at outputs/small_model_rocm/final_model
```

**Solution**: Make sure training completed successfully. Check:
```bash
ls -la outputs/small_model_rocm/final_model/
```

### Tokenizer Not Found
```
Error: Tokenizer not found at tokenizer/tokenizer.json
```

**Solution**: Train the tokenizer first:
```bash
make prepare-data
```

### Out of Memory During Inference
```
RuntimeError: HIP out of memory
```

**Solution**: Use CPU instead:
```python
engine = DiagnosticEngine(..., device="cpu")
```

Or reduce max_length:
```python
result = engine.diagnose(..., max_length=200)  # Shorter responses
```

## Next Steps

1. **Test the model**: Try the interactive mode
2. **Fine-tune**: Train on your specific equipment data
3. **Deploy**: Integrate into your application
4. **Evaluate**: Test accuracy on real diagnostic scenarios

## Need Help?

- Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training info
- Review [scripts/inference.py](scripts/inference.py) for implementation details
- See [data/examples/sample_scenario.json](data/examples/sample_scenario.json) for example data format

Happy diagnosing! 🔧
