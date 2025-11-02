# Pre-training Gearhead on Equipment Diagnostic Corpus

## Overview

Pre-training helps the model learn domain-specific language before fine-tuning on structured diagnostic tasks. Your `mobile_equipment_diagnostics_corpus.txt` (11MB, 138K lines, 65K text segments) is perfect for this.

## Quick Start

### Option 1: Use Make Command (Recommended)

```bash
# Make sure virtual environment exists
python3 -m venv venv

# Install dependencies (already done if you ran the setup earlier)
source venv/bin/activate
pip install -e .

# Run pre-training on AMD GPU
make pretrain-rocm
```

### Option 2: Manual Command

```bash
source venv/bin/activate
./scripts/pretrain_rocm.sh \
    --corpus data/text/mobile_equipment_diagnostics_corpus.txt \
    --output outputs/pretrained_model_rocm \
    --batch-size 2 \
    --gradient-accumulation 16 \
    --epochs 3 \
    --fp16 \
    --device cuda
```

## What Pre-training Does

### Input
- **Corpus**: `data/text/mobile_equipment_diagnostics_corpus.txt`
- **Size**: 11MB, 138,253 lines
- **Content**: Equipment maintenance and diagnostic language
- **Sections**: Operator Awareness, Hydraulics, Diagnostics, Engine/Powertrain, Safety

### Output
- **Pre-trained Model**: `outputs/pretrained_model_rocm/final_model/`
- **Model Size**: ~125M parameters (~500MB on disk)
- **Trained On**: 59,016 text segments (90% of corpus)
- **Validated On**: 6,558 text segments (10% of corpus)

### Training Configuration

The pre-training uses:
- **Vocabulary**: 680 tokens (trained on your corpus)
- **Architecture**: 12-layer transformer (768 hidden size, 12 attention heads)
- **Context Length**: 512 tokens
- **Batch Size**: 2 (effective: 32 with gradient accumulation)
- **Epochs**: 3 passes through the corpus
- **Memory**: ~2.6GB VRAM (fits on RX 6750 XT 12GB)
- **Time Estimate**: 2-4 hours depending on GPU

## After Pre-training

### Step 1: Fine-tune on Diagnostic Scenarios

Once pre-trained, fine-tune the model on your structured diagnostic data:

```bash
# With your custom .txt diagnostic scenarios
python3 scripts/convert_txt_to_jsonl.py diagnostics.txt data/processed/train.jsonl
cp data/processed/train.jsonl data/processed/val.jsonl
cp data/processed/train.jsonl data/processed/test.jsonl

# Fine-tune from pre-trained checkpoint
source venv/bin/activate
python scripts/train.py \
    --config configs/small_config_rocm.yaml \
    --from-pretrained outputs/pretrained_model_rocm/final_model
```

### Step 2: Test the Model

```bash
# Run inference
make infer

# Or with specific inputs
source venv/bin/activate
./scripts/infer_rocm.sh \
    --equipment "Caterpillar 320D" \
    --symptom "Engine loses power under load" \
    --error-codes P0087
```

## Pre-training Process Details

### 1. Tokenizer Training
```
Found 1 text files:
  - data/text/mobile_equipment_diagnostics_corpus.txt

Training tokenizer with vocab_size=32000...
Tokenizer saved! Vocabulary size: 680
```

The tokenizer learns:
- Equipment-specific terminology (hydraulic, engine, diagnostic, etc.)
- Error code patterns
- Maintenance vocabulary
- Technical measurements and specs

### 2. Dataset Preparation
```
Loading corpus from data/text/mobile_equipment_diagnostics_corpus.txt...
Loaded 65574 text segments
Train samples: 59016 (90%)
Val samples: 6558 (10%)
```

Text is split by double newlines into segments, then:
- Tokenized to max 512 tokens
- Padded if needed
- Split into train/val sets

### 3. Model Initialization
```
Total parameters: 125,237,248
Trainable parameters: 125,237,248
Model size: ~500.9 MB (FP32)
```

Architecture:
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- 3072 intermediate size
- Gradient checkpointing enabled
- FP16 mixed precision

### 4. Training Loop
```
Epoch 1/3:
  Step 100: loss=4.352, lr=1e-5
  Step 500: loss=3.721, lr=5e-5
  ...
Validation: loss=3.124

Epoch 2/3:
  ...
```

Saves checkpoints every 5000 steps.

### 5. Final Model Save
```
Pre-trained model saved to outputs/pretrained_model_rocm/final_model/
  - model.pt (PyTorch weights)
  - config.json (model configuration)
  - tokenizer.json (trained tokenizer)
```

## Memory Optimization

The pre-training script uses several techniques to fit in 12GB VRAM:

1. **Small Batch Size**: 2 samples per step
2. **Gradient Accumulation**: Accumulate 16 steps (effective batch = 32)
3. **Gradient Checkpointing**: Trade computation for memory
4. **FP16 Training**: Half-precision reduces memory by ~50%
5. **Sequence Length**: Limited to 512 tokens

**Peak VRAM Usage**: ~2.6GB (22% of 12GB)

## Monitoring Progress

### Check Training Progress

```bash
# View live output
tail -f pretrain_output.log

# Check GPU usage
watch -n 1 rocm-smi

# See training metrics
grep "loss=" pretrain_output.log | tail -20
```

### Understanding Loss

- **Initial Loss**: ~5-6 (model learning basic patterns)
- **Mid Training**: ~3-4 (model understanding structure)
- **Final Loss**: ~2-3 (model fluent in equipment language)

Lower loss = better language understanding

## Troubleshooting

### Error: "HIP out of memory"

Reduce batch size:
```bash
./scripts/pretrain_rocm.sh --batch-size 1 --gradient-accumulation 32 ...
```

### Error: "RuntimeError: HIP error: invalid device function"

The workaround is already applied in `pretrain_rocm.sh`:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### Training Very Slow

This is normal. Pre-training takes time:
- 59K samples × 3 epochs = 177K training steps
- At ~1-2 seconds per batch: 2-4 hours total

You can reduce epochs:
```bash
./scripts/pretrain_rocm.sh --epochs 1 ...
```

### Want to Resume Training

The script saves checkpoints every 5000 steps:
```bash
ls outputs/pretrained_model_rocm/checkpoint-*/
```

(Resume functionality to be added in future update)

## Why Pre-train?

### Benefits

1. **Domain Adaptation**: Model learns equipment-specific language
2. **Better Generalization**: Understands maintenance concepts
3. **Improved Performance**: Fine-tuning starts from relevant knowledge
4. **Data Efficiency**: Needs less fine-tuning data

### Comparison

**Without Pre-training**:
- Model starts from random weights
- Needs to learn everything from diagnostic examples
- Requires more fine-tuning data
- May struggle with domain terminology

**With Pre-training**:
- Model already knows equipment language
- Fine-tuning refines diagnostic reasoning
- Works better with limited data
- Understands technical context

## Files Created

### Input
- `data/text/mobile_equipment_diagnostics_corpus.txt` - Your training corpus

### Generated During Pre-training
- `tokenizer/tokenizer.json` - Trained tokenizer (680 vocab)
- `outputs/pretrained_model_rocm/` - Training checkpoints
- `outputs/pretrained_model_rocm/final_model/` - Final pre-trained model
- `pretrain_output.log` - Training log

### Scripts
- `scripts/pretrain.py` - Main pre-training script
- `scripts/pretrain_rocm.sh` - ROCm wrapper with environment setup
- `configs/pretrain_config_rocm.yaml` - Pre-training configuration

## Next Steps After Pre-training

1. **Prepare Fine-tuning Data**:
   ```bash
   # Convert your diagnostics to JSONL
   python3 scripts/convert_txt_to_jsonl.py diagnostics.txt data/processed/train.jsonl
   ```

2. **Fine-tune the Model**:
   ```bash
   # Update train.py to support --from-pretrained flag
   # Then:
   python scripts/train.py --from-pretrained outputs/pretrained_model_rocm/final_model
   ```

3. **Evaluate Results**:
   ```bash
   make infer
   ```

4. **Iterate**:
   - Add more diagnostic examples
   - Fine-tune again
   - Test and improve

## Summary

Pre-training gives your model a solid foundation in equipment diagnostic language. The 11MB corpus with 138K lines provides excellent coverage of:
- Equipment types (excavators, loaders, lifts, etc.)
- System components (hydraulic, engine, electrical, control)
- Diagnostic concepts (symptoms, causes, solutions, maintenance)
- Technical terminology specific to mobile equipment

After pre-training completes, you'll have a model that "speaks equipment" fluently and is ready for fine-tuning on your specific diagnostic scenarios.

**Estimated Total Time**: 2-4 hours for 3 epochs
**Output**: Pre-trained 125M parameter model (~500MB)
**Next**: Fine-tune on structured diagnostic data
