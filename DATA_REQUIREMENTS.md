# Data Requirements for Training

## Current Problem

The sample data is **not usable** for real training:
- Only 16 examples (need 1000s)
- Generic placeholders like "Generic Equipment Type 0"
- No real diagnostic knowledge

**Result**: Model trains in seconds, produces gibberish output

## What You Need

### Minimum Data Requirements

For a useful model, you need:
- **Training examples**: 1,000+ (ideally 10,000+)
- **Real equipment**: Actual makes/models
- **Real symptoms**: Detailed problem descriptions
- **Real error codes**: OBD-II, manufacturer codes
- **Real solutions**: Actual repair procedures

### Data Format

Each example should be in JSONL format:

```json
{
  "equipment": "Caterpillar 320 Excavator",
  "symptom": "Engine starts but loses power under load after 30 minutes, white smoke from exhaust",
  "error_codes": ["P0087", "SPN 157", "FMI 1"],
  "probable_cause": "Low fuel pressure due to contaminated fuel filter restricting flow",
  "solution": "Replace primary and secondary fuel filters. Inspect fuel lines for air leaks. Bleed fuel system. Check fuel pressure at rail (should be 1600-1800 psi). If pressure still low, inspect fuel pump."
}
```

**Key characteristics**:
- Specific equipment (manufacturer + model)
- Detailed symptoms (what, when, how)
- Actual error codes (P-codes, SPN, FMI, manufacturer codes)
- Technical probable causes
- Step-by-step solutions

## Where to Get Data

### Option 1: Your Own Data (Best!)

If you work with heavy equipment:
1. Export service tickets from your system
2. Convert to JSONL format
3. Include: equipment, symptoms, codes, diagnoses, solutions

### Option 2: Service Manuals

Digitize diagnostic flowcharts from:
- OEM service manuals
- Technical bulletins
- Troubleshooting guides

Format each diagnostic scenario as a training example.

### Option 3: Online Forums & Databases

Aggregate from:
- Equipment forums (HeavyEquipmentForums, IronPlanet)
- Diagnostic databases
- Technical Q&A sites

**Note**: Respect copyright and terms of service!

### Option 4: Generate Synthetic Data

Use a larger model (ChatGPT, Claude) to generate realistic examples:

```
Prompt: Generate 100 realistic heavy equipment diagnostic scenarios for Caterpillar excavators, including specific symptoms, error codes, probable causes, and detailed repair solutions. Format as JSON.
```

Then manually review for accuracy.

### Option 5: Example Real Data

Here's what good training data looks like:

```json
{"equipment": "Caterpillar 320D Excavator", "symptom": "Engine hard to start when cold, excessive white smoke on startup, clears after warmup", "error_codes": ["P0380", "P0382"], "probable_cause": "Glow plug system malfunction - one or more glow plugs not heating properly", "solution": "Test each glow plug with multimeter (should read 0.5-2 ohms). Replace failed glow plugs. Check glow plug relay and timer. Verify glow plug controller operation."}

{"equipment": "John Deere 410 Backhoe", "symptom": "Hydraulic system responds slowly, especially when warm. Bucket curl weak under load", "error_codes": ["HY001", "HY025"], "probable_cause": "Hydraulic oil viscosity breakdown due to overheating or contamination", "solution": "Check hydraulic oil temperature (should be below 180°F). Inspect oil cooler for blockage. Check oil filter for contamination. Take oil sample for analysis. Replace hydraulic oil and filter if contaminated. Inspect hydraulic pump for wear."}

{"equipment": "Komatsu PC200-8 Excavator", "symptom": "Intermittent loss of power, engine warning light, happens under heavy load", "error_codes": ["P0234", "P242F"], "probable_cause": "Turbocharger overboost condition due to wastegate sticking or boost control solenoid failure", "solution": "Inspect turbocharger wastegate for carbon buildup or sticking. Clean or replace wastegate actuator. Test boost pressure control solenoid. Check for exhaust restrictions. Verify intercooler for leaks. Replace turbocharger if wastegate mechanism damaged."}

{"equipment": "Case 580 Super N Backhoe", "symptom": "Transmission slips in 2nd gear, especially when cold, operates normally when warm", "error_codes": ["TR002"], "probable_cause": "2nd gear clutch pack worn or 2nd gear oil passages restricted", "solution": "Check transmission fluid level and condition. Perform pressure test on 2nd gear circuit. If pressure low when cold, inspect 2nd gear clutch pack for wear. Check valve body for stuck valves. Replace clutch pack if worn. Clean valve body passages."}
```

## Data Preparation Script

I'll create a script to help you format your data:

```python
# scripts/prepare_custom_data.py
import json

# Your data (replace with real data)
diagnostics = [
    {
        "equipment": "Caterpillar 320D Excavator",
        "symptom": "Engine hard to start when cold",
        "error_codes": ["P0380", "P0382"],
        "probable_cause": "Glow plug system malfunction",
        "solution": "Test and replace glow plugs"
    },
    # Add 1000+ more examples here...
]

# Save as JSONL
with open("data/processed/train.jsonl", "w") as f:
    for item in diagnostics:
        f.write(json.dumps(item) + "\n")
```

## How Much Data?

| Dataset Size | Expected Quality | Training Time |
|--------------|------------------|---------------|
| 100 examples | Poor - overfits | 5 minutes |
| 1,000 examples | Basic - limited knowledge | 30 minutes |
| 10,000 examples | Good - useful diagnostics | 2-3 hours |
| 100,000 examples | Excellent - expert level | 20-30 hours |

## Data Quality Checklist

For each example, ensure:
- ✅ Real equipment (no "Generic Equipment Type")
- ✅ Specific symptoms (detailed, not vague)
- ✅ Actual error codes (verifiable)
- ✅ Technical causes (root cause analysis)
- ✅ Actionable solutions (step-by-step procedures)
- ✅ Diverse scenarios (different equipment, issues)

## Next Steps

### 1. Decide on Data Source

Choose one:
- [ ] Export your company's service data
- [ ] Digitize service manuals
- [ ] Generate synthetic data with LLM
- [ ] Find open diagnostic databases

### 2. Collect Data

Gather 1,000+ examples minimum

### 3. Format Data

Convert to JSONL format (one JSON object per line)

### 4. Split Data

- Training: 80% (train.jsonl)
- Validation: 10% (val.jsonl)
- Test: 10% (test.jsonl)

### 5. Re-train Model

```bash
# After preparing real data
make train-small-rocm
```

## Reality Check

**Current situation**:
- Sample data: 16 examples of generic text
- Training: Seconds per epoch, loss ~4.5
- Model output: Gibberish

**With real data** (1000+ examples):
- Training: 30-60 minutes
- Loss: Should drop to ~1.5-2.5
- Model output: Coherent diagnostic advice

## Can I Test Without Real Data?

Yes! Use a pre-trained model or:

1. **Generate synthetic data** with ChatGPT/Claude
2. Use it to test the training pipeline
3. Replace with real data later

Would you like me to:
1. Generate sample realistic data for testing?
2. Create a data collection script?
3. Show how to use an existing dataset?

## Bottom Line

**You need real diagnostic data to train a useful model.**

The current sample data is just for testing the pipeline, not for creating a functional diagnostic system.
