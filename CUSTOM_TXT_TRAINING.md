# Training with Custom .txt Files

## Quick Start

You have two options to train with custom text files:

### Option 1: Convert .txt to JSONL Format (Recommended)

Your text needs to be in JSONL format with this structure:

```json
{"equipment": "Equipment Name", "symptom": "Problem description", "error_codes": ["CODE1"], "probable_cause": "Why it happens", "solution": "How to fix it"}
```

**Create a conversion script:**

```bash
# Create convert_txt_to_jsonl.py
cat > convert_txt_to_jsonl.py << 'EOF'
#!/usr/bin/env python3
"""Convert plain text diagnostic data to JSONL format."""

import json
import sys

def convert_text_to_jsonl(input_file, output_file, equipment_name="Custom Equipment"):
    """
    Convert plain text to JSONL format.

    Expected text format (one scenario per paragraph):

    Problem: Engine won't start
    Error Codes: P0087, P0335
    Cause: Fuel pressure sensor failure
    Solution: Replace fuel pressure sensor and test

    ---

    Problem: Hydraulic leak
    ...
    """

    with open(input_file, 'r') as f:
        text = f.read()

    # Split by separator or double newlines
    scenarios = []

    # Simple paragraph-based splitting
    paragraphs = text.split('\n\n')

    current_scenario = {}

    for para in paragraphs:
        para = para.strip()
        if not para or para == '---':
            if current_scenario:
                scenarios.append(current_scenario)
                current_scenario = {}
            continue

        # Parse lines
        lines = para.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                if 'problem' in key or 'symptom' in key or 'issue' in key:
                    current_scenario['symptom'] = value
                elif 'error' in key or 'code' in key or 'dtc' in key:
                    # Split by comma
                    codes = [c.strip() for c in value.split(',')]
                    current_scenario['error_codes'] = codes
                elif 'cause' in key or 'reason' in key:
                    current_scenario['probable_cause'] = value
                elif 'solution' in key or 'fix' in key or 'repair' in key:
                    current_scenario['solution'] = value
                elif 'equipment' in key or 'machine' in key:
                    current_scenario['equipment'] = value

        # Set defaults
        if 'symptom' in current_scenario:
            if 'equipment' not in current_scenario:
                current_scenario['equipment'] = equipment_name
            if 'error_codes' not in current_scenario:
                current_scenario['error_codes'] = []

    # Add last scenario
    if current_scenario:
        scenarios.append(current_scenario)

    # Write JSONL
    with open(output_file, 'w') as f:
        for scenario in scenarios:
            # Ensure all required fields exist
            if 'symptom' in scenario and 'probable_cause' in scenario and 'solution' in scenario:
                f.write(json.dumps(scenario) + '\n')

    print(f"Converted {len(scenarios)} scenarios")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_txt_to_jsonl.py input.txt output.jsonl [equipment_name]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    equipment_name = sys.argv[3] if len(sys.argv) > 3 else "Custom Equipment"

    convert_text_to_jsonl(input_file, output_file, equipment_name)
EOF

chmod +x convert_txt_to_jsonl.py
```

**Convert your text files:**

```bash
# Convert single file
python convert_txt_to_jsonl.py my_diagnostics.txt data/processed/train.jsonl "My Equipment"

# Or convert multiple files
python convert_txt_to_jsonl.py file1.txt data1.jsonl
python convert_txt_to_jsonl.py file2.txt data2.jsonl
python convert_txt_to_jsonl.py file3.txt data3.jsonl

# Combine them
cat data1.jsonl data2.jsonl data3.jsonl > data/processed/train.jsonl
```

**Then train:**

```bash
make train-small-rocm
```

### Option 2: Manual JSONL Creation

Create `my_training_data.jsonl` manually:

```jsonl
{"equipment": "Caterpillar 320D", "symptom": "Engine loses power under load", "error_codes": ["P0087"], "probable_cause": "Fuel filter restriction", "solution": "Replace fuel filter and check fuel pressure"}
{"equipment": "John Deere 644K", "symptom": "Hydraulic system slow response", "error_codes": [], "probable_cause": "Low hydraulic fluid or worn pump", "solution": "Check fluid level, test pump pressure, replace if needed"}
```

Then:

```bash
# Copy to training location
cp my_training_data.jsonl data/processed/train.jsonl

# Create val and test splits (or duplicate for testing)
cp my_training_data.jsonl data/processed/val.jsonl
cp my_training_data.jsonl data/processed/test.jsonl

# Train
make train-small-rocm
```

## Text Format Examples

### Format 1: Structured with Labels

```txt
Problem: Engine hard to start when cold
Error Codes: P0380, P0382
Cause: Glow plug system malfunction
Solution: Test glow plugs with multimeter. Replace failed glow plugs. Check relay and timer.

---

Problem: Hydraulic system overheating
Error Codes: None
Cause: Cooling system restriction or relief valve issue
Solution: Clean oil cooler fins. Test and adjust main relief valve to spec.

---

Problem: Transmission slipping in 3rd gear
Error Codes: P0735
Cause: Worn clutch pack or low fluid
Solution: Check fluid level. Inspect clutch pack. Replace if worn.
```

### Format 2: Free-form (Requires More Manual Work)

If your text is free-form notes, you'll need to manually structure it:

```txt
Customer reported engine starting issues on the 320D. After diagnostics,
found P0087 and P0340 codes. Fuel pressure was low at 180 bar (spec is 280-320).
Replaced fuel filter and pressure sensor. Tested - now within spec.

The 644K loader had slow hydraulic response. Checked fluid - 2 quarts low.
Topped up and tested pump pressure. Found pump output at 2000 PSI (should be 2500-2800).
Replaced hydraulic pump and filter. System now operating normally.
```

**Convert to JSONL:**

```jsonl
{"equipment": "Caterpillar 320D", "symptom": "Engine starting issues", "error_codes": ["P0087", "P0340"], "probable_cause": "Low fuel pressure due to filter restriction and faulty pressure sensor", "solution": "Replace fuel filter and fuel pressure sensor. Verify fuel pressure is within spec (280-320 bar)"}
{"equipment": "John Deere 644K Loader", "symptom": "Slow hydraulic response", "error_codes": [], "probable_cause": "Low hydraulic fluid and worn pump (low output pressure)", "solution": "Check and top up hydraulic fluid. Test pump pressure. Replace hydraulic pump and filter if pressure below spec (2500-2800 PSI)"}
```

## Step-by-Step Process

### 1. Prepare Your Text Files

Organize your diagnostic information into one of these formats:
- Service manual excerpts
- Diagnostic notes
- Troubleshooting guides
- Error code references

### 2. Convert to JSONL

**Option A - Use the conversion script** (for structured text):

```bash
python convert_txt_to_jsonl.py diagnostics.txt data/processed/train.jsonl "Equipment Type"
```

**Option B - Manual conversion** (for free-form text):

Create `data/processed/train.jsonl` with your scenarios in JSON format.

**Option C - Use AI to help**:

```bash
# Extract text
cat diagnostics.txt | head -50 > sample.txt

# Give to ChatGPT/Claude with this prompt:
# "Convert this diagnostic text into JSONL format with fields:
# equipment, symptom, error_codes (array), probable_cause, solution
# Output one JSON object per line."
```

### 3. Verify the Data

```bash
# Check format
head -3 data/processed/train.jsonl

# Count lines
wc -l data/processed/train.jsonl

# Validate JSON
python -c "
import json
with open('data/processed/train.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            assert 'symptom' in data
            assert 'probable_cause' in data
            assert 'solution' in data
        except Exception as e:
            print(f'Error on line {i}: {e}')
            break
    else:
        print('✓ All lines valid!')
"
```

### 4. Create Val/Test Splits (Optional)

```bash
# If you have lots of data, split it:
python scripts/prepare_data.py --input data/processed --output data/processed

# Or just duplicate for testing:
cp data/processed/train.jsonl data/processed/val.jsonl
cp data/processed/train.jsonl data/processed/test.jsonl
```

### 5. Train the Model

```bash
# ROCm (AMD)
make train-small-rocm

# Apple Silicon
make train-small-mps

# NVIDIA
make train-small
```

## Required JSONL Fields

Each line must be a JSON object with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `equipment` | string | Yes | Equipment type/model |
| `symptom` | string | Yes | Problem description |
| `error_codes` | array | Yes | List of error codes (can be empty `[]`) |
| `probable_cause` | string | Yes | Why the problem occurs |
| `solution` | string | Yes | How to fix it |

**Example:**

```json
{
  "equipment": "Caterpillar 320D Excavator",
  "symptom": "Engine loses power under load after 30 minutes of operation",
  "error_codes": ["P0087", "SPN 157"],
  "probable_cause": "Fuel filter restriction causing low fuel pressure",
  "solution": "Replace primary and secondary fuel filters. Check fuel quality. Verify fuel pressure is 280-320 bar at operating temperature."
}
```

## Common Issues

### Issue: "Not enough training data"

**Solution:** You need at least 100 examples, ideally 1000+. If you have less:

```bash
# Generate synthetic data to supplement
python scripts/generate_sample_data.py

# Combine with your data
cat data/processed/train.jsonl synthetic_data.jsonl > combined.jsonl
mv combined.jsonl data/processed/train.jsonl
```

### Issue: "Invalid JSON format"

**Solution:** Each line must be valid JSON. Check for:
- Unescaped quotes inside strings
- Missing commas
- Trailing commas

```bash
# Validate each line
python -c "
import json
with open('data/processed/train.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"
```

### Issue: "Text is not in the right format"

**Solution:** Your text needs structure. If it's completely unstructured:

1. Use AI to help structure it (ChatGPT/Claude)
2. Manually organize into Problem-Cause-Solution format
3. Use PDF extraction if from manuals (see PDF_TRAINING_GUIDE.md)

## Examples

### Example 1: Service Notes

**Input** (`service_notes.txt`):

```
2024-01-15 - Cat 320D - P0087 low fuel pressure
Customer reported power loss. Found contaminated fuel filter.
Replaced filter, bled system, tested OK.

2024-01-20 - JD 644K - Hydraulic slow
Low fluid level, topped up. Pump pressure low at 2000 PSI.
Replaced pump, now 2600 PSI. Working normally.
```

**Convert to JSONL:**

```jsonl
{"equipment": "Caterpillar 320D", "symptom": "Engine loses power under load", "error_codes": ["P0087"], "probable_cause": "Contaminated fuel filter causing low fuel pressure", "solution": "Replace fuel filter and bleed fuel system to remove air. Verify fuel pressure meets specification."}
{"equipment": "John Deere 644K", "symptom": "Hydraulic system responds slowly", "error_codes": [], "probable_cause": "Low hydraulic fluid and worn pump with insufficient output pressure", "solution": "Check and top up hydraulic fluid. Test pump pressure (should be 2500-2800 PSI). Replace hydraulic pump if pressure is low."}
```

### Example 2: Error Code Reference

**Input** (`error_codes.txt`):

```
P0087 - Fuel Rail Pressure Too Low
Common on diesel engines. Usually fuel filter restriction or pump failure.
Check fuel pressure, replace filter, test pump output.

P0340 - Camshaft Position Sensor Circuit Malfunction
Engine may not start or run rough. Sensor failure or wiring issue.
Test sensor resistance, check wiring, replace if needed.
```

**Convert to JSONL:**

```jsonl
{"equipment": "Diesel Engine", "symptom": "Low fuel rail pressure, possible power loss", "error_codes": ["P0087"], "probable_cause": "Fuel filter restriction or fuel pump failure", "solution": "Check fuel pressure at rail. Replace fuel filter. Test fuel pump output pressure. Replace pump if output is low."}
{"equipment": "Engine", "symptom": "Engine won't start or runs rough", "error_codes": ["P0340"], "probable_cause": "Camshaft position sensor failure or wiring fault", "solution": "Test camshaft position sensor resistance. Inspect wiring for damage. Replace sensor if resistance is out of specification."}
```

## Summary

**Quick workflow:**

1. Organize your text into Problem-Cause-Solution format
2. Convert to JSONL (one JSON object per line)
3. Save as `data/processed/train.jsonl`
4. Run: `make train-small-rocm`

**Need help?** See also:
- [PDF_TRAINING_GUIDE.md](PDF_TRAINING_GUIDE.md) - Extract from PDF manuals
- [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) - Data quality guidelines
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - How to use trained models
