# Quick Start: Train with Your Custom .txt Files

## 3-Step Process

### Step 1: Prepare Your Text File

Create a `.txt` file with this format:

```txt
Problem: Engine won't start
Equipment: Caterpillar 320D
Error Codes: P0087, P0335
Cause: Fuel pressure sensor failure
Solution: Replace fuel pressure sensor and test

---

Problem: Hydraulic leak
Equipment: John Deere 644K
Error Codes: None
Cause: Worn seal
Solution: Replace hydraulic cylinder seal
```

**Key points:**
- Use `---` to separate scenarios
- Include: Problem, Cause, Solution (required)
- Equipment and Error Codes are optional per scenario
- Error Codes can be "None" or comma-separated list

### Step 2: Convert to JSONL

**From `/home/mike/gearhead` directory:**

```bash
python3 scripts/convert_txt_to_jsonl.py your_file.txt data/processed/train.jsonl "Equipment Name"
```

**Or use Make:**

```bash
make convert-txt TXT=your_file.txt OUT=data/processed/train.jsonl EQUIPMENT="Equipment Name"
```

**Example:**
```bash
python3 scripts/convert_txt_to_jsonl.py example_diagnostics.txt data/processed/train.jsonl
```

### Step 3: Create Val/Test Splits and Train

```bash
# Create validation and test splits (or just duplicate for testing)
cp data/processed/train.jsonl data/processed/val.jsonl
cp data/processed/train.jsonl data/processed/test.jsonl

# Train on your data
make train-small-rocm  # For AMD GPU
make train-small-mps   # For Apple Silicon
make train-small       # For NVIDIA GPU
```

## Common Errors and Solutions

### Error: "can't open file ... No such file or directory"

**Problem:** Wrong directory or wrong path

**Solution:**
```bash
# Make sure you're in the gearhead directory
cd /home/mike/gearhead

# Use correct relative or absolute paths
python3 scripts/convert_txt_to_jsonl.py ./my_file.txt ./output.jsonl
```

### Error: "No scenarios extracted"

**Problem:** Text format not recognized

**Solution:** Make sure your text has:
- Clear labels: "Problem:", "Cause:", "Solution:"
- Separator `---` between scenarios
- At least these three fields in each scenario

**Example of what works:**
```txt
Problem: Issue description
Cause: Why it happens
Solution: How to fix

---

Problem: Another issue
Cause: Another cause
Solution: Another fix
```

### Error: "Invalid JSON format"

**Problem:** Special characters in text not properly escaped

**Solution:** The script handles this automatically. If you still get errors, check for:
- Unmatched quotes in your text
- Very long lines (script handles these)

## Text Format Examples

### Format 1: Full Details (Recommended)

```txt
Problem: Engine loses power under load
Equipment: Caterpillar 320D Excavator
Error Codes: P0087, SPN 157
Cause: Fuel filter restriction
Solution: Replace fuel filter and check pressure

---
```

### Format 2: Minimal (Also Works)

```txt
Problem: Engine won't start
Cause: Dead battery
Solution: Replace battery

---
```

### Format 3: Alternative Keywords

Any of these work:
- **Problem/Symptom/Issue**: for the problem description
- **Cause/Reason/Diagnosis**: for the root cause
- **Solution/Fix/Repair/Action**: for the fix
- **Error Codes/Error/DTC/Codes**: for error codes
- **Equipment/Machine/Model**: for equipment type

## Complete Example Workflow

```bash
# 1. Start in gearhead directory
cd /home/mike/gearhead

# 2. Check your text file format
cat my_diagnostics.txt

# 3. Convert to JSONL
python3 scripts/convert_txt_to_jsonl.py my_diagnostics.txt data/processed/train.jsonl "My Equipment"

# 4. Verify conversion
cat data/processed/train.jsonl | head -1 | python3 -m json.tool

# 5. Create splits
cp data/processed/train.jsonl data/processed/val.jsonl
cp data/processed/train.jsonl data/processed/test.jsonl

# 6. Train
make train-small-rocm

# 7. After training, test inference
make infer
```

## Using the Example File

An example file is included: [example_diagnostics.txt](example_diagnostics.txt)

```bash
# Convert the example
python3 scripts/convert_txt_to_jsonl.py example_diagnostics.txt data/processed/train.jsonl

# Create splits
cp data/processed/train.jsonl data/processed/val.jsonl
cp data/processed/train.jsonl data/processed/test.jsonl

# Train
make train-small-rocm
```

## Script Usage

```bash
python3 scripts/convert_txt_to_jsonl.py INPUT OUTPUT [EQUIPMENT_NAME]
```

**Parameters:**
- `INPUT`: Path to your .txt file
- `OUTPUT`: Where to save the .jsonl file
- `EQUIPMENT_NAME`: (Optional) Default equipment name for scenarios without one

**Examples:**

```bash
# Basic usage
python3 scripts/convert_txt_to_jsonl.py notes.txt train.jsonl

# With equipment name
python3 scripts/convert_txt_to_jsonl.py cat320.txt train.jsonl "Caterpillar 320D"

# Output to training directory
python3 scripts/convert_txt_to_jsonl.py diagnostics.txt data/processed/train.jsonl
```

## Next Steps

After training with your custom data:

1. **Test the model:**
   ```bash
   make infer
   ```

2. **Add more data:** Keep adding real diagnostic scenarios to improve the model

3. **Combine sources:**
   ```bash
   # Convert multiple files
   python3 scripts/convert_txt_to_jsonl.py file1.txt data1.jsonl
   python3 scripts/convert_txt_to_jsonl.py file2.txt data2.jsonl

   # Combine them
   cat data1.jsonl data2.jsonl > data/processed/train.jsonl
   ```

4. **See also:**
   - [CUSTOM_TXT_TRAINING.md](CUSTOM_TXT_TRAINING.md) - Full guide
   - [PDF_TRAINING_GUIDE.md](PDF_TRAINING_GUIDE.md) - Extract from PDFs
   - [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) - Data quality tips

## Tips for Best Results

✅ **DO:**
- Use real diagnostic scenarios from your experience
- Include specific error codes when available
- Provide detailed solutions with measurements/specs
- Use consistent format throughout
- Aim for 100+ scenarios minimum

❌ **DON'T:**
- Mix multiple problems in one scenario
- Skip the separator `---` between scenarios
- Use very short descriptions (too little info for model)
- Include personal information or sensitive data

## Minimum Data Requirements

- **Testing**: 10+ scenarios (like example_diagnostics.txt)
- **Useful model**: 100+ scenarios
- **Production quality**: 1,000+ scenarios

See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for details.
