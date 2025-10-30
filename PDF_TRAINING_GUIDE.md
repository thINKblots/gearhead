# Training from PDF Service Manuals

Yes! You can use PDF files to train your model. Here's how.

## Quick Start

### 1. Install PDF Libraries

```bash
# Install PDF extraction tools
pip install PyPDF2 pdfplumber

# Or just one (pdfplumber is better for tables)
pip install pdfplumber
```

### 2. Extract Data from PDF

```bash
# Single PDF file
python scripts/extract_from_pdf.py \
    --pdf manuals/cat320_service_manual.pdf \
    --equipment "Caterpillar 320 Excavator" \
    --output extracted_data.jsonl \
    --review

# Directory of PDFs
python scripts/extract_from_pdf.py \
    --pdf-dir manuals/ \
    --output extracted_data.jsonl \
    --review
```

### 3. Review and Clean

The `--review` flag lets you manually review each extracted scenario.

### 4. Merge with Training Data

```bash
# Combine with existing data
cat extracted_data.jsonl >> data/processed/train.jsonl

# Or replace entirely
cp extracted_data.jsonl data/processed/train.jsonl
```

### 5. Train Model

```bash
make train-small-rocm
```

## PDF Format Requirements

### What Works Best

PDFs with structured diagnostic information:

**Good formats**:
```
Problem: Engine loses power under load
Probable Cause: Fuel pressure low
Solution: Replace fuel filter, check fuel pump

---

Error Code: P0087
Description: Fuel Rail Pressure Too Low
Cause: Restricted fuel filter or weak fuel pump
Repair: Replace fuel filter. Test fuel pressure.
```

**Also works**:
- Diagnostic flowcharts (if text-based)
- Troubleshooting tables
- Error code reference sections
- Service procedures

**Difficult**:
- Scanned images (need OCR)
- Complex diagrams
- Multi-column layouts
- Non-standard formats

## Extraction Methods

### Method 1: Automatic (Best for Structured PDFs)

```bash
python scripts/extract_from_pdf.py \
    --pdf manual.pdf \
    --equipment "Caterpillar 320" \
    --output data.jsonl
```

The script looks for patterns like:
- "Problem: ... Cause: ... Solution: ..."
- "Error Code ... Cause: ... Action: ..."

### Method 2: Manual Review (Most Accurate)

```bash
python scripts/extract_from_pdf.py \
    --pdf manual.pdf \
    --review \
    --output data.jsonl
```

Review each extracted scenario and edit/skip as needed.

### Method 3: Semi-Automatic

1. Extract text to file:
```python
import pdfplumber

with pdfplumber.open("manual.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

with open("manual_text.txt", "w") as f:
    f.write(text)
```

2. Manually format into JSONL
3. Use your text editor or write custom parsing

### Method 4: Use AI to Help

Extract text, then use ChatGPT/Claude to structure it:

```bash
# Extract text
python -c "
import pdfplumber
with pdfplumber.open('manual.pdf') as pdf:
    for page in pdf.pages[:10]:  # First 10 pages
        print(page.extract_text())
" > manual_excerpt.txt

# Then give to ChatGPT with prompt:
# "Convert this diagnostic text into JSONL format with fields:
# equipment, symptom, error_codes, probable_cause, solution"
```

## Customizing the Extraction Script

The extraction script uses regex patterns. Customize for your PDFs:

```python
# In scripts/extract_from_pdf.py, modify this function:
def extract_diagnostic_scenarios(text: str, equipment: str = None):
    # Add your own patterns

    # Example: Your manual might use this format:
    # "SYMPTOM: Engine stalls
    #  DIAGNOSIS: Check fuel system
    #  REPAIR: Replace fuel pump"

    pattern = re.compile(
        r'SYMPTOM:\s*(.+?)\n'
        r'DIAGNOSIS:\s*(.+?)\n'
        r'REPAIR:\s*(.+?)(?:\n\n|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    # Extract and format...
```

## Example: Real PDF Extraction

### Step-by-Step

1. **Get PDFs**
   - Service manuals (legal copies only!)
   - Technical bulletins
   - Diagnostic guides

2. **Extract text**
```bash
python scripts/extract_from_pdf.py \
    --pdf Cat_320D_Hydraulic_System.pdf \
    --equipment "Caterpillar 320D Excavator" \
    --review \
    --output hydraulic_diagnostics.jsonl
```

3. **Review output**
```bash
# Check what was extracted
head -5 hydraulic_diagnostics.jsonl
```

4. **Clean and verify**
```python
import json

# Load and verify
with open('hydraulic_diagnostics.jsonl') as f:
    for line in f:
        data = json.loads(line)
        print(f"Equipment: {data['equipment']}")
        print(f"Symptom: {data['symptom']}")
        print(f"Cause: {data['probable_cause']}")
        print()
```

5. **Combine datasets**
```bash
# Merge multiple extracted files
cat hydraulic_diagnostics.jsonl \
    engine_diagnostics.jsonl \
    electrical_diagnostics.jsonl \
    > combined_diagnostics.jsonl

# Split into train/val/test
python scripts/split_data.py combined_diagnostics.jsonl
```

6. **Train**
```bash
make train-small-rocm
```

## OCR for Scanned PDFs

If your PDFs are scanned images:

```bash
# Install OCR tools
pip install pytesseract pdf2image
# Also need tesseract: sudo apt install tesseract-ocr

# OCR the PDF
python -c "
from pdf2image import convert_from_path
import pytesseract

images = convert_from_path('scanned_manual.pdf')
text = ''
for img in images:
    text += pytesseract.image_to_string(img)

with open('ocr_output.txt', 'w') as f:
    f.write(text)
"

# Then process the text file
```

## Quality Tips

### Good Data from PDFs

✅ **Clear diagnostic sections**
- Problem/Cause/Solution format
- Error code descriptions
- Troubleshooting flowcharts (text-based)

✅ **Specific information**
- Exact error codes
- Detailed symptoms
- Step-by-step solutions

✅ **Consistent formatting**
- Same structure throughout
- Clear section markers

### Skip These Sections

❌ Safety warnings (not diagnostic)
❌ Parts lists (not diagnostic)
❌ Maintenance schedules (not diagnostic)
❌ Diagrams without text
❌ Specifications tables

## Example Output

Good extraction from PDF:

```json
{"equipment": "Caterpillar 320D Excavator", "symptom": "Hydraulic system responds slowly when warm, bucket curl weak under load", "error_codes": ["HY001"], "probable_cause": "Hydraulic oil viscosity breakdown due to overheating or contamination", "solution": "Check hydraulic oil temperature (should be below 180°F). Inspect oil cooler for blockage. Check oil filter for contamination. Take oil sample for analysis. Replace hydraulic oil and filter if contaminated."}

{"equipment": "Caterpillar 320D Excavator", "symptom": "Engine hard to start when cold, excessive white smoke on startup", "error_codes": ["P0380", "P0382"], "probable_cause": "Glow plug system malfunction - one or more glow plugs not heating properly", "solution": "Test each glow plug with multimeter (should read 0.5-2 ohms). Replace failed glow plugs. Check glow plug relay and timer. Verify glow plug controller operation."}
```

## Common PDF Formats

### Caterpillar SIS
- Well-structured
- Error codes clearly marked
- Good for automatic extraction

### John Deere Technical Manuals
- Diagnostic trouble codes section
- Symptom-based diagnostics
- May need custom patterns

### Komatsu Shop Manuals
- Troubleshooting flowcharts
- May need manual extraction
- Good quality information

### OEM Service Bulletins
- Specific issues
- Detailed solutions
- Perfect for training data

## Combining Multiple Sources

```bash
# Extract from different manuals
python scripts/extract_from_pdf.py --pdf cat_320_hydraulic.pdf --output cat_hyd.jsonl
python scripts/extract_from_pdf.py --pdf cat_320_engine.pdf --output cat_eng.jsonl
python scripts/extract_from_pdf.py --pdf cat_320_electrical.pdf --output cat_elec.jsonl

# Combine
cat cat_*.jsonl > data/processed/train.jsonl

# Train on comprehensive dataset
make train-small-rocm
```

## Validation

After extraction, validate your data:

```python
import json

with open('extracted_data.jsonl') as f:
    scenarios = [json.loads(line) for line in f]

print(f"Total scenarios: {len(scenarios)}")

# Check for quality
for s in scenarios[:5]:
    assert len(s['symptom']) > 20, "Symptom too short"
    assert len(s['probable_cause']) > 10, "Cause too short"
    assert len(s['solution']) > 20, "Solution too short"

print("✓ Data validation passed")
```

## Summary

**Can you use PDFs?** Yes! ✅

**Best approach**:
1. Install: `pip install pdfplumber`
2. Extract: `python scripts/extract_from_pdf.py --pdf manual.pdf --review --output data.jsonl`
3. Verify quality
4. Train: `make train-small-rocm`

**Quality matters more than quantity!**
- 100 high-quality extractions > 1000 poor extractions
- Manual review is worth the time
- Clean data = better model

**Need help?** The extraction script is customizable for your PDF format.
