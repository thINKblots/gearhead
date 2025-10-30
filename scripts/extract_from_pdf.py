#!/usr/bin/env python3
"""
Extract diagnostic information from PDF service manuals.

This script helps convert PDF service manuals into training data format.

Usage:
    python scripts/extract_from_pdf.py --pdf manual.pdf --output data.jsonl
    python scripts/extract_from_pdf.py --pdf-dir manuals/ --output data.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    print("Warning: PyPDF2 not installed. Run: pip install PyPDF2")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed. Run: pip install pdfplumber")


def extract_text_pypdf2(pdf_path: str) -> str:
    """Extract text from PDF using PyPDF2."""
    if not HAS_PYPDF2:
        raise ImportError("PyPDF2 not installed")

    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_pdfplumber(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber (better for tables)."""
    if not HAS_PDFPLUMBER:
        raise ImportError("pdfplumber not installed")

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_diagnostic_scenarios(text: str, equipment: str = None) -> List[Dict]:
    """
    Extract diagnostic scenarios from manual text.

    This is a basic example - you'll need to customize based on
    your PDF format and structure.
    """
    scenarios = []

    # Example patterns - customize for your PDFs
    # Look for common diagnostic patterns

    # Pattern 1: Problem-Cause-Solution format
    # "Problem: Engine loses power\nProbable Cause: Fuel filter\nSolution: Replace filter"
    pattern1 = re.compile(
        r'(?:Problem|Symptom|Issue):\s*(.+?)\n'
        r'(?:Probable\s*)?Cause:\s*(.+?)\n'
        r'(?:Solution|Repair|Action):\s*(.+?)(?:\n\n|\n(?=[A-Z]))',
        re.DOTALL | re.IGNORECASE
    )

    for match in pattern1.finditer(text):
        symptom = match.group(1).strip()
        cause = match.group(2).strip()
        solution = match.group(3).strip()

        scenarios.append({
            "equipment": equipment or "Unknown Equipment",
            "symptom": symptom,
            "error_codes": [],
            "probable_cause": cause,
            "solution": solution
        })

    # Pattern 2: Error code sections
    # "P0087 - Low Fuel Pressure\nCause: Fuel filter restriction\nAction: Replace filter"
    pattern2 = re.compile(
        r'([A-Z0-9]{4,6})\s*[-:]\s*(.+?)\n'
        r'(?:Cause|Reason):\s*(.+?)\n'
        r'(?:Action|Solution|Repair):\s*(.+?)(?:\n\n|\n(?=[A-Z]))',
        re.DOTALL | re.IGNORECASE
    )

    for match in pattern2.finditer(text):
        error_code = match.group(1).strip()
        symptom = match.group(2).strip()
        cause = match.group(3).strip()
        solution = match.group(4).strip()

        scenarios.append({
            "equipment": equipment or "Unknown Equipment",
            "symptom": symptom,
            "error_codes": [error_code],
            "probable_cause": cause,
            "solution": solution
        })

    return scenarios


def extract_from_pdf(
    pdf_path: str,
    equipment: Optional[str] = None,
    use_pdfplumber: bool = True
) -> List[Dict]:
    """
    Extract diagnostic scenarios from a PDF file.

    Args:
        pdf_path: Path to PDF file
        equipment: Equipment type (e.g., "Caterpillar 320 Excavator")
        use_pdfplumber: Use pdfplumber instead of PyPDF2 (better for tables)

    Returns:
        List of diagnostic scenarios
    """
    print(f"Extracting text from {pdf_path}...")

    # Extract text
    if use_pdfplumber and HAS_PDFPLUMBER:
        text = extract_text_pdfplumber(pdf_path)
    elif HAS_PYPDF2:
        text = extract_text_pypdf2(pdf_path)
    else:
        raise ImportError("Neither PyPDF2 nor pdfplumber is installed")

    # Extract equipment name from filename if not provided
    if not equipment:
        filename = Path(pdf_path).stem
        # Try to extract equipment from filename
        # e.g., "Cat_320_Service_Manual.pdf" -> "Caterpillar 320"
        equipment = filename.replace('_', ' ').replace('-', ' ')

    # Extract scenarios
    scenarios = extract_diagnostic_scenarios(text, equipment)

    print(f"Found {len(scenarios)} diagnostic scenarios")

    return scenarios


def manual_review_mode(scenarios: List[Dict]) -> List[Dict]:
    """
    Interactive mode to review and edit extracted scenarios.
    """
    print("\n" + "="*70)
    print("MANUAL REVIEW MODE")
    print("="*70)
    print("\nReview each scenario. Type 'k' to keep, 'e' to edit, 's' to skip")
    print("Type 'q' to quit and save what you have\n")

    reviewed = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}/{len(scenarios)} ---")
        print(f"Equipment: {scenario['equipment']}")
        print(f"Symptom: {scenario['symptom']}")
        print(f"Error codes: {', '.join(scenario['error_codes']) or 'None'}")
        print(f"Cause: {scenario['probable_cause']}")
        print(f"Solution: {scenario['solution']}")

        action = input("\nAction (k/e/s/q): ").lower()

        if action == 'q':
            break
        elif action == 's':
            continue
        elif action == 'e':
            print("\nEdit fields (press Enter to keep current value):")

            equipment = input(f"Equipment [{scenario['equipment']}]: ").strip()
            if equipment:
                scenario['equipment'] = equipment

            symptom = input(f"Symptom [{scenario['symptom'][:50]}...]: ").strip()
            if symptom:
                scenario['symptom'] = symptom

            cause = input(f"Cause [{scenario['probable_cause'][:50]}...]: ").strip()
            if cause:
                scenario['probable_cause'] = cause

            solution = input(f"Solution [{scenario['solution'][:50]}...]: ").strip()
            if solution:
                scenario['solution'] = solution

            reviewed.append(scenario)
        else:  # 'k' or just Enter
            reviewed.append(scenario)

    return reviewed


def save_jsonl(scenarios: List[Dict], output_path: str):
    """Save scenarios to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + '\n')

    print(f"\nSaved {len(scenarios)} scenarios to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract diagnostic data from PDF service manuals"
    )

    parser.add_argument('--pdf', type=str, help='Path to PDF file')
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDFs')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--equipment', type=str, help='Equipment name (e.g., "Caterpillar 320")')
    parser.add_argument('--review', action='store_true', help='Enable manual review mode')
    parser.add_argument('--use-pypdf2', action='store_true', help='Use PyPDF2 instead of pdfplumber')

    args = parser.parse_args()

    # Check dependencies
    if not HAS_PYPDF2 and not HAS_PDFPLUMBER:
        print("\nError: No PDF library installed!")
        print("Install one of:")
        print("  pip install PyPDF2")
        print("  pip install pdfplumber  (recommended)")
        return

    all_scenarios = []

    # Process single PDF
    if args.pdf:
        scenarios = extract_from_pdf(
            args.pdf,
            equipment=args.equipment,
            use_pdfplumber=not args.use_pypdf2
        )
        all_scenarios.extend(scenarios)

    # Process directory of PDFs
    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = list(pdf_dir.glob('*.pdf'))

        print(f"Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            scenarios = extract_from_pdf(
                str(pdf_file),
                equipment=args.equipment,
                use_pdfplumber=not args.use_pypdf2
            )
            all_scenarios.extend(scenarios)

    else:
        print("Error: Must specify --pdf or --pdf-dir")
        return

    # Review mode
    if args.review and all_scenarios:
        all_scenarios = manual_review_mode(all_scenarios)

    # Save
    if all_scenarios:
        save_jsonl(all_scenarios, args.output)

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print(f"\n1. Review the output file: {args.output}")
        print("2. Manually verify and clean the data")
        print("3. Split into train/val/test sets")
        print("4. Place in data/processed/")
        print("5. Run: make train-small-rocm")
    else:
        print("\nNo scenarios extracted. You may need to:")
        print("1. Customize the extraction patterns in extract_diagnostic_scenarios()")
        print("2. Use --review mode to manually extract data")
        print("3. Check the PDF text quality")


if __name__ == "__main__":
    main()
