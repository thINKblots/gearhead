#!/usr/bin/env python3
"""Convert plain text diagnostic data to JSONL format."""

import json
import sys
from pathlib import Path


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

    scenarios = []
    current_scenario = {}

    # Split by separator or double newlines
    sections = text.split('---')

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Parse lines in this section
        lines = section.split('\n')
        temp_scenario = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                if any(x in key for x in ['problem', 'symptom', 'issue']):
                    temp_scenario['symptom'] = value
                elif any(x in key for x in ['error', 'code', 'dtc']):
                    # Split by comma or semicolon
                    if value.lower() in ['none', 'n/a', '']:
                        temp_scenario['error_codes'] = []
                    else:
                        codes = [c.strip() for c in value.replace(';', ',').split(',') if c.strip()]
                        temp_scenario['error_codes'] = codes
                elif any(x in key for x in ['cause', 'reason', 'diagnosis']):
                    temp_scenario['probable_cause'] = value
                elif any(x in key for x in ['solution', 'fix', 'repair', 'action']):
                    temp_scenario['solution'] = value
                elif any(x in key for x in ['equipment', 'machine', 'model']):
                    temp_scenario['equipment'] = value

        # Set defaults and validate
        if temp_scenario.get('symptom') and temp_scenario.get('probable_cause') and temp_scenario.get('solution'):
            if 'equipment' not in temp_scenario:
                temp_scenario['equipment'] = equipment_name
            if 'error_codes' not in temp_scenario:
                temp_scenario['error_codes'] = []

            scenarios.append(temp_scenario)

    # Write JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + '\n')

    print(f"✓ Converted {len(scenarios)} scenarios")
    print(f"✓ Saved to: {output_file}")

    if len(scenarios) == 0:
        print("\nWarning: No scenarios found!")
        print("Expected format:")
        print("  Problem: <description>")
        print("  Cause: <cause>")
        print("  Solution: <solution>")
        print("  ---")
        print("  Problem: <description>")
        print("  ...")

    return len(scenarios)


def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python convert_txt_to_jsonl.py input.txt output.jsonl [equipment_name]")
        print("\nExample:")
        print("  python convert_txt_to_jsonl.py diagnostics.txt data/processed/train.jsonl 'Caterpillar 320D'")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    equipment_name = sys.argv[3] if len(sys.argv) > 3 else "Custom Equipment"

    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    count = convert_text_to_jsonl(input_file, output_file, equipment_name)

    if count > 0:
        print("\nNext steps:")
        print(f"  1. Review the output: cat {output_file}")
        print(f"  2. Copy to training location: cp {output_file} data/processed/train.jsonl")
        print("  3. Train model: make train-small-rocm")


if __name__ == "__main__":
    main()
