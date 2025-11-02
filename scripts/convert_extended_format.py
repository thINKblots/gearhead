#!/usr/bin/env python3
"""Convert extended diagnostic format to standard training format."""

import json
import sys

def convert_record(record):
    """Convert extended format to standard format."""
    return {
        "equipment": record.get("equipment_type", "Unknown Equipment"),
        "symptom": record.get("issue", ""),
        "error_codes": [record.get("fault_code", "")] if record.get("fault_code") else [],
        "probable_cause": record.get("fault_label", ""),
        "solution": f"Severity: {record.get('severity', 'unknown')}. "
                   f"Check sensors: {record.get('sensors_involved', 'N/A')}. "
                   f"Tools needed: {record.get('recommended_tools', 'Standard diagnostic tools')}. "
                   f"{record.get('notes', '')}"
    }

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed/extended_mobile_equipment_diagnostics_expanded.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.jsonl', '_converted.jsonl')

    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if line.strip():
                record = json.loads(line)
                converted = convert_record(record)
                fout.write(json.dumps(converted) + '\n')
                count += 1

    print(f"✓ Converted {count} records")
    print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    main()
