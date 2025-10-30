#!/usr/bin/env python3
"""
Data preparation script for Gearhead.

This script:
1. Trains a tokenizer on raw text data
2. Processes and formats diagnostic data
3. Splits into train/val/test sets

Example usage:
    python scripts/prepare_data.py --input data/raw --output data/processed
    python scripts/prepare_data.py --train-tokenizer --text-files data/raw/*.txt
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gearhead.data import GearheadTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for Gearhead training")

    # Tokenizer training
    parser.add_argument("--train-tokenizer", action="store_true",
                        help="Train a new tokenizer")
    parser.add_argument("--text-files", nargs="+", help="Text files for tokenizer training")
    parser.add_argument("--tokenizer-output", type=str, default="tokenizer/tokenizer.json",
                        help="Path to save tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Tokenizer vocabulary size")

    # Data processing
    parser.add_argument("--input", type=str, help="Input data directory")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help="Train/val/test split ratio (default: 0.8 0.1 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def train_tokenizer(text_files: List[str], output_path: str, vocab_size: int):
    """
    Train a BPE tokenizer on text files.

    Args:
        text_files: List of text file paths
        output_path: Path to save trained tokenizer
        vocab_size: Target vocabulary size
    """
    print("=" * 60)
    print("Training Tokenizer")
    print("=" * 60)

    # Check files exist
    existing_files = []
    for pattern in text_files:
        matches = list(Path().glob(pattern))
        existing_files.extend([str(f) for f in matches if f.is_file()])

    if not existing_files:
        print("Error: No text files found")
        sys.exit(1)

    print(f"\nFound {len(existing_files)} text files:")
    for f in existing_files[:10]:
        print(f"  - {f}")
    if len(existing_files) > 10:
        print(f"  ... and {len(existing_files) - 10} more")

    # Create tokenizer
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    tokenizer = GearheadTokenizer()

    # Train on files
    tokenizer.train(
        files=existing_files,
        vocab_size=vocab_size,
        min_frequency=2,
    )

    # Save tokenizer
    print(f"\nSaving tokenizer to {output_path}...")
    tokenizer.save(output_path)

    print(f"Tokenizer saved! Vocabulary size: {len(tokenizer)}")
    print("=" * 60)


def create_sample_data(output_dir: Path):
    """
    Create sample diagnostic data for testing.

    Args:
        output_dir: Directory to save sample data
    """
    sample_data = [
        {
            "equipment": "Caterpillar 320 Excavator",
            "symptom": "Engine starts but loses power under load",
            "error_codes": ["P0087", "SPN 157"],
            "diagnostic_steps": [
                "Check fuel pressure at rail",
                "Inspect fuel filters for contamination",
                "Test fuel pump output",
                "Check for air in fuel system"
            ],
            "probable_cause": "Low fuel pressure due to contaminated fuel filter",
            "solution": "Replace primary and secondary fuel filters. Bleed fuel system to remove air. Check fuel pressure meets specification (280-320 bar)."
        },
        {
            "equipment": "John Deere 644K Loader",
            "symptom": "Transmission slipping in 3rd gear",
            "error_codes": ["P0735"],
            "diagnostic_steps": [
                "Check transmission fluid level and condition",
                "Scan for additional DTCs",
                "Test pressure in 3rd gear circuit",
                "Inspect clutch pack condition"
            ],
            "probable_cause": "Worn clutch pack in 3rd gear assembly or low transmission fluid",
            "solution": "Check and top up transmission fluid. If fluid level is correct, inspect and replace 3rd gear clutch pack assembly. Update transmission software if TSB available."
        },
        {
            "equipment": "Volvo EC480D Excavator",
            "symptom": "Hydraulic system overheating",
            "error_codes": [],
            "diagnostic_steps": [
                "Check hydraulic oil level and temperature",
                "Inspect cooling system operation",
                "Test relief valve settings",
                "Check for internal leakage in cylinders"
            ],
            "probable_cause": "Cooling system restriction or relief valve stuck partially open",
            "solution": "Clean hydraulic oil cooler fins. Check coolant flow and radiator fan operation. Test and adjust main relief valve to specification (350 bar)."
        },
        {
            "equipment": "Komatsu PC200 Excavator",
            "symptom": "Engine cranks but won't start",
            "error_codes": ["P0340", "P0335"],
            "diagnostic_steps": [
                "Check battery voltage",
                "Test crankshaft position sensor",
                "Test camshaft position sensor",
                "Check fuel supply to injectors"
            ],
            "probable_cause": "Faulty crankshaft or camshaft position sensor",
            "solution": "Test resistance of crankshaft position sensor (should be 500-1500 ohms). Replace sensor if out of specification. Clear codes and verify engine starts. If problem persists, check sensor wiring and ECM."
        },
        {
            "equipment": "Case 580 Backhoe",
            "symptom": "Loader bucket moves slowly",
            "error_codes": [],
            "diagnostic_steps": [
                "Check hydraulic fluid level",
                "Test pump output pressure",
                "Inspect loader control valve",
                "Check for restrictions in hydraulic lines"
            ],
            "probable_cause": "Worn hydraulic pump or restricted hydraulic filter",
            "solution": "Replace hydraulic filter and check fluid level. Test pump pressure - should be 2500-2800 PSI. If low, rebuild or replace hydraulic pump."
        },
    ]

    # Add more variations
    for i in range(15):
        sample_data.append({
            "equipment": f"Generic Equipment Type {i}",
            "symptom": f"Sample symptom description {i}",
            "error_codes": [f"ERR{i:04d}"],
            "diagnostic_steps": [
                f"Diagnostic step 1 for issue {i}",
                f"Diagnostic step 2 for issue {i}",
            ],
            "probable_cause": f"Probable cause {i}",
            "solution": f"Solution for issue {i}",
        })

    return sample_data


def process_and_split_data(input_dir: str, output_dir: str, split_ratio: List[float], seed: int):
    """
    Process diagnostic data and split into train/val/test sets.

    Args:
        input_dir: Input directory with raw data
        output_dir: Output directory for processed data
        split_ratio: [train, val, test] split ratios
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("Processing and Splitting Data")
    print("=" * 60)

    random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load or create data
    all_data = []

    if input_path.exists():
        # Load existing data files
        json_files = list(input_path.glob("*.json")) + list(input_path.glob("*.jsonl"))

        if json_files:
            print(f"\nFound {len(json_files)} data files in {input_dir}")
            for file_path in json_files:
                if file_path.suffix == ".json":
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                elif file_path.suffix == ".jsonl":
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.strip():
                                all_data.append(json.loads(line))
        else:
            print(f"\nNo data files found in {input_dir}")
            print("Creating sample data for testing...")
            all_data = create_sample_data(output_path)
    else:
        print(f"\nInput directory {input_dir} not found")
        print("Creating sample data for testing...")
        all_data = create_sample_data(output_path)

    print(f"\nTotal samples: {len(all_data)}")

    # Shuffle data
    random.shuffle(all_data)

    # Split data
    train_ratio, val_ratio, test_ratio = split_ratio
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    n = len(all_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Save splits
    def save_jsonl(data: List[dict], path: Path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    print(f"\nSaving to {output_dir}...")
    save_jsonl(train_data, output_path / "train.jsonl")
    save_jsonl(val_data, output_path / "val.jsonl")
    save_jsonl(test_data, output_path / "test.jsonl")

    print("Data preparation complete!")
    print("=" * 60)


def main():
    """Main function."""
    args = parse_args()

    # Train tokenizer if requested
    if args.train_tokenizer:
        if not args.text_files:
            print("Error: --text-files required when training tokenizer")
            sys.exit(1)

        train_tokenizer(
            text_files=args.text_files,
            output_path=args.tokenizer_output,
            vocab_size=args.vocab_size,
        )

    # Process data if input provided
    if args.input:
        process_and_split_data(
            input_dir=args.input,
            output_dir=args.output,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )
    elif not args.train_tokenizer:
        # If no specific action, create sample data
        print("No input specified. Creating sample data...")
        process_and_split_data(
            input_dir="data/raw",  # Will create sample data since it doesn't exist
            output_dir=args.output,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
