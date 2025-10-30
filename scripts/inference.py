#!/usr/bin/env python3
"""
Inference script for Gearhead diagnostic engine.

Example usage:
    # Interactive mode
    python scripts/inference.py --model outputs/final_model --tokenizer tokenizer/tokenizer.json

    # Single diagnosis
    python scripts/inference.py --model outputs/final_model --tokenizer tokenizer/tokenizer.json \
        --equipment "Caterpillar 320" --symptom "Engine loses power" --error-codes P0087

    # Batch mode
    python scripts/inference.py --model outputs/final_model --tokenizer tokenizer/tokenizer.json \
        --batch-file scenarios.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gearhead.data import GearheadTokenizer
from gearhead.inference import DiagnosticEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Gearhead diagnostic inference")

    # Required
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer file")

    # Single diagnosis mode
    parser.add_argument("--equipment", type=str, help="Equipment type/model")
    parser.add_argument("--symptom", type=str, help="Observed symptom")
    parser.add_argument("--error-codes", nargs="+", help="Error codes (space separated)")

    # Batch mode
    parser.add_argument("--batch-file", type=str, help="JSON file with multiple scenarios")

    # Generation parameters
    parser.add_argument("--max-length", type=int, default=500, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")

    # Output
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    return parser.parse_args()


def interactive_mode(engine: DiagnosticEngine, args):
    """
    Run interactive diagnostic session.

    Args:
        engine: DiagnosticEngine instance
        args: Command line arguments
    """
    print("=" * 70)
    print("Gearhead Interactive Diagnostic Mode")
    print("=" * 70)
    print("\nType 'quit' or 'exit' to end session\n")

    while True:
        print("-" * 70)

        # Get equipment
        equipment = input("Equipment type/model: ").strip()
        if equipment.lower() in ["quit", "exit"]:
            break

        # Get symptom
        symptom = input("Symptom/problem: ").strip()
        if symptom.lower() in ["quit", "exit"]:
            break

        # Get error codes (optional)
        error_codes_input = input("Error codes (comma separated, or press Enter): ").strip()
        error_codes = None
        if error_codes_input:
            error_codes = [code.strip() for code in error_codes_input.split(",")]

        # Run diagnosis
        print("\nAnalyzing...\n")

        result = engine.diagnose(
            equipment=equipment,
            symptom=symptom,
            error_codes=error_codes,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        # Display results
        print("=" * 70)
        print("DIAGNOSTIC REPORT")
        print("=" * 70)
        print(f"\nEquipment: {result['equipment']}")
        print(f"Symptom: {result['symptom']}")

        if result['error_codes']:
            print(f"Error Codes: {', '.join(result['error_codes'])}")

        print(f"\nProbable Cause:\n{result['probable_cause']}")
        print(f"\nRecommended Solution:\n{result['solution']}")
        print("=" * 70)
        print()


def single_diagnosis(engine: DiagnosticEngine, args):
    """
    Run single diagnosis from command line arguments.

    Args:
        engine: DiagnosticEngine instance
        args: Command line arguments
    """
    print("Running diagnosis...")

    result = engine.diagnose(
        equipment=args.equipment,
        symptom=args.symptom,
        error_codes=args.error_codes,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Display results
    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"\nEquipment: {result['equipment']}")
    print(f"Symptom: {result['symptom']}")

    if result['error_codes']:
        print(f"Error Codes: {', '.join(result['error_codes'])}")

    print(f"\nProbable Cause:\n{result['probable_cause']}")
    print(f"\nRecommended Solution:\n{result['solution']}")
    print("=" * 70)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


def batch_diagnosis(engine: DiagnosticEngine, args):
    """
    Run batch diagnosis from JSON file.

    Args:
        engine: DiagnosticEngine instance
        args: Command line arguments
    """
    print(f"Loading scenarios from {args.batch_file}...")

    with open(args.batch_file, "r") as f:
        scenarios = json.load(f)

    if not isinstance(scenarios, list):
        scenarios = [scenarios]

    print(f"Running {len(scenarios)} diagnoses...\n")

    results = engine.batch_diagnose(
        scenarios=scenarios,
        max_length=args.max_length,
        temperature=args.temperature,
    )

    # Display results
    for i, result in enumerate(results, 1):
        print("=" * 70)
        print(f"DIAGNOSIS {i}/{len(results)}")
        print("=" * 70)
        print(f"Equipment: {result['equipment']}")
        print(f"Symptom: {result['symptom']}")

        if result['error_codes']:
            print(f"Error Codes: {', '.join(result['error_codes'])}")

        print(f"\nProbable Cause:\n{result['probable_cause']}")
        print(f"\nRecommended Solution:\n{result['solution']}")
        print()

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nAll results saved to {args.output}")


def main():
    """Main function."""
    args = parse_args()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    # Check tokenizer exists
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {args.tokenizer}")
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = GearheadTokenizer(args.tokenizer)

    # Initialize engine
    print(f"Loading model from {args.model}...")
    engine = DiagnosticEngine(
        model_path=args.model,
        tokenizer=tokenizer,
    )

    print("Engine ready!\n")

    # Determine mode
    if args.batch_file:
        batch_diagnosis(engine, args)
    elif args.equipment and args.symptom:
        single_diagnosis(engine, args)
    else:
        interactive_mode(engine, args)


if __name__ == "__main__":
    main()
