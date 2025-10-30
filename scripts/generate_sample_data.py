#!/usr/bin/env python3
"""
Generate realistic sample diagnostic data for testing.

This creates synthetic but realistic-looking diagnostic scenarios
to test the training pipeline. For production use, replace with
real diagnostic data.
"""

import json
import random
from pathlib import Path

# Equipment types
EQUIPMENT = [
    "Caterpillar 320D Excavator",
    "Caterpillar 330 Excavator",
    "John Deere 410 Backhoe",
    "John Deere 310 Backhoe",
    "Komatsu PC200-8 Excavator",
    "Komatsu PC300 Excavator",
    "Case 580 Super N Backhoe",
    "Case 590 Backhoe",
    "Bobcat E85 Excavator",
    "Kubota KX080 Excavator",
]

# Diagnostic scenarios
SCENARIOS = [
    {
        "symptom": "Engine starts but loses power under load after {time} minutes",
        "error_codes": ["P0087", "SPN 157"],
        "probable_cause": "Low fuel pressure due to contaminated fuel filter restricting flow",
        "solution": "Replace primary and secondary fuel filters. Inspect fuel lines for air leaks. Bleed fuel system. Check fuel pressure at rail."
    },
    {
        "symptom": "Engine hard to start when cold, excessive white smoke on startup",
        "error_codes": ["P0380", "P0382"],
        "probable_cause": "Glow plug system malfunction - one or more glow plugs not heating properly",
        "solution": "Test each glow plug with multimeter (should read 0.5-2 ohms). Replace failed glow plugs. Check glow plug relay and timer."
    },
    {
        "symptom": "Hydraulic system responds slowly, especially when warm",
        "error_codes": ["HY001", "HY025"],
        "probable_cause": "Hydraulic oil viscosity breakdown due to overheating or contamination",
        "solution": "Check hydraulic oil temperature. Inspect oil cooler for blockage. Check oil filter. Take oil sample for analysis. Replace oil and filter if contaminated."
    },
    {
        "symptom": "Intermittent loss of power, engine warning light flashing",
        "error_codes": ["P0234", "P242F"],
        "probable_cause": "Turbocharger overboost condition due to wastegate sticking",
        "solution": "Inspect turbocharger wastegate for carbon buildup. Clean or replace wastegate actuator. Test boost pressure control solenoid."
    },
    {
        "symptom": "Transmission slips in {gear} gear, especially when cold",
        "error_codes": ["TR002", "TR015"],
        "probable_cause": "Clutch pack worn or oil passages restricted in {gear} gear circuit",
        "solution": "Check transmission fluid level and condition. Perform pressure test. Inspect clutch pack for wear. Clean valve body passages."
    },
    {
        "symptom": "Excessive black smoke from exhaust under load",
        "error_codes": ["P0101", "P0299"],
        "probable_cause": "Air intake restriction or turbocharger underboost",
        "solution": "Inspect air filter for clogging. Check intake system for leaks. Test turbocharger boost pressure. Inspect exhaust system for restrictions."
    },
    {
        "symptom": "Engine overheating after {time} minutes of operation",
        "error_codes": ["P0217", "P0118"],
        "probable_cause": "Cooling system failure - low coolant, failed thermostat, or blocked radiator",
        "solution": "Check coolant level and condition. Test thermostat operation. Inspect radiator for blockage. Check water pump operation. Pressure test cooling system."
    },
    {
        "symptom": "Electrical system voltage fluctuating, lights dimming",
        "error_codes": ["P0560", "P0562"],
        "probable_cause": "Alternator output low or battery connection issues",
        "solution": "Test alternator output (should be 13.5-14.5V). Check battery terminals for corrosion. Test battery condition. Inspect alternator belt tension."
    },
    {
        "symptom": "Hydraulic system leaking oil near {component}",
        "error_codes": ["HY010", "HY032"],
        "probable_cause": "Hydraulic seal failure in {component} cylinder or valve",
        "solution": "Identify exact leak source. Replace damaged hydraulic seals. Inspect cylinder rod for scoring. Check valve o-rings. Refill hydraulic reservoir."
    },
    {
        "symptom": "Engine vibration increases at {rpm} RPM",
        "error_codes": ["P0300", "P0301"],
        "probable_cause": "Engine misfire due to fuel injector malfunction or air in fuel system",
        "solution": "Test fuel injectors individually. Check fuel pressure. Bleed air from fuel system. Inspect fuel lines for leaks. Replace faulty injector."
    },
]

def generate_scenario(equipment, scenario_template):
    """Generate a diagnostic scenario with variations."""
    scenario = scenario_template.copy()

    # Add variations
    scenario["symptom"] = scenario["symptom"].format(
        time=random.choice(["15", "20", "30", "45"]),
        gear=random.choice(["2nd", "3rd", "reverse"]),
        component=random.choice(["boom", "bucket", "arm", "swing motor"]),
        rpm=random.choice(["1800", "2000", "2200", "2400"])
    )

    scenario["probable_cause"] = scenario["probable_cause"].format(
        gear=random.choice(["2nd", "3rd", "reverse"]),
        component=random.choice(["boom", "bucket", "arm", "swing motor"])
    )

    return {
        "equipment": equipment,
        "symptom": scenario["symptom"],
        "error_codes": scenario["error_codes"],
        "probable_cause": scenario["probable_cause"],
        "solution": scenario["solution"]
    }

def generate_dataset(num_examples=1000):
    """Generate a dataset of diagnostic examples."""
    examples = []

    for i in range(num_examples):
        equipment = random.choice(EQUIPMENT)
        scenario = random.choice(SCENARIOS)
        example = generate_scenario(equipment, scenario)
        examples.append(example)

    return examples

def split_dataset(examples, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train, validation, and test sets."""
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:]
    }

def save_jsonl(examples, filepath):
    """Save examples to JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

def main():
    """Generate and save sample dataset."""
    print("Generating sample diagnostic data...")
    print("=" * 60)

    # Generate examples
    num_examples = 1000  # Change this for more/less data
    examples = generate_dataset(num_examples)
    print(f"Generated {len(examples)} examples")

    # Split dataset
    splits = split_dataset(examples)
    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Val:   {len(splits['val'])} examples")
    print(f"  Test:  {len(splits['test'])} examples")

    # Save to files
    output_dir = Path("data/processed")
    save_jsonl(splits["train"], output_dir / "train.jsonl")
    save_jsonl(splits["val"], output_dir / "val.jsonl")
    save_jsonl(splits["test"], output_dir / "test.jsonl")

    print(f"\nData saved to {output_dir}/")
    print("=" * 60)
    print("\nSample example:")
    print(json.dumps(examples[0], indent=2))
    print("\n⚠️  NOTE: This is synthetic test data!")
    print("For production use, replace with real diagnostic data.")
    print("See DATA_REQUIREMENTS.md for more information.")

if __name__ == "__main__":
    main()
