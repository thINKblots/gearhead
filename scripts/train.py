#!/usr/bin/env python3
"""
Training script for Gearhead model.

Example usage:
    python scripts/train.py --config configs/small_config.yaml
    python scripts/train.py --train-data data/processed/train.jsonl --val-data data/processed/val.jsonl
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml

from gearhead.data import DiagnosticDataCollator, DiagnosticDataset, GearheadTokenizer
from gearhead.model.gearhead_model import GearheadConfig, GearheadModel
from gearhead.training import GearheadTrainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Gearhead model")

    # Data
    parser.add_argument("--train-data", type=str, help="Path to training data")
    parser.add_argument("--val-data", type=str, help="Path to validation data")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json",
                        help="Path to tokenizer file")

    # Model
    parser.add_argument("--model-config", type=str, help="Path to model config file")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")

    # Training
    parser.add_argument("--config", type=str, help="Path to training config YAML file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")

    # Optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gearhead", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")

    # Checkpointing
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")

    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    args = parse_args()

    # Load config from YAML if provided
    if args.config:
        yaml_config = load_config_from_yaml(args.config)

        # Override with command line args
        for key, value in vars(args).items():
            if value is not None and key != "config":
                yaml_config[key] = value

        # Convert to namespace
        args = argparse.Namespace(**yaml_config)

    print("=" * 60)
    print("Gearhead Training")
    print("=" * 60)

    # Check for required arguments
    if not args.train_data:
        raise ValueError("--train-data is required")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer_path = Path(args.tokenizer)

    if not tokenizer_path.exists():
        print(f"Warning: Tokenizer not found at {args.tokenizer}")
        print("You need to train a tokenizer first using scripts/prepare_data.py")
        sys.exit(1)

    tokenizer = GearheadTokenizer(str(tokenizer_path))
    print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer)}")

    # Create model config
    print("\nInitializing model...")
    model_config = GearheadConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_position_embeddings=args.max_seq_length,
    )

    # Create model
    model = GearheadModel(model_config)
    print(f"Model initialized with {model.num_parameters():,} parameters")

    # Load datasets
    print(f"\nLoading training data from {args.train_data}...")
    train_dataset = DiagnosticDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )
    print(f"Training samples: {len(train_dataset)}")

    val_dataset = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_dataset = DiagnosticDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Create data collator
    data_collator = DiagnosticDataCollator(pad_token_id=tokenizer.pad_token_id)

    # Create training config
    training_config = TrainingConfig(
        train_data_path=args.train_data,
        val_data_path=args.val_data if args.val_data else None,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.wandb_run_name if args.wandb else None,
        fp16=args.fp16,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = GearheadTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=training_config,
        data_collator=data_collator,
    )

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
