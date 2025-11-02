#!/usr/bin/env python3
"""
Training script for conversational dialogue data.

Example usage:
    python scripts/train_conversational.py --config configs/pretrain_config_conversational_mps.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml

from gearhead.data import DialogueDataset, DiagnosticDataCollator, GearheadTokenizer
from gearhead.model.gearhead_model import GearheadConfig, GearheadModel
from gearhead.training import GearheadTrainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Gearhead on conversational data")

    # Config file
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML file")

    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    args = parse_args()

    # Load config from YAML
    config = load_config_from_yaml(args.config)

    print("=" * 60)
    print("Gearhead Conversational Pre-Training")
    print("=" * 60)

    # Check for required config
    if "train_data" not in config:
        raise ValueError("train_data is required in config")

    # Load tokenizer
    tokenizer_path = config.get("tokenizer", "tokenizer/tokenizer.json")
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("You need to train a tokenizer first using scripts/prepare_data.py")
        sys.exit(1)

    tokenizer = GearheadTokenizer(str(tokenizer_path))
    print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer)}")

    # Create model config
    print("\nInitializing model...")
    model_config = GearheadConfig(
        vocab_size=config.get("vocab_size", len(tokenizer)),
        hidden_size=config.get("hidden_size", 768),
        num_hidden_layers=config.get("num_layers", 12),
        num_attention_heads=config.get("num_attention_heads", 12),
        intermediate_size=config.get("intermediate_size", 3072),
        max_position_embeddings=config.get("max_position_embeddings", 512),
    )

    # Create model
    model = GearheadModel(model_config)
    print(f"Model initialized with {model.num_parameters():,} parameters")

    # Load dataset using DialogueDataset
    train_data_path = config["train_data"]
    max_seq_length = config.get("max_seq_length", 512)

    print(f"\nLoading training data from {train_data_path}...")
    train_dataset = DialogueDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_seq_length,
    )
    print(f"Training samples: {len(train_dataset)}")

    # Create data collator
    data_collator = DiagnosticDataCollator(pad_token_id=tokenizer.pad_token_id)

    # Create training config
    output_dir = config.get("output_dir", "outputs/pretrained_conversational_mps")
    training_config = TrainingConfig(
        train_data_path=train_data_path,
        val_data_path=None,
        max_seq_length=max_seq_length,
        batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 5.0e-5),
        num_epochs=config.get("num_epochs", 3),
        warmup_steps=config.get("warmup_steps", 1000),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        output_dir=output_dir,
        save_steps=config.get("save_steps", 5000),
        eval_steps=config.get("eval_steps", 2000),
        logging_steps=config.get("logging_steps", 100),
        use_wandb=False,  # Disable W&B for now
        fp16=config.get("fp16", True),
        device=config.get("device", "mps"),
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = GearheadTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        tokenizer=tokenizer,
        config=training_config,
        data_collator=data_collator,
    )

    # Print training info
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Dataset: {train_data_path}")
    print(f"Examples: {len(train_dataset):,}")
    print(f"Epochs: {training_config.num_epochs}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Device: {training_config.device}")
    print(f"FP16: {training_config.fp16}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Start training
    print("\nStarting training...")
    print("=" * 60 + "\n")

    trainer.train()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
