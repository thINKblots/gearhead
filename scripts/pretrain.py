#!/usr/bin/env python3
"""
Pre-train Gearhead model on raw text corpus for domain adaptation.

This script performs language model pre-training on equipment diagnostic text
to help the model learn domain-specific language before fine-tuning on structured
diagnostic scenarios.

Usage:
    python scripts/pretrain.py --corpus data/text/mobile_equipment_diagnostics_corpus.txt
    python scripts/pretrain.py --config configs/pretrain_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import Dataset, DataLoader
from gearhead.data import GearheadTokenizer
from gearhead.model import GearheadModel, GearheadConfig
from gearhead.training import GearheadTrainer, TrainingConfig


class PretrainingDataset(Dataset):
    """Dataset for pre-training on raw text corpus."""

    def __init__(self, corpus_path: str, tokenizer: GearheadTokenizer, max_length: int = 512):
        """
        Initialize pre-training dataset.

        Args:
            corpus_path: Path to text corpus file
            tokenizer: Trained tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read corpus
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            text = f.read()

        # Split into sentences/paragraphs (simple splitting by double newline)
        self.texts = [t.strip() for t in text.split('\n\n') if t.strip()]

        print(f"Loaded {len(self.texts)} text segments")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Get tokenized text segment."""
        text = self.texts[idx]

        # Tokenize (encode returns list of IDs directly)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate to max_length
        input_ids = input_ids[:self.max_length]

        # Pad if necessary
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length

        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # For language modeling, labels are shifted input_ids
        labels = input_ids.clone()

        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pre-train Gearhead on text corpus")

    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to text corpus file')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/tokenizer.json',
                        help='Path to tokenizer')
    parser.add_argument('--output', type=str, default='outputs/pretrained_model',
                        help='Output directory for pre-trained model')

    # Model config
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--hidden-size', type=int, default=768)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--num-heads', type=int, default=12)
    parser.add_argument('--max-seq-length', type=int, default=512)

    # Training config
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def main():
    """Main pre-training function."""
    args = parse_args()

    print("="*70)
    print("GEARHEAD MODEL PRE-TRAINING")
    print("="*70)

    # Check if corpus exists
    if not Path(args.corpus).exists():
        print(f"Error: Corpus file not found: {args.corpus}")
        sys.exit(1)

    # Load or train tokenizer
    print("\n1. Loading Tokenizer")
    print("-"*70)

    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.exists():
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = GearheadTokenizer(tokenizer_path=str(tokenizer_path))
    else:
        print(f"Training new tokenizer on corpus...")
        tokenizer = GearheadTokenizer()
        tokenizer.train(
            files=[args.corpus],
            vocab_size=args.vocab_size,
            min_frequency=2
        )
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")

    print(f"Vocabulary size: {len(tokenizer)}")

    # Create dataset
    print("\n2. Preparing Dataset")
    print("-"*70)

    dataset = PretrainingDataset(
        corpus_path=args.corpus,
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )

    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Save datasets to temporary files for trainer
    print("\nSaving datasets to temporary files...")
    temp_data_dir = Path(args.output) / "temp_data"
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    train_data_path = temp_data_dir / "train.jsonl"
    val_data_path = temp_data_dir / "val.jsonl"

    # Save train data
    import json
    with open(train_data_path, 'w') as f:
        for idx in range(len(train_dataset)):
            # Get the actual sample from the subset
            sample = train_dataset[idx]
            # Convert to diagnostic format expected by trainer
            text = tokenizer.decode(sample['input_ids'].tolist(), skip_special_tokens=True)
            record = {
                "equipment": "Equipment Corpus",
                "symptom": text[:100] if len(text) > 100 else text,
                "error_codes": [],
                "probable_cause": "",
                "solution": text
            }
            f.write(json.dumps(record) + '\n')

    # Save val data
    with open(val_data_path, 'w') as f:
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            text = tokenizer.decode(sample['input_ids'].tolist(), skip_special_tokens=True)
            record = {
                "equipment": "Equipment Corpus",
                "symptom": text[:100] if len(text) > 100 else text,
                "error_codes": [],
                "probable_cause": "",
                "solution": text
            }
            f.write(json.dumps(record) + '\n')

    print(f"Train data saved to: {train_data_path}")
    print(f"Val data saved to: {val_data_path}")

    # Create model
    print("\n3. Initializing Model")
    print("-"*70)

    model_config = GearheadConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_position_embeddings=args.max_seq_length,
    )

    model = GearheadModel(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1e6:.1f} MB (FP32)")

    # Training config
    print("\n4. Training Configuration")
    print("-"*70)

    training_config = TrainingConfig(
        train_data_path=str(train_data_path),
        val_data_path=str(val_data_path),
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        device=args.device,
        max_seq_length=args.max_seq_length,
        save_steps=5000,
        eval_steps=1000,
        logging_steps=100,
        warmup_steps=500,
        use_rocm=True,
        rocm_optimize=True,
    )

    print(f"Batch size: {training_config.batch_size}")
    print(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Epochs: {training_config.num_epochs}")
    print(f"FP16: {training_config.fp16}")
    print(f"Device: {training_config.device}")

    # Create trainer (it will load datasets from files)
    # Note: We pass empty datasets since they're loaded from files by the trainer
    trainer = GearheadTrainer(
        model=model,
        config=training_config,
        train_dataset=None,  # Loaded from train_data_path
        tokenizer=tokenizer,
    )

    # Train
    print("\n5. Starting Pre-training")
    print("-"*70)
    print(f"Pre-training on {len(train_dataset)} text segments...")
    print(f"Validation on {len(val_dataset)} text segments...")
    print()

    trainer.train()

    # Save final model
    print("\n6. Saving Pre-trained Model")
    print("-"*70)

    output_dir = Path(args.output) / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Save config
    import json
    with open(output_dir / "config.json", 'w') as f:
        json.dump(model_config.__dict__, f, indent=2)

    # Save tokenizer
    tokenizer.save(str(output_dir / "tokenizer.json"))

    print(f"Pre-trained model saved to {output_dir}")
    print()
    print("="*70)
    print("PRE-TRAINING COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Fine-tune on diagnostic scenarios:")
    print(f"     python scripts/train.py --from-pretrained {output_dir}")
    print("  2. Or continue training:")
    print(f"     python scripts/pretrain.py --corpus data/text/more_text.txt --from-checkpoint {output_dir}")
    print()


if __name__ == "__main__":
    main()
