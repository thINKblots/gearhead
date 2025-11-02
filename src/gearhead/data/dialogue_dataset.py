"""
Dataset class for conversational dialogue data.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    """
    Dataset for conversational dialogue data.

    Expected data format (JSONL):
    {
        "dialogue": [
            "A: First utterance",
            "B: Response",
            "A: Follow-up",
            "B: Final response"
        ]
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSONL file with dialogue data
            tokenizer: GearheadTokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load dialogue data from JSONL file."""
        data = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # Skip invalid lines
        else:
            raise ValueError(f"Expected .jsonl file, got: {path.suffix}")

        return data

    def _format_dialogue(self, item: Dict) -> str:
        """
        Format dialogue item as text.

        Joins all dialogue turns into a single sequence.
        """
        if "dialogue" not in item:
            return ""

        # Join dialogue turns with newlines
        dialogue_text = "\n".join(item["dialogue"])
        return dialogue_text

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.

        Returns:
            Dict with keys:
                - input_ids: Token IDs (includes <EOS> at end)
                - attention_mask: Attention mask
                - labels: Target token IDs for language modeling
        """
        item = self.data[idx]

        # Format dialogue as text
        text = self._format_dialogue(item)

        # Tokenize (returns list of token IDs)
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate if needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)

        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # For causal language modeling, labels are the same as input_ids
        # (shifted by 1 position during training)
        labels = input_ids.clone()

        # Mask padding tokens in labels (-100 is ignored by loss function)
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
