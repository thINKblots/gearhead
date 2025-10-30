"""
Dataset classes for diagnostic data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class DiagnosticDataset(Dataset):
    """
    Dataset for equipment diagnostic scenarios.

    Expected data format (JSON):
    {
        "equipment": "Caterpillar 320 Excavator",
        "symptom": "Engine starts but loses power under load",
        "error_codes": ["P0087", "SPN 157"],
        "diagnostic_steps": ["Check fuel pressure...", "Inspect filters..."],
        "probable_cause": "Low fuel pressure",
        "solution": "Replace fuel filter"
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        include_error_codes: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON or JSONL file
            tokenizer: GearheadTokenizer instance
            max_length: Maximum sequence length
            include_error_codes: Whether to include error code IDs
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_error_codes = include_error_codes

        # Load data
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load diagnostic data from file."""
        data = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if path.suffix == ".json":
            with open(path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
                else:
                    data = [loaded]
        elif path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return data

    def _format_diagnostic_text(self, item: Dict) -> str:
        """
        Format diagnostic item as structured text.

        Example output:
        <equipment> Caterpillar 320 Excavator
        <symptom> Engine starts but loses power under load
        <error> P0087, SPN 157
        <cause> Low fuel pressure - contaminated fuel filter
        <solution> Replace fuel filter and check for air in system
        """
        parts = []

        # Equipment
        if "equipment" in item:
            parts.append(f"{self.tokenizer.EQUIPMENT_TOKEN} {item['equipment']}")

        # Symptom
        if "symptom" in item:
            parts.append(f"{self.tokenizer.SYMPTOM_TOKEN} {item['symptom']}")

        # Error codes
        if "error_codes" in item and item["error_codes"]:
            error_str = ", ".join(item["error_codes"])
            parts.append(f"{self.tokenizer.ERROR_CODE_TOKEN} {error_str}")

        # Probable cause
        if "probable_cause" in item:
            parts.append(f"{self.tokenizer.CAUSE_TOKEN} {item['probable_cause']}")

        # Solution
        if "solution" in item:
            parts.append(f"{self.tokenizer.SOLUTION_TOKEN} {item['solution']}")

        return "\n".join(parts)

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.

        Returns:
            Dictionary with:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Labels for language modeling (same as input_ids)
        """
        item = self.data[idx]

        # Format as text
        text = self._format_diagnostic_text(item)

        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Create attention mask
        attention_mask = [1] * len(token_ids)

        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Labels for language modeling (same as input_ids, with -100 for padding)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return result


class DiagnosticDataCollator:
    """
    Data collator for batching diagnostic samples.
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Initialize collator.

        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into batch.

        Args:
            features: List of feature dicts from dataset

        Returns:
            Batched tensors
        """
        # Stack all features
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }

        return batch
