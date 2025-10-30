"""
Unit tests for diagnostic dataset.
"""

import json
import pytest
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gearhead.data import DiagnosticDataset, DiagnosticDataCollator, GearheadTokenizer


class TestDiagnosticDataset:
    """Test cases for DiagnosticDataset."""

    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create sample diagnostic data file."""
        data = [
            {
                "equipment": "Caterpillar 320 Excavator",
                "symptom": "Engine loses power",
                "error_codes": ["P0087"],
                "probable_cause": "Low fuel pressure",
                "solution": "Replace fuel filter"
            },
            {
                "equipment": "Komatsu PC200",
                "symptom": "Hydraulic system overheating",
                "error_codes": [],
                "probable_cause": "Cooling system restriction",
                "solution": "Clean cooler fins"
            }
        ]

        data_file = tmp_path / "data.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

        return str(data_file)

    @pytest.fixture
    def tokenizer(self, tmp_path):
        """Create a simple trained tokenizer."""
        # Create sample text for training
        text_file = tmp_path / "train_text.txt"
        with open(text_file, "w") as f:
            f.write("equipment symptom error code cause solution " * 100)
            f.write("engine hydraulic fuel pressure temperature " * 100)

        tok = GearheadTokenizer()
        tok.train([str(text_file)], vocab_size=1000)

        return tok

    def test_dataset_loading(self, sample_data_file, tokenizer):
        """Test loading dataset from file."""
        dataset = DiagnosticDataset(
            data_path=sample_data_file,
            tokenizer=tokenizer,
            max_length=512,
        )

        assert len(dataset) == 2

    def test_dataset_getitem(self, sample_data_file, tokenizer):
        """Test getting item from dataset."""
        dataset = DiagnosticDataset(
            data_path=sample_data_file,
            tokenizer=tokenizer,
            max_length=512,
        )

        item = dataset[0]

        # Check keys
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

        # Check types
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

        # Check shapes
        assert item["input_ids"].shape == (512,)
        assert item["attention_mask"].shape == (512,)
        assert item["labels"].shape == (512,)

    def test_dataset_padding(self, sample_data_file, tokenizer):
        """Test that sequences are padded correctly."""
        dataset = DiagnosticDataset(
            data_path=sample_data_file,
            tokenizer=tokenizer,
            max_length=512,
        )

        item = dataset[0]

        # Check that padding is applied
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]

        # Find where padding starts
        pad_id = tokenizer.pad_token_id
        pad_positions = (input_ids == pad_id).nonzero(as_tuple=True)[0]

        if len(pad_positions) > 0:
            first_pad = pad_positions[0].item()

            # Attention mask should be 0 for padding
            assert attention_mask[first_pad] == 0

            # Labels should be -100 for padding
            assert labels[first_pad] == -100

    def test_dataset_jsonl_format(self, tmp_path, tokenizer):
        """Test loading JSONL format."""
        data = [
            {"equipment": "Test 1", "symptom": "Symptom 1", "error_codes": [],
             "probable_cause": "Cause 1", "solution": "Solution 1"},
            {"equipment": "Test 2", "symptom": "Symptom 2", "error_codes": ["ERR001"],
             "probable_cause": "Cause 2", "solution": "Solution 2"},
        ]

        data_file = tmp_path / "data.jsonl"
        with open(data_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        dataset = DiagnosticDataset(
            data_path=str(data_file),
            tokenizer=tokenizer,
            max_length=256,
        )

        assert len(dataset) == 2


class TestDiagnosticDataCollator:
    """Test cases for DiagnosticDataCollator."""

    @pytest.fixture
    def collator(self):
        """Create data collator."""
        return DiagnosticDataCollator(pad_token_id=0)

    @pytest.fixture
    def sample_features(self):
        """Create sample features for batching."""
        return [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 0, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 0, 0]),
                "labels": torch.tensor([1, 2, 3, 4, -100, -100]),
            },
            {
                "input_ids": torch.tensor([5, 6, 7, 8, 9, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 0]),
                "labels": torch.tensor([5, 6, 7, 8, 9, -100]),
            },
        ]

    def test_collator_batching(self, collator, sample_features):
        """Test batching features."""
        batch = collator(sample_features)

        # Check keys
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        # Check shapes (batch_size=2, seq_len=6)
        assert batch["input_ids"].shape == (2, 6)
        assert batch["attention_mask"].shape == (2, 6)
        assert batch["labels"].shape == (2, 6)

        # Check values are preserved
        assert torch.equal(batch["input_ids"][0], sample_features[0]["input_ids"])
        assert torch.equal(batch["input_ids"][1], sample_features[1]["input_ids"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
