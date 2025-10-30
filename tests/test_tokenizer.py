"""
Unit tests for Gearhead tokenizer.
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gearhead.data import GearheadTokenizer


class TestGearheadTokenizer:
    """Test cases for GearheadTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a fresh tokenizer instance."""
        return GearheadTokenizer()

    def test_tokenizer_initialization(self, tokenizer):
        """Test tokenizer initialization."""
        assert tokenizer is not None
        assert tokenizer.tokenizer is not None

    def test_special_tokens(self, tokenizer):
        """Test that special tokens are defined."""
        assert tokenizer.PAD_TOKEN == "<pad>"
        assert tokenizer.BOS_TOKEN == "<s>"
        assert tokenizer.EOS_TOKEN == "</s>"
        assert tokenizer.UNK_TOKEN == "<unk>"
        assert tokenizer.ERROR_CODE_TOKEN == "<error>"
        assert tokenizer.SYMPTOM_TOKEN == "<symptom>"
        assert tokenizer.CAUSE_TOKEN == "<cause>"
        assert tokenizer.SOLUTION_TOKEN == "<solution>"
        assert tokenizer.EQUIPMENT_TOKEN == "<equipment>"

    def test_train_and_encode(self, tmp_path):
        """Test training tokenizer and encoding text."""
        # Create sample text file
        text_file = tmp_path / "sample.txt"
        with open(text_file, "w") as f:
            f.write("This is a test sentence.\n")
            f.write("Another test sentence with some words.\n")
            f.write("Engine, hydraulic, pressure, diagnostic.\n")

        # Train tokenizer
        tokenizer = GearheadTokenizer()
        tokenizer.train([str(text_file)], vocab_size=500, min_frequency=1)

        # Test encoding
        text = "This is a test"
        token_ids = tokenizer.encode(text)

        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)

    def test_encode_decode_roundtrip(self, tmp_path):
        """Test that encode->decode returns similar text."""
        # Create and train tokenizer
        text_file = tmp_path / "sample.txt"
        with open(text_file, "w") as f:
            f.write("engine starts but loses power under load\n" * 10)
            f.write("check fuel pressure and filters\n" * 10)

        tokenizer = GearheadTokenizer()
        tokenizer.train([str(text_file)], vocab_size=500, min_frequency=1)

        # Test roundtrip
        original_text = "engine starts but loses power"
        token_ids = tokenizer.encode(original_text, add_special_tokens=False)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Should be similar (may differ in whitespace/casing due to normalization)
        assert "engine" in decoded_text.lower()
        assert "power" in decoded_text.lower()

    def test_batch_encode(self, tmp_path):
        """Test batch encoding."""
        # Create and train tokenizer
        text_file = tmp_path / "sample.txt"
        with open(text_file, "w") as f:
            f.write("test " * 100)

        tokenizer = GearheadTokenizer()
        tokenizer.train([str(text_file)], vocab_size=500)

        # Batch encode
        texts = ["first sentence", "second sentence", "third sentence"]
        token_ids_list = tokenizer.encode(texts)

        assert isinstance(token_ids_list, list)
        assert len(token_ids_list) == 3
        assert all(isinstance(ids, list) for ids in token_ids_list)

    def test_batch_decode(self, tmp_path):
        """Test batch decoding."""
        text_file = tmp_path / "sample.txt"
        with open(text_file, "w") as f:
            f.write("test sentence words " * 50)

        tokenizer = GearheadTokenizer()
        tokenizer.train([str(text_file)], vocab_size=500)

        # Encode then decode multiple sequences
        texts = ["test one", "test two"]
        token_ids_list = tokenizer.encode(texts, add_special_tokens=False)
        decoded_texts = tokenizer.decode(token_ids_list, skip_special_tokens=True)

        assert isinstance(decoded_texts, list)
        assert len(decoded_texts) == 2

    def test_save_and_load(self, tmp_path):
        """Test saving and loading tokenizer."""
        # Create and train tokenizer
        text_file = tmp_path / "sample.txt"
        with open(text_file, "w") as f:
            f.write("diagnostic error code symptom solution " * 50)

        tokenizer = GearheadTokenizer()
        tokenizer.train([str(text_file)], vocab_size=500)

        # Save tokenizer
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(save_path))

        assert save_path.exists()

        # Load tokenizer
        loaded_tokenizer = GearheadTokenizer(str(save_path))

        # Test that loaded tokenizer works
        text = "diagnostic error"
        ids1 = tokenizer.encode(text)
        ids2 = loaded_tokenizer.encode(text)

        assert ids1 == ids2

    def test_special_token_ids(self, tmp_path):
        """Test accessing special token IDs."""
        text_file = tmp_path / "sample.txt"
        with open(text_file, "w") as f:
            f.write("test " * 100)

        tokenizer = GearheadTokenizer()
        tokenizer.train([str(text_file)], vocab_size=500)

        # Check that special token IDs are accessible
        assert tokenizer.pad_token_id is not None
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.unk_token_id is not None

        # IDs should be different
        assert len({
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id
        }) == 4

    def test_vocab_size(self, tmp_path):
        """Test getting vocabulary size."""
        text_file = tmp_path / "sample.txt"
        with open(text_file, "w") as f:
            f.write("word " * 100)

        tokenizer = GearheadTokenizer()
        tokenizer.train([str(text_file)], vocab_size=500)

        vocab_size = len(tokenizer)
        assert vocab_size > 0
        assert vocab_size <= 500  # Should not exceed target


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
