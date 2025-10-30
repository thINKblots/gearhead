"""
Unit tests for Gearhead model.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gearhead.model.gearhead_model import GearheadConfig, GearheadModel


class TestGearheadModel:
    """Test cases for GearheadModel."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return GearheadConfig(
            vocab_size=1000,
            max_position_embeddings=512,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
        )

    @pytest.fixture
    def model(self, small_config):
        """Create a model instance."""
        return GearheadModel(small_config)

    def test_model_initialization(self, model, small_config):
        """Test that model initializes correctly."""
        assert model is not None
        assert model.config == small_config
        assert model.num_parameters() > 0

    def test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        batch_size = 2
        seq_length = 10

        # Create dummy input
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        labels = input_ids.clone()

        # Forward pass
        logits, loss = model(input_ids, labels=labels)

        # Check outputs
        assert logits.shape == (batch_size, seq_length, 1000)
        assert loss is not None
        assert loss.item() > 0

    def test_forward_without_labels(self, model):
        """Test forward pass without labels."""
        batch_size = 2
        seq_length = 10

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))

        logits, loss = model(input_ids)

        assert logits.shape == (batch_size, seq_length, 1000)
        assert loss is None

    def test_generate(self, model):
        """Test text generation."""
        # Create starting sequence
        input_ids = torch.randint(0, 1000, (1, 5))

        # Generate
        output = model.generate(input_ids, max_length=20, temperature=1.0)

        # Check output
        assert output.shape[0] == 1
        assert output.shape[1] > input_ids.shape[1]
        assert output.shape[1] <= 20 + input_ids.shape[1]

    def test_causal_mask(self, model):
        """Test causal attention mask generation."""
        seq_length = 5
        device = torch.device("cpu")

        mask = model.get_causal_mask(seq_length, device)

        # Check shape
        assert mask.shape == (1, 1, seq_length, seq_length)

        # Check that it's upper triangular with -inf
        mask_2d = mask[0, 0]
        for i in range(seq_length):
            for j in range(seq_length):
                if j > i:
                    assert mask_2d[i, j] == float('-inf')
                else:
                    assert mask_2d[i, j] == 0

    def test_model_save_load(self, model, small_config, tmp_path):
        """Test saving and loading model."""
        # Save model
        save_path = tmp_path / "model.pt"
        config_path = tmp_path / "config.pt"

        torch.save(model.state_dict(), save_path)
        torch.save(model.config, config_path)

        # Create new model and load weights
        new_model = GearheadModel(small_config)
        new_model.load_state_dict(torch.load(save_path))

        # Test that loaded model works
        input_ids = torch.randint(0, 1000, (1, 10))
        output1 = model(input_ids)[0]
        output2 = new_model(input_ids)[0]

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_parameter_count(self, small_config):
        """Test parameter counting."""
        model = GearheadModel(small_config)
        num_params = model.num_parameters()

        # Count manually
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert num_params == manual_count
        assert num_params > 0


class TestGearheadConfig:
    """Test cases for GearheadConfig."""

    def test_config_creation(self):
        """Test creating a config."""
        config = GearheadConfig(
            vocab_size=32000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
        )

        assert config.vocab_size == 32000
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12

    def test_config_validation(self):
        """Test config validation."""
        # This should raise an error (hidden_size not divisible by num_heads)
        with pytest.raises(ValueError):
            GearheadConfig(
                hidden_size=768,
                num_attention_heads=7,  # 768 / 7 is not an integer
            )

    def test_default_values(self):
        """Test that default values are set."""
        config = GearheadConfig()

        assert config.vocab_size == 32000
        assert config.max_position_embeddings == 2048
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
