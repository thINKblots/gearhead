"""
Core Gearhead model architecture based on GPT-style transformer.
Optimized for diagnostic reasoning and troubleshooting tasks.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ROCmCompatibleEmbedding(nn.Embedding):
    """
    Embedding layer with ROCm compatibility fixes for RDNA GPUs.

    Works around known issues with embedding operations on gfx103x GPUs.
    """

    def forward(self, input):
        # Check if we're on ROCm with RDNA GPU
        is_rocm = torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None

        if is_rocm and input.is_cuda:
            # Workaround: Ensure input is contiguous and properly aligned
            input = input.contiguous()

            # Ensure weight is contiguous
            if not self.weight.is_contiguous():
                self.weight.data = self.weight.data.contiguous()

        return super().forward(input)


@dataclass
class GearheadConfig:
    """Configuration for Gearhead model."""

    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5

    # Diagnostic-specific configurations
    num_error_code_embeddings: int = 10000  # Dedicated embeddings for error codes
    use_diagnostic_attention: bool = True    # Use specialized attention for diagnostics
    diagnostic_head_size: int = 64           # Size of diagnostic reasoning head

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )


class GearheadAttention(nn.Module):
    """Multi-head self-attention mechanism with optional diagnostic bias."""

    def __init__(self, config: GearheadConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.size()[:2]

        # Compute Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask (for causal masking)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        output = self.output_proj(context_layer)
        return output


class GearheadMLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: GearheadConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GearheadLayer(nn.Module):
    """Single transformer layer."""

    def __init__(self, config: GearheadConfig):
        super().__init__()
        self.attention = GearheadAttention(config)
        self.mlp = GearheadMLP(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attn_output

        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states


class GearheadModel(nn.Module):
    """
    Gearhead: Small language model optimized for equipment diagnostics.

    This is a GPT-style decoder-only transformer with specialized components
    for diagnostic reasoning and troubleshooting tasks.
    """

    def __init__(self, config: GearheadConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embeddings = ROCmCompatibleEmbedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = ROCmCompatibleEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Optional: Dedicated error code embeddings
        self.error_code_embeddings = ROCmCompatibleEmbedding(
            config.num_error_code_embeddings, config.hidden_size
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer layers
        self.layers = nn.ModuleList([
            GearheadLayer(config) for _ in range(config.num_hidden_layers)
        ])

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between token embeddings and lm_head
        self.lm_head.weight = self.token_embeddings.weight

        # Diagnostic reasoning head (optional)
        if config.use_diagnostic_attention:
            self.diagnostic_head = nn.Linear(config.hidden_size, config.diagnostic_head_size)

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing support
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Embedding, ROCmCompatibleEmbedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            # Ensure embedding weights are contiguous for ROCm
            if not module.weight.is_contiguous():
                module.weight.data = module.weight.data.contiguous()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        error_code_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token indices (batch_size, seq_length)
            attention_mask: Optional attention mask
            error_code_ids: Optional error code indices for specialized embeddings
            labels: Optional labels for computing loss

        Returns:
            logits: Output logits (batch_size, seq_length, vocab_size)
            loss: Optional language modeling loss
        """
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings
        hidden_states = token_embeds + position_embeds

        # Add error code embeddings if provided
        if error_code_ids is not None:
            error_embeds = self.error_code_embeddings(error_code_ids)
            hidden_states = hidden_states + error_embeds

        hidden_states = self.dropout(hidden_states)

        # Generate causal mask
        causal_mask = self.get_causal_mask(seq_length, device)

        # Pass through transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                hidden_states = checkpoint(
                    layer, hidden_states, causal_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, causal_mask)

        hidden_states = self.ln_f(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            input_ids: Starting token indices
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            generated_ids: Generated token sequence
        """
        self.eval()
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check for max length
                if generated.size(1) >= self.config.max_position_embeddings:
                    break

        return generated

    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
