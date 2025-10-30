"""
Specialized attention mechanisms for diagnostic reasoning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagnosticAttention(nn.Module):
    """
    Diagnostic-aware attention mechanism that can bias attention
    towards symptom-cause relationships and error code contexts.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, \
            "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Diagnostic bias: learnable bias for symptom-cause attention
        self.diagnostic_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # Error code attention gate
        self.error_gate = nn.Linear(hidden_size, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        has_error_codes: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, 1, 1, seq_len) or None
            has_error_codes: (batch, seq_len) binary mask indicating error code positions
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Add diagnostic bias
        attn_scores = attn_scores + self.diagnostic_bias

        # Add error code attention boost
        if has_error_codes is not None:
            # Compute error code gate
            error_gate = self.error_gate(hidden_states)  # (batch, seq_len, num_heads)
            error_gate = error_gate.transpose(1, 2).unsqueeze(2)  # (batch, num_heads, 1, seq_len)
            error_gate = torch.sigmoid(error_gate)

            # Boost attention to error code positions
            error_mask = has_error_codes.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            error_boost = error_mask.float() * error_gate
            attn_scores = attn_scores + error_boost

        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Normalize
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)

        return output


class HierarchicalDiagnosticAttention(nn.Module):
    """
    Hierarchical attention for multi-level diagnostic reasoning:
    - Equipment level (vehicle/machine type)
    - System level (engine, hydraulics, electrical)
    - Component level (specific parts)
    """

    def __init__(self, hidden_size: int, num_heads: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Separate attention for each level
        self.level_attentions = nn.ModuleList([
            DiagnosticAttention(hidden_size, num_heads)
            for _ in range(num_levels)
        ])

        # Level fusion
        self.level_fusion = nn.Linear(hidden_size * num_levels, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        level_masks: list[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input representations
            level_masks: List of masks for each hierarchical level
            attention_mask: Standard attention mask
        """
        outputs = []

        for i, attn_layer in enumerate(self.level_attentions):
            level_mask = level_masks[i] if level_masks else None
            output = attn_layer(hidden_states, attention_mask, level_mask)
            outputs.append(output)

        # Concatenate and fuse
        combined = torch.cat(outputs, dim=-1)
        fused = self.level_fusion(combined)

        return fused
