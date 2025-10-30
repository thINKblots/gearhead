"""
Evaluation metrics for language models.
"""

import math
from typing import List

import torch
import torch.nn.functional as F


def calculate_perplexity(model, dataloader, device: str = "cuda") -> float:
    """
    Calculate perplexity on a dataset.

    Args:
        model: Gearhead model
        dataloader: DataLoader for evaluation
        device: Device to run on

    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits, loss = model(**batch)

            # Count non-padding tokens
            labels = batch["labels"]
            non_padding = (labels != -100).sum().item()

            total_loss += loss.item() * non_padding
            total_tokens += non_padding

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score for generated text.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        BLEU score (0-100)
    """
    try:
        from datasets import load_metric
        bleu = load_metric("bleu")

        # Tokenize
        predictions_tokenized = [pred.split() for pred in predictions]
        references_tokenized = [[ref.split()] for ref in references]

        result = bleu.compute(
            predictions=predictions_tokenized,
            references=references_tokenized,
        )

        return result["bleu"] * 100

    except ImportError:
        print("Warning: datasets library not available, returning 0.0 for BLEU")
        return 0.0


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate token-level accuracy.

    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        labels: True labels (batch_size, seq_len)

    Returns:
        Accuracy percentage
    """
    predictions = torch.argmax(logits, dim=-1)

    # Mask padding tokens
    mask = labels != -100
    correct = (predictions == labels) & mask

    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy * 100


def calculate_top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Calculate top-k accuracy.

    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        labels: True labels (batch_size, seq_len)
        k: Top-k to consider

    Returns:
        Top-k accuracy percentage
    """
    # Get top-k predictions
    top_k_preds = torch.topk(logits, k=k, dim=-1).indices

    # Expand labels to compare with top-k
    labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)

    # Check if true label is in top-k
    mask = labels != -100
    mask_expanded = mask.unsqueeze(-1).expand_as(top_k_preds)

    correct = (top_k_preds == labels_expanded) & mask_expanded
    correct = correct.any(dim=-1)

    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy * 100
