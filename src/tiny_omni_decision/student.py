"""Minimal parameter-free option readout for embedding students."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def mean_pool_projected_tokens(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Apply the pinned Sentence Transformers mean-pooling semantics."""
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [sequence, tokens, dimensions]")
    if attention_mask.ndim != 2 or attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention_mask must match the first two token_embeddings dimensions")

    mask = attention_mask.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    token_counts = mask.sum(dim=1, keepdim=True)
    if torch.any(token_counts <= 0):
        raise ValueError("every sequence must contain at least one attended token")
    return (token_embeddings * mask.unsqueeze(-1)).sum(dim=1) / token_counts


def supplied_option_logits(
    query_embedding: Tensor,
    option_embeddings: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Score supplied options by cosine similarity, preserving their input order."""
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a one-dimensional vector")
    if option_embeddings.ndim != 2 or option_embeddings.shape[0] < 2:
        raise ValueError("option_embeddings must have shape [at least two options, dimensions]")
    if option_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("query and option embedding dimensions must match")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(query_embedding).all() or not torch.isfinite(option_embeddings).all():
        raise ValueError("embeddings must be finite")

    query = F.normalize(query_embedding.float(), dim=-1)
    options = F.normalize(option_embeddings.float(), dim=-1)
    return options @ query / temperature
