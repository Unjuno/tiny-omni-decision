"""Minimal parameter-free option readout for embedding students."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

SENTENCE_SIMILARITY_PREFIX = "task: sentence similarity | query:"


def decision_query_text(state: str, question: str, *, media_token: str | None = None) -> str:
    """Build the symmetric similarity-task query without leaking any answer option."""
    content = "\n".join(part.strip() for part in (state, question) if part.strip())
    if not content:
        raise ValueError("decision query must include state or question text")
    query = f"{SENTENCE_SIMILARITY_PREFIX} {content}"
    if media_token is not None:
        if not media_token.strip():
            raise ValueError("media_token must be non-empty when provided")
        query = f"{query}\n{media_token}"
    return query


def decision_option_text(option: str) -> str:
    """Build a candidate string using the same symmetric task prefix as the query."""
    content = option.strip()
    if not content:
        raise ValueError("decision option must be non-empty")
    return f"{SENTENCE_SIMILARITY_PREFIX} {content}"


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


def model_sentence_embeddings(model: Any, model_inputs: Mapping[str, Any]) -> Tensor:
    """Run an EmbeddingGemma-style encoder and apply its native pooling pipeline."""
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        raise ValueError("model_inputs must include attention_mask for native mean pooling")
    output = model(**model_inputs)
    token_embeddings = getattr(output, "last_hidden_state", None)
    if token_embeddings is None:
        raise ValueError("embedding model output must include last_hidden_state")

    pooled = mean_pool_projected_tokens(token_embeddings, attention_mask).float()
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled embeddings must be finite")
    norms = torch.linalg.vector_norm(pooled, dim=-1, keepdim=True)
    if torch.any(norms == 0):
        raise ValueError("pooled embeddings must be nonzero")
    return pooled / norms


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
    if torch.linalg.vector_norm(query_embedding) == 0 or torch.any(
        torch.linalg.vector_norm(option_embeddings, dim=-1) == 0
    ):
        raise ValueError("query and option embeddings must be nonzero")

    query = F.normalize(query_embedding.float(), dim=-1)
    options = F.normalize(option_embeddings.float(), dim=-1)
    return options @ query / temperature
