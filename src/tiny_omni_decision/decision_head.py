"""A compact shared scorer over frozen query and candidate embeddings."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from .student_training import student_option_distillation_loss


class OptionDecisionHead(nn.Module):
    """Score any number of supplied options with a shared 768-D MLP scorer."""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128) -> None:
        super().__init__()
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: Tensor, options: Tensor) -> Tensor:
        if query.ndim != 1 or query.shape[0] != self.embedding_dim:
            raise ValueError(f"query must have shape [{self.embedding_dim}]")
        if options.ndim != 2 or options.shape[0] < 2 or options.shape[1] != self.embedding_dim:
            raise ValueError(f"options must have shape [at least 2, {self.embedding_dim}]")
        if query.device != options.device:
            raise ValueError("query and option embeddings must be on the same device")
        if not torch.isfinite(query).all() or not torch.isfinite(options).all():
            raise ValueError("query and option embeddings must be finite")
        expanded_query = query.unsqueeze(0).expand(options.shape[0], -1)
        features = torch.cat((expanded_query, options, expanded_query * options), dim=-1)
        return self.scorer(features).squeeze(-1)


def decision_head_parameter_count(head: nn.Module) -> int:
    return sum(parameter.numel() for parameter in head.parameters())


def train_decision_head_one_pass(
    head: OptionDecisionHead,
    query_embeddings: Tensor,
    option_embeddings: Tensor,
    option_offsets: Tensor,
    targets: Tensor,
    teacher_probabilities: Tensor,
    *,
    learning_rate: float,
    weight_decay: float,
    option_kl_weight: float,
    cross_entropy_weight: float,
    brier_weight: float,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train only the head once in input order over ragged per-example options."""
    sample_count = int(query_embeddings.shape[0])
    if query_embeddings.ndim != 2 or query_embeddings.shape[1] != head.embedding_dim:
        raise ValueError("query embeddings do not match the configured embedding dimension")
    if targets.ndim != 1 or targets.shape[0] != sample_count:
        raise ValueError("targets must contain one index per query")
    if (
        option_offsets.ndim != 1
        or option_offsets.shape[0] != sample_count + 1
        or int(option_offsets[0]) != 0
        or int(option_offsets[-1]) != option_embeddings.shape[0]
        or option_embeddings.ndim != 2
        or option_embeddings.shape[1] != head.embedding_dim
        or teacher_probabilities.shape != (option_embeddings.shape[0],)
    ):
        raise ValueError("ragged option tensors have inconsistent shapes")
    if sample_count < 1 or torch.any(option_offsets[1:] < option_offsets[:-1]):
        raise ValueError("training features must contain at least one ordered sample")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (option_kl_weight, cross_entropy_weight, brier_weight)
    ):
        raise ValueError("loss weights must be finite and non-negative")

    device = next(head.parameters()).device
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    head.train()
    for index in range(sample_count):
        start, end = int(option_offsets[index]), int(option_offsets[index + 1])
        if end - start < 2:
            raise ValueError(f"sample {index} must have at least two options")
        target_index = int(targets[index])
        if target_index < 0 or target_index >= end - start:
            raise ValueError(f"sample {index} target is outside its option range")
        query = query_embeddings[index].to(device=device, dtype=torch.float32)
        options = option_embeddings[start:end].to(device=device, dtype=torch.float32)
        teacher = teacher_probabilities[start:end].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = head(query, options)
        loss, parts = student_option_distillation_loss(
            logits,
            teacher,
            target_index=target_index,
            option_kl_weight=option_kl_weight,
            cross_entropy_weight=cross_entropy_weight,
            brier_weight=brier_weight,
        )
        loss.backward()
        gradients = [parameter.grad for parameter in head.parameters()]
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("every decision head parameter must receive a gradient")
        if any(
            not torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
        ):
            raise ValueError(f"non-finite decision head gradient at sample {index}")
        optimizer.step()
        row = {
            "update": index + 1,
            "loss": float(loss.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in parts.items()},
        }
        history.append(row)
        if on_step is not None:
            on_step(row)
    head.eval()
    return {
        "examples_consumed": sample_count,
        "optimizer_updates": sample_count,
        "mean_training_loss": sum(row["loss"] for row in history) / sample_count,
        "history": history,
    }
