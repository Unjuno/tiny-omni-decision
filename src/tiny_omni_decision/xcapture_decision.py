"""Small frozen-feature decision lens for X-Capture cross-point retrieval."""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODALITY_DIMS = {"image": 512, "audio": 384, "tactile": 512}
EMBEDDING_DIM = 128


class FrozenFeatureDecision(nn.Module):
    """Train only small modality projections over previously extracted features."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                modality: nn.Linear(input_dim, embedding_dim, bias=False)
                for modality, input_dim in MODALITY_DIMS.items()
            }
        )
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0)), dtype=torch.float32))

    def encode(self, modality: str, features: torch.Tensor) -> torch.Tensor:
        if modality not in self.projections:
            raise ValueError(f"unsupported modality: {modality}")
        return F.normalize(self.projections[modality](features), dim=-1)

    def pair_logits(
        self,
        query_modality: str,
        query_features: torch.Tensor,
        candidate_modality: str,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        query = self.encode(query_modality, query_features)
        candidates = self.encode(candidate_modality, candidate_features)
        scale = self.logit_scale.clamp(min=0.0, max=float(np.log(100.0))).exp()
        return query @ candidates.transpose(-1, -2) * scale

    def option_probabilities(
        self,
        query_modality: str,
        query_features: torch.Tensor,
        candidate_modality: str,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.pair_logits(
            query_modality, query_features, candidate_modality, candidate_features
        )
        return logits.softmax(dim=-1)


def paired_cross_modal_loss(
    model: FrozenFeatureDecision,
    features: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Symmetric cross-point InfoNCE over all ordered modality pairs."""
    first_features = next(iter(features.values()))
    if first_features.ndim != 3 or first_features.shape[1] < 2:
        raise ValueError("features must have shape [objects, points>=2, dimensions]")
    batch_size, point_count = first_features.shape[:2]
    labels = torch.arange(batch_size, device=first_features.device)
    if any(
        tensor.ndim != 3 or tensor.shape[:2] != (batch_size, point_count)
        for tensor in features.values()
    ):
        raise ValueError("modalities must share object and capture-point dimensions")
    losses = []
    for first, second in combinations(MODALITY_DIMS, 2):
        encoded_first = model.encode(first, features[first])
        encoded_second = model.encode(second, features[second])
        scale = model.logit_scale.clamp(min=0.0, max=float(np.log(100.0))).exp()
        for query_point in range(point_count):
            for candidate_point in range(point_count):
                if query_point == candidate_point:
                    continue
                logits = (
                    encoded_first[:, query_point]
                    @ encoded_second[:, candidate_point].transpose(0, 1)
                    * scale
                )
                reverse_logits = (
                    encoded_second[:, candidate_point]
                    @ encoded_first[:, query_point].transpose(0, 1)
                    * scale
                )
                losses.extend(
                    (
                        F.cross_entropy(logits, labels),
                        F.cross_entropy(reverse_logits, labels),
                    )
                )
    return torch.stack(losses).mean()


def multiclass_metrics(
    logits: torch.Tensor,
    *,
    targets: torch.Tensor | None = None,
    ece_bins: int = 15,
) -> dict[str, float]:
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("multiclass logits must have shape [examples, options>=2]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf")
    if targets is None:
        if logits.shape[0] != logits.shape[1]:
            raise ValueError("targets are required when example count differs from option count")
        targets = torch.arange(logits.shape[0], device=logits.device)
    else:
        targets = targets.to(device=logits.device, dtype=torch.long)
        if targets.shape != (logits.shape[0],):
            raise ValueError("targets must contain one class index per example")
        if (targets < 0).any() or (targets >= logits.shape[1]).any():
            raise ValueError("target class index is outside the option range")
    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correctness = predictions.eq(targets).float()
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(probabilities.dtype)
    ece = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    edges = torch.linspace(0.0, 1.0, ece_bins + 1, device=probabilities.device)
    for index in range(ece_bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == ece_bins - 1 else confidence < upper
        )
        if selected.any():
            ece += selected.float().mean() * (
                confidence[selected].mean() - correctness[selected].mean()
            ).abs()
    return {
        "accuracy": float(correctness.mean().item()),
        "nll": float(F.cross_entropy(logits, targets).item()),
        "brier": float((probabilities - one_hot).square().sum(dim=-1).mean().item()),
        "ece": float(ece.item()),
        "mean_confidence": float(confidence.mean().item()),
    }


def decision_json(option_ids: list[str], probabilities: np.ndarray | torch.Tensor) -> str:
    values = (
        probabilities.detach().cpu().numpy()
        if isinstance(probabilities, torch.Tensor)
        else np.asarray(probabilities)
    )
    if values.ndim != 1 or len(values) != len(option_ids) or len(option_ids) < 2:
        raise ValueError("provide one probability per option")
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("option IDs must be unique")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(values.sum())
    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError("probabilities must sum to one")
    predicted = int(values.argmax())
    return json.dumps(
        {
            "predicted_option_id": option_ids[predicted],
            "probabilities": [
                {"option_id": option_id, "probability": float(probability)}
                for option_id, probability in zip(option_ids, values, strict=True)
            ],
        },
        separators=(",", ":"),
    )


def projection_parameter_count(model: FrozenFeatureDecision) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
