from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DecisionOutput:
    option_logits: Tensor
    probabilities: Tensor
    predicted_index: int
    confidence: float

    def to_decision_json(
        self,
        request_id: str,
        decision_type: str,
        option_ids: list[str],
        option_texts: list[str],
        ordinal_values: list[float] | None = None,
    ) -> str:
        """Serialize a typed choice result while preserving supplied option IDs/order."""
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(decision_type, str) or decision_type not in {
            "choice", "yes_no", "ordinal_score"
        }:
            raise ValueError("unsupported decision_type")
        if len(option_ids) < 2 or len(option_ids) != len(option_texts):
            raise ValueError("option IDs and texts must have matching lengths of at least two")
        if (
            any(not isinstance(value, str) or not value.strip() for value in option_ids)
            or len(set(option_ids)) != len(option_ids)
        ):
            raise ValueError("option IDs must be nonempty unique strings")
        if any(not isinstance(value, str) or not value.strip() for value in option_texts):
            raise ValueError("option texts must be nonempty strings")
        if decision_type == "yes_no" and len(option_ids) != 2:
            raise ValueError("yes_no decisions require exactly two options")
        if decision_type == "ordinal_score":
            if ordinal_values is None or len(ordinal_values) != len(option_ids):
                raise ValueError("ordinal_score requires one numeric value per option")
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in ordinal_values
                )
                or any(left >= right for left, right in zip(
                    ordinal_values[:-1], ordinal_values[1:], strict=True
                ))
            ):
                raise ValueError("ordinal values must be finite and strictly increasing")
        elif ordinal_values is not None:
            raise ValueError("ordinal_values are only valid for ordinal_score decisions")
        if self.probabilities.ndim != 1 or self.probabilities.numel() != len(option_ids):
            raise ValueError("probability vector must match the supplied option count")
        if not 0 <= self.predicted_index < len(option_ids):
            raise ValueError("predicted_index is outside the supplied option range")
        probabilities = self.probabilities.detach().cpu().tolist()
        if (
            any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
        ):
            raise ValueError("invalid probability distribution")
        if self.predicted_index != max(range(len(probabilities)), key=probabilities.__getitem__):
            raise ValueError("predicted_index does not match the probability maximum")
        if not math.isfinite(self.confidence) or not math.isclose(
            self.confidence, probabilities[self.predicted_index], abs_tol=1e-5
        ):
            raise ValueError("confidence does not match the selected-option probability")
        selected = option_ids[self.predicted_index]
        result = {
            "schema_version": 1,
            "request_id": request_id,
            "decision_type": decision_type,
            "selected_option_id": selected,
            "confidence": probabilities[self.predicted_index],
            "options": [
                {"id": option_id, "text": option_text, "probability": probability}
                for option_id, option_text, probability in zip(
                    option_ids, option_texts, probabilities, strict=True
                )
            ],
        }
        if ordinal_values is not None:
            result["expected_score"] = sum(
                probability * value
                for probability, value in zip(probabilities, ordinal_values, strict=True)
            )
        return json.dumps(result, ensure_ascii=False, allow_nan=False)

    def to_json(self, question_name: str, option_names: list[str]) -> str:
        """Serialize existing probabilities without generation or recalibration."""
        if not isinstance(question_name, str) or not question_name.strip():
            raise ValueError("question name must be a nonempty string")
        if (
            any(not isinstance(name, str) or not name.strip() for name in option_names)
            or len(set(option_names)) != len(option_names)
        ):
            raise ValueError("option names must be nonempty unique strings")
        if self.probabilities.ndim != 1 or self.probabilities.numel() < 2:
            raise ValueError("probabilities must be a vector with at least two options")
        values = self.probabilities.detach().cpu().tolist()
        if len(option_names) != len(values):
            raise ValueError("option names must match probability order and count")
        if (
            any(not math.isfinite(p) or p < 0 or p > 1 for p in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-5)
        ):
            raise ValueError("invalid probability distribution")
        return json.dumps(
            {question_name: dict(zip(option_names, values, strict=True))},
            ensure_ascii=False,
            allow_nan=False,
        )


class FrozenFeatureCandidateScorer(nn.Module):
    """Score each supplied option from precomputed observation/question features.

    Encoders and feature construction stay outside this module, so callers can
    reuse one cached observation representation across independently authored
    questions and variable-sized candidate sets.
    """

    def __init__(self, feature_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        if feature_size < 1 or hidden_size < 1:
            raise ValueError("feature_size and hidden_size must be positive")
        self.feature_size = feature_size
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, candidate_features: Tensor) -> Tensor:
        if candidate_features.ndim < 2:
            raise ValueError("candidate features must include option and feature dimensions")
        if candidate_features.shape[-1] != self.feature_size:
            raise ValueError(
                f"candidate feature size must be {self.feature_size}, "
                f"got {candidate_features.shape[-1]}"
            )
        return self.network(candidate_features).squeeze(-1)

    def decide(self, candidate_features: Tensor, temperature: float = 1.0) -> DecisionOutput:
        if candidate_features.ndim != 2 or candidate_features.shape[0] < 2:
            raise ValueError("candidate features must contain at least two options")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        logits = self(candidate_features).float() / temperature
        if not torch.isfinite(logits).all():
            raise ValueError("candidate scoring produced non-finite option logits")
        probabilities = normalize_probabilities(logits)
        predicted_index = int(probabilities.argmax().item())
        return DecisionOutput(
            option_logits=logits,
            probabilities=probabilities,
            predicted_index=predicted_index,
            confidence=float(probabilities[predicted_index].item()),
        )


def option_logits_from_vocab(last_logits: Tensor, label_ids: list[int]) -> Tensor:
    if last_logits.ndim != 1:
        raise ValueError("last_logits must be a one-dimensional vocabulary-logit vector")
    return last_logits[label_ids]


def normalize_probabilities(logits: Tensor) -> Tensor:
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("option logits must be a vector with at least two options")
    return torch.softmax(logits.float(), dim=-1)


def brier_loss(probabilities: Tensor, target_indices: Tensor) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, options]")
    targets = F.one_hot(target_indices, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    return ((probabilities - targets) ** 2).sum(dim=-1).mean()


def decision_loss(
    logits: Tensor,
    target_indices: Tensor,
    brier_weight: float = 0.2,
    cross_entropy_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    ce = F.cross_entropy(logits.float(), target_indices)
    brier = brier_loss(probabilities, target_indices)
    return cross_entropy_weight * ce + brier_weight * brier, ce, brier
