"""Small tested helpers for supplied-option scoring on the Clevr-4 image task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def build_candidate_features(
    image_features: Tensor,
    option_features: Mapping[str, Tensor],
    taxonomies: Sequence[str],
) -> Tensor:
    if image_features.ndim != 2 or image_features.shape[1] != 768:
        raise ValueError("image features must have shape [batch,768]")
    if not taxonomies or len(set(taxonomies)) != len(taxonomies):
        raise ValueError("taxonomies must be nonempty and unique")
    candidates = []
    for taxonomy in taxonomies:
        options = option_features.get(taxonomy)
        if options is None or options.ndim != 2 or options.shape[1] != 384:
            raise ValueError(f"{taxonomy} options must have shape [options,384]")
        if options.shape[0] < 2:
            raise ValueError("each decision needs at least two supplied options")
        image = image_features[:, None, :].expand(-1, options.shape[0], -1)
        candidates.append(
            torch.cat([image, options[None, :, :].expand(len(image), -1, -1)], dim=-1)
        )
    widths = {candidate.shape[1] for candidate in candidates}
    if len(widths) != 1:
        raise ValueError("all taxonomy decisions in a batch must have equal option counts")
    return torch.stack(candidates, dim=1)


def classification_quality(
    probabilities: Tensor, targets: Tensor, *, bins: int = 15
) -> dict[str, float]:
    if probabilities.ndim != 2 or targets.ndim != 1 or len(probabilities) != len(targets):
        raise ValueError("expected [examples,options] probabilities and [examples] targets")
    if probabilities.shape[1] < 2 or bins < 1:
        raise ValueError("at least two options and one calibration bin are required")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not torch.allclose(
        probabilities.sum(-1), torch.ones_like(targets, dtype=torch.float32), atol=1e-5
    ):
        raise ValueError("each probability row must sum to one")
    if (targets < 0).any() or (targets >= probabilities.shape[1]).any():
        raise ValueError("target index is outside the option range")
    probabilities = probabilities.float()
    targets = targets.long()
    confidence, predictions = probabilities.max(-1)
    correct = predictions.eq(targets)
    selected = probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12)
    one_hot = torch.nn.functional.one_hot(targets, probabilities.shape[1]).float()
    ece = probabilities.new_zeros(())
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidence >= lower) & (
            confidence < upper if index + 1 < bins else confidence <= upper
        )
        if mask.any():
            ece += (
                mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
            )
    return {
        "accuracy": float(correct.float().mean()),
        "nll": float(-selected.log().mean()),
        "brier": float((probabilities - one_hot).square().sum(-1).mean()),
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()),
        "examples": int(len(targets)),
    }
