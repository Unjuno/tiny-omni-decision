"""Pure metric and selection helpers for synthetic AV readout training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor, ece_bins: int = 15
) -> dict[str, float]:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("binary readout logits must have shape [N,2]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0] or logits.shape[0] < 1:
        raise ValueError("targets must contain one class for each non-empty batch row")
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("metrics inputs must be finite")
    targets = targets.long()
    probabilities = torch.softmax(logits, dim=-1)
    predictions = logits.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    accuracy = (predictions == targets).float()
    nll = F.cross_entropy(logits, targets, reduction="none")
    one_hot = F.one_hot(targets, num_classes=2).to(probabilities.dtype)
    brier = (probabilities - one_hot).square().sum(dim=-1)
    ece = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.float().mean() * (
                accuracy[selected].mean() - confidence[selected].mean()
            ).abs()
    return {
        "accuracy": float(accuracy.mean().item()),
        "nll": float(nll.mean().item()),
        "brier": float(brier.mean().item()),
        "ece": float(ece.item()),
        "mean_confidence": float(confidence.mean().item()),
    }


@dataclass
class BestByNLL:
    """Keep the earliest checkpoint on exact ties; select by validation NLL only."""

    score: float = float("inf")
    step: int | None = None

    def update(self, validation_nll: float, step: int) -> bool:
        if not torch.isfinite(torch.tensor(validation_nll)):
            raise ValueError("validation NLL must be finite")
        if step < 0:
            raise ValueError("step must be non-negative")
        if validation_nll < self.score:
            self.score = float(validation_nll)
            self.step = int(step)
            return True
        return False


def epoch_permutations(example_count: int, epochs: int, seed: int) -> list[list[int]]:
    if example_count < 1 or epochs < 1 or seed < 0:
        raise ValueError("example_count and epochs must be positive; seed non-negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randperm(example_count, generator=generator).tolist() for _ in range(epochs)
    ]
