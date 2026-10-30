"""Ordinal losses for small, closed-set Decision probes."""

from __future__ import annotations

import torch
from torch import Tensor


def cumulative_ordinal_nll(logits: Tensor, targets: Tensor) -> Tensor:
    """Sum binary log losses over all ordered class thresholds.

    ``targets`` are zero-based class indexes. For each threshold k this scores
    the event ``y > k`` using the class-probability mass on either side of k.
    The resulting loss penalizes distant ordinal mistakes at more thresholds
    than adjacent mistakes while retaining one normalized categorical output.
    """
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits must be [batch, classes] and targets [batch]")
    if logits.shape[1] < 2 or targets.numel() == 0:
        raise ValueError("ordinal loss requires at least two classes and one target")
    if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must have integer dtype")
    if bool((targets < 0).any()) or bool((targets >= logits.shape[1]).any()):
        raise ValueError("target class index is out of range")

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    class_count = logits.shape[1]
    log_cdf = torch.stack(
        [
            torch.logsumexp(log_probs[:, : threshold + 1], dim=-1)
            for threshold in range(class_count - 1)
        ],
        dim=-1,
    )
    log_survival = torch.stack(
        [
            torch.logsumexp(log_probs[:, threshold + 1 :], dim=-1)
            for threshold in range(class_count - 1)
        ],
        dim=-1,
    )
    thresholds = torch.arange(class_count - 1, device=targets.device)
    target_above_threshold = targets[:, None] > thresholds[None, :]
    threshold_losses = torch.where(target_above_threshold, -log_survival, -log_cdf)
    return threshold_losses.sum(dim=-1).mean()
