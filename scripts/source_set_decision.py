"""Exploratory set-valued loss and point decision for source candidates.

Candidate sigmoid scores are conditional on a visible source; the final logit
is the none-visible probability. This is not a normalized distribution over
nonempty sets, nor are its scores proven calibrated.
"""

import torch
import torch.nn.functional as F


def _validate(logits, mask):
    if logits.ndim != 2 or mask.ndim != 2 or logits.shape != (mask.shape[0], mask.shape[1] + 1):
        raise ValueError("logits and mask shapes differ")
    if mask.dtype != torch.bool or not mask.any(dim=1).all():
        raise ValueError("each row needs at least one valid candidate")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite logits")


def set_loss(logits, mask, targets):
    """Return mean visibility BCE plus mean conditional membership BCE."""
    _validate(logits, mask)
    if targets.shape != logits.shape or targets.dtype != torch.bool:
        raise ValueError("invalid targets")
    visible_targets = targets[:, :-1]
    none_targets = targets[:, -1]
    if (visible_targets & ~mask).any() or not torch.equal(
        visible_targets.any(dim=1), ~none_targets
    ):
        raise ValueError("inconsistent or padded targets")
    none_loss = F.binary_cross_entropy_with_logits(
        logits[:, -1], none_targets.float(), reduction="none"
    )
    member_loss = F.binary_cross_entropy_with_logits(
        logits[:, :-1], visible_targets.float(), reduction="none"
    )
    member_loss = (member_loss * mask).sum(dim=1) / mask.sum(dim=1)
    return (none_loss + member_loss * (~none_targets)).mean()


def decode_set(logits, mask):
    """Return a boolean set using fixed 0.5 thresholds and nonempty fallback."""
    _validate(logits, mask)
    none = logits[:, -1] >= 0
    selected = (logits[:, :-1] >= 0) & mask & ~none[:, None]
    fallback_rows = ~none & ~selected.any(dim=1)
    masked = logits[:, :-1].masked_fill(~mask, -torch.inf)
    best = masked.max(dim=1, keepdim=True).values
    tied = torch.isclose(masked, best, rtol=0.0, atol=1e-6)
    selected |= fallback_rows[:, None] & mask & tied
    return torch.cat((selected, none[:, None]), dim=1)
