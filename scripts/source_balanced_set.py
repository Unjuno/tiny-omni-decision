"""Class-balanced membership objective for an exploratory source readout."""

import torch
import torch.nn.functional as F

from scripts.source_set_decision import _validate


def balanced_set_loss(logits, mask, targets):
    """Keep none BCE and average present positive and negative class losses."""
    _validate(logits, mask)
    if targets.shape != logits.shape or targets.dtype != torch.bool:
        raise ValueError("invalid targets")
    positive = targets[:, :-1]
    none = targets[:, -1]
    if (positive & ~mask).any() or not torch.equal(positive.any(dim=1), ~none):
        raise ValueError("inconsistent or padded targets")
    negative = mask & ~positive
    none_loss = F.binary_cross_entropy_with_logits(logits[:, -1], none.float(), reduction="none")
    raw = F.binary_cross_entropy_with_logits(logits[:, :-1], positive.float(), reduction="none")
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    pos_mean = (raw * positive).sum(dim=1) / positive_count.clamp_min(1)
    neg_mean = (raw * negative).sum(dim=1) / negative_count.clamp_min(1)
    present = (positive_count > 0).float() + (negative_count > 0).float()
    membership = (pos_mean + neg_mean) / present.clamp_min(1)
    return (none_loss + membership * (~none)).mean()
