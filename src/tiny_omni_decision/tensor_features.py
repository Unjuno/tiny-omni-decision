"""Small tensor helpers used by frozen-feature experiment pipelines."""

from __future__ import annotations

from torch import Tensor


def compact_cls_features(last_hidden: Tensor) -> Tensor:
    """Copy CLS rows into compact CPU storage, releasing full-token tensors."""
    return last_hidden[:, 0].float().clone().cpu()
