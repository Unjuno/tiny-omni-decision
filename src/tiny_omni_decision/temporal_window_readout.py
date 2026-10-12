"""Candidate-conditioned scoring of variable-length frozen video windows."""

import math

import torch
from torch import nn


class TemporalWindowReadout(nn.Module):
    def __init__(self, mode="attention", hidden=64):
        super().__init__()
        if mode not in {"attention", "mean"}:
            raise ValueError("unknown mode")
        self.mode = mode
        self.query = nn.Linear(1536, hidden)
        self.key = nn.Linear(768, hidden)
        self.value = nn.Linear(768, hidden)
        self.position = nn.Linear(1, hidden, bias=False)
        self.scale = math.sqrt(hidden)

    def forward(self, windows, candidates, mask):
        if windows.ndim != 3 or windows.shape[-1] != 768:
            raise ValueError("expected [batch,windows,768]")
        if mask.shape != windows.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("aligned boolean window mask required")
        if not mask.any(-1).all():
            raise ValueError("empty video")
        if candidates.ndim != 3 or candidates.shape[0] != windows.shape[0]:
            raise ValueError("aligned candidates required")
        if candidates.shape[-1] != 1536:
            raise ValueError("expected 1536 candidate features")
        query = self.query(candidates)
        # Relative position is computed over real windows, independent of padding.
        position = (mask.cumsum(-1) - 1) / (mask.sum(-1, keepdim=True) - 1).clamp_min(1)
        position = 2 * position - 1
        if self.mode == "attention":
            keys = self.key(windows) + self.position(position[..., None].to(windows.dtype))
            scores = torch.einsum("bch,bnh->bcn", query, keys) / self.scale
            weights = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
            observed = torch.einsum("bcn,bnh->bch", weights, self.value(windows))
        else:
            pooled = (windows * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            observed = self.value(pooled)[:, None]
        return (observed * query).sum(-1) / self.scale
