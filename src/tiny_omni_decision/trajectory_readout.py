"""Candidate-conditioned attention over complete frozen trajectory tokens."""

import math

import torch
from torch import nn


class TrajectoryReadout(nn.Module):
    """Point-order invariant; temporal order is encoded inside each token."""

    def __init__(self, hidden=64):
        super().__init__()
        self.query = nn.Linear(1536, hidden)
        self.key = nn.Linear(512, hidden)
        self.value = nn.Linear(512, hidden)
        self.scale = math.sqrt(hidden)

    def forward(self, points, candidates, mask=None):
        if points.ndim != 3 or points.shape[-1] != 512 or points.shape[1] == 0:
            raise ValueError("nonempty [batch,points,512] required")
        if (
            candidates.ndim != 3
            or candidates.shape[0] != points.shape[0]
            or candidates.shape[-1] != 1536
            or candidates.shape[1] == 0
        ):
            raise ValueError("aligned nonempty [batch,candidates,1536] required")
        if not torch.isfinite(points).all() or not torch.isfinite(candidates).all():
            raise ValueError("nonfinite inputs")
        if mask is None:
            mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if mask.shape != points.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all():
            raise ValueError("aligned boolean mask with at least one point required")
        query = self.query(candidates)
        scores = torch.einsum("bch,bnh->bcn", query, self.key(points)) / self.scale
        weights = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        observed = torch.einsum("bcn,bnh->bch", weights, self.value(points))
        return (query * observed).sum(-1) / self.scale
