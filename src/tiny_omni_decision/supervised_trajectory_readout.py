"""Shared role-supervised representations for supplied-option Video decisions."""

import math

import torch
from torch import nn

from .cover_point_readout import CoverPointReadout


class SupervisedTrajectoryReadout(nn.Module):
    """Inference receives only frozen state/candidate features, never source labels."""

    def __init__(self, hidden=64):
        super().__init__()
        self.cover = CoverPointReadout(hidden=hidden)
        self.query = nn.Linear(1536, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.scale = math.sqrt(hidden)

    def forward(self, points, candidates, mask=None):
        if (
            candidates.ndim != 3
            or candidates.shape[0] != points.shape[0]
            or candidates.shape[-1] != 1536
            or candidates.shape[1] == 0
            or not torch.isfinite(candidates).all()
        ):
            raise ValueError("aligned nonempty finite supplied-candidate features required")
        features, mask = self.cover.encode(points, mask)
        cover_logits = self.cover.score(features).squeeze(-1)
        query = self.query(candidates)
        scores = torch.einsum("bch,bnh->bcn", query, features) / self.scale
        scores = scores + cover_logits[:, None]
        weights = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        observed = torch.einsum("bcn,bnh->bch", weights, self.value(features))
        return (query * observed).sum(-1) / self.scale, cover_logits.masked_fill(~mask, -torch.inf)
