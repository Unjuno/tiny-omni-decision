"""Candidate-conditioned readout of frozen VideoPrism space/time tokens."""

import math

import torch
from torch import nn


class VideoTokenReadout(nn.Module):
    def __init__(self, mode="attention", hidden=128):
        super().__init__()
        if mode not in {"attention", "mean", "no_video"}:
            raise ValueError("unknown readout mode")
        self.mode = mode
        self.query = nn.Linear(1536, hidden)
        self.key = nn.Linear(768, hidden)
        self.value = nn.Linear(768, hidden)
        self.position = nn.Linear(3, hidden, bias=False)
        self.scale = math.sqrt(hidden)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, 8),
                torch.linspace(-1, 1, 16),
                torch.linspace(-1, 1, 16),
                indexing="ij",
            ),
            -1,
        ).reshape(2048, 3)
        self.register_buffer("coordinates", coords)

    def forward(self, tokens, candidates):
        if tokens.ndim != 3 or tokens.shape[1:] != (2048, 768):
            raise ValueError("expected [batch,2048,768] tokens")
        if (
            candidates.ndim != 3
            or candidates.shape[0] != tokens.shape[0]
            or candidates.shape[-1] != 1536
        ):
            raise ValueError("expected aligned candidate features")
        query = self.query(candidates)
        if self.mode == "attention":
            keys = self.key(tokens) + self.position(self.coordinates)[None]
            scores = torch.einsum("bch,bnh->bcn", query, keys) / self.scale
            attention = scores.softmax(-1)
            observed = torch.einsum("bcn,bnh->bch", attention, self.value(tokens))
        else:
            pooled = tokens.mean(1)
            if self.mode == "no_video":
                pooled = torch.zeros_like(pooled)
            observed = self.value(pooled)[:, None]
        return (observed * query).sum(-1) / self.scale
