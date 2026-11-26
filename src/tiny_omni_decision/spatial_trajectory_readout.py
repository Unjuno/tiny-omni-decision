"""Small candidate-conditioned readout over frozen trajectory/spatial context."""

import math

import torch
from torch import nn

from .cover_point_readout import CoverPointReadout


class SpatialTrajectoryReadout(nn.Module):
    """Transient state reuse under fixed weights; no persistent cache claim."""

    def __init__(self, hidden=64, latent_count=16, heads=4):
        super().__init__()
        if min(hidden, latent_count, heads) < 1 or hidden % heads:
            raise ValueError("positive latent dimensions and divisible attention width required")
        self.hidden = hidden
        self.cover = CoverPointReadout(hidden=hidden)
        self.context_projection = nn.Linear(387, hidden)
        self.latents = nn.Parameter(torch.randn(latent_count, hidden) * 0.02)
        self.context_attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden)
        self.context_ff = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
        )
        self.context_ff_norm = nn.LayerNorm(hidden)
        self.point_attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.point_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(1536, hidden)
        self.query_attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden)
        self.value = nn.Linear(hidden, hidden)
        self.scale = math.sqrt(hidden)

    def encode_state(self, points, context, mask=None, context_mask=None):
        features, mask = self.cover.encode(points, mask)
        if (
            context.ndim != 3
            or context.shape[0] != points.shape[0]
            or context.shape[1] == 0
            or context.shape[2] != 387
            or context.dtype != self.context_projection.weight.dtype
            or not torch.isfinite(context).all()
        ):
            raise ValueError("aligned nonempty finite387-d spatial context required")
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        if (
            context_mask.shape != context.shape[:2]
            or context_mask.dtype != torch.bool
            or not context_mask.any(1).all()
        ):
            raise ValueError(
                "aligned Boolean spatial mask with at least one token per clip required"
            )
        projected = self.context_projection(context)
        seeds = self.latents[None].expand(len(points), -1, -1)
        observed, _ = self.context_attention(
            seeds, projected, projected, key_padding_mask=~context_mask, need_weights=False
        )
        latent = self.context_norm(seeds + observed)
        latent = self.context_ff_norm(latent + self.context_ff(latent))
        seen, _ = self.point_attention(features, latent, latent, need_weights=False)
        features = self.point_norm(features + seen)
        cover = self.cover.score(features).squeeze(-1).masked_fill(~mask, -torch.inf)
        return features, cover, latent, mask

    def score_candidates(self, state, candidates):
        features, cover, latent, mask = state
        if (
            candidates.ndim != 3
            or candidates.shape[0] != features.shape[0]
            or candidates.shape[1] == 0
            or candidates.shape[2] != 1536
            or candidates.dtype != self.query.weight.dtype
            or not torch.isfinite(candidates).all()
        ):
            raise ValueError("aligned nonempty finite supplied-candidate features required")
        query = self.query(candidates)
        seen, _ = self.query_attention(query, latent, latent, need_weights=False)
        query = self.query_norm(query + seen)
        scores = torch.einsum("bch,bnh->bcn", query, features) / self.scale + cover[:, None]
        weights = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        values = torch.einsum("bcn,bnh->bch", weights, self.value(features))
        return (query * values).sum(-1) / self.scale

    def forward(self, points, candidates, context, mask=None, context_mask=None):
        state = self.encode_state(points, context, mask, context_mask)
        return self.score_candidates(state, candidates), state[1]
