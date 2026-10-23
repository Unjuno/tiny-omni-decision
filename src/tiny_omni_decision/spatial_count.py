"""Small option-conditioned readout over frozen spatial video features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def pool_video_tokens(tokens: Tensor, output_grid_size: int = 8) -> Tensor:
    """Average the repeated-frame time axis, retaining an ordered spatial grid."""
    if tokens.ndim != 3 or output_grid_size < 1:
        raise ValueError("tokens must be [batch, tokens, channels] and grid size positive")
    batch, token_count, channels = tokens.shape
    if token_count != 8 * 16 * 16:
        raise ValueError("expected VideoPrism tokens for eight 16x16 repeated-frame grids")
    if not torch.isfinite(tokens).all():
        raise ValueError("VideoPrism tokens must be finite")
    grids = tokens.reshape(batch, 8, 16, 16, channels).mean(dim=1)
    channels_first = grids.permute(0, 3, 1, 2).contiguous()
    pooled = F.adaptive_avg_pool2d(channels_first, (output_grid_size, output_grid_size))
    return pooled.flatten(2).transpose(1, 2).contiguous()


class CandidateConditionedSpatialScorer(nn.Module):
    """Attend to frozen spatial cells separately for each supplied option."""

    def __init__(self, visual_size: int = 768, option_size: int = 384, hidden_size: int = 128):
        super().__init__()
        self.key = nn.Linear(visual_size, hidden_size, bias=False)
        self.value = nn.Linear(visual_size, hidden_size, bias=False)
        self.query = nn.Linear(option_size, hidden_size, bias=False)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size + option_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, visual: Tensor, options: Tensor) -> Tensor:
        if visual.ndim != 3 or options.ndim != 3:
            raise ValueError("visual and option inputs must be batched sequences")
        if visual.shape[0] != options.shape[0] or visual.shape[-1] != self.key.in_features:
            raise ValueError("visual input shape is incompatible with scorer")
        if options.shape[-1] != self.query.in_features or options.shape[1] < 2:
            raise ValueError("at least two compatible option vectors are required")
        if not torch.isfinite(visual).all() or not torch.isfinite(options).all():
            raise ValueError("scorer inputs must be finite")
        query = self.query(options)
        keys = self.key(visual)
        values = self.value(visual)
        attention_logits = torch.einsum("bnh,bkh->bnk", query, keys)
        weights = torch.softmax(attention_logits / (keys.shape[-1] ** 0.5), dim=-1)
        attended = torch.einsum("bnk,bkh->bnh", weights, values)
        return self.scorer(torch.cat((attended, options), dim=-1)).squeeze(-1)
