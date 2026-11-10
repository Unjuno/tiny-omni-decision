"""Small candidate-equivariant reader over frozen VideoMAE tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PerceptionVideoDecisionReadout(nn.Module):
    """Score each supplied option by attending to one frozen video token grid.

    Text encoders and the VideoMAE encoder stay outside this module. Each
    candidate query is encoded from a fixed question/candidate string and the
    video memory is a per-video `[tokens, 384]` grid. No option-position
    embedding is used, so permuting candidates only permutes their scores.
    """

    def __init__(self, *, input_size: int = 384, hidden_size: int = 192) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 4 or hidden_size % 4:
            raise ValueError("input size must be positive and hidden size divisible by four")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.video_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, dropout=0.0, batch_first=True
        )
        self.scorer = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, video_tokens: Tensor, candidate_queries: Tensor) -> Tensor:
        if (
            video_tokens.ndim != 3
            or candidate_queries.ndim != 3
            or video_tokens.shape[0] != candidate_queries.shape[0]
            or video_tokens.shape[1] < 1
            or candidate_queries.shape[1] < 2
            or video_tokens.shape[2] != self.input_size
            or candidate_queries.shape[2] != self.input_size
        ):
            raise ValueError(
                "expected video [batch,tokens,input] and queries [batch,options,input]"
            )
        if (
            video_tokens.device != candidate_queries.device
            or video_tokens.dtype != candidate_queries.dtype
            or video_tokens.device != self.video_projection[0].weight.device
            or video_tokens.dtype != self.video_projection[0].weight.dtype
        ):
            raise ValueError("inputs must match each other and the readout device/dtype")
        if not torch.isfinite(video_tokens).all() or not torch.isfinite(candidate_queries).all():
            raise ValueError("video tokens and candidate queries must be finite")

        batch = video_tokens.shape[0]
        option_count = candidate_queries.shape[1]
        memory = self.video_projection(video_tokens)
        query = self.query_projection(candidate_queries)
        query_flat = query.reshape(batch * option_count, 1, self.hidden_size)
        memory_flat = memory.repeat_interleave(option_count, dim=0)
        attended, _ = self.attention(query_flat, memory_flat, memory_flat, need_weights=False)
        attended = attended.reshape(batch, option_count, self.hidden_size)
        features = torch.cat([query, attended, query * attended, (query - attended).abs()], dim=-1)
        logits = self.scorer(features).squeeze(-1)
        if logits.shape != (batch, option_count) or not torch.isfinite(logits).all():
            raise ValueError("readout produced invalid candidate logits")
        return logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
