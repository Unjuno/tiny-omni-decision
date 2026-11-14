"""Small option-conditioned temporal readout for frozen VideoPrism tokens."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CaterTemporalDecisionHead(nn.Module):
    """Attend to frozen spatial tokens per option, then read their time sequence."""

    def __init__(
        self,
        *,
        text_dim: int = 384,
        video_dim: int = 768,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        self.question_projection = nn.Linear(text_dim, hidden_dim)
        self.option_projection = nn.Linear(text_dim, hidden_dim)
        self.query_projection = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh())
        self.key_projection = nn.Linear(video_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(video_dim, hidden_dim)
        self.temporal_reader = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        video_tokens: Tensor,
        question_embedding: Tensor,
        option_embeddings: Tensor,
        start_frame: int,
        end_frame: int,
    ) -> Tensor:
        if video_tokens.ndim != 3 or video_tokens.shape[2] != self.video_dim:
            raise ValueError("video tokens must have shape [frames, spatial_tokens, video_dim]")
        if question_embedding.shape != (self.text_dim,):
            raise ValueError("question embedding has the wrong shape")
        if option_embeddings.ndim != 2 or option_embeddings.shape[1] != self.text_dim:
            raise ValueError("option embeddings must have shape [options, text_dim]")
        if option_embeddings.shape[0] < 2:
            raise ValueError("decision requires at least two options")
        if not 0 <= start_frame < end_frame < video_tokens.shape[0]:
            raise ValueError("invalid temporal frame range")

        question = self.question_projection(question_embedding)
        options = self.option_projection(option_embeddings)
        query = self.query_projection(torch.cat([question.expand_as(options), options], dim=-1))
        interval_tokens = video_tokens[start_frame : end_frame + 1]
        keys = self.key_projection(interval_tokens)
        values = self.value_projection(interval_tokens)
        attention = torch.einsum("od,tnd->otn", query, keys) / math.sqrt(self.hidden_dim)
        attention = attention.softmax(dim=-1)
        per_frame = torch.einsum("otn,tnd->otd", attention, values)
        sequence, _ = self.temporal_reader(per_frame)
        first_state = sequence[:, 0]
        last_state = sequence[:, -1]
        state_delta = last_state - first_state
        features = torch.cat(
            [last_state, first_state, state_delta, question.expand_as(options), options], dim=-1
        )
        return self.scorer(features).squeeze(-1)
