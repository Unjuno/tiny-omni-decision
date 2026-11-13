"""Compact shared-projection scorer for audio-to-video matching choices."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AVMatchReadout(nn.Module):
    """Score each candidate video against one audio query with a low-rank bilinear lens."""

    def __init__(self, *, rank: int = 8, feature_width: int = 768):
        super().__init__()
        if rank < 1 or feature_width < 1:
            raise ValueError("rank and feature_width must be positive")
        self.rank = rank
        self.feature_width = feature_width
        self.audio_projection = nn.Linear(feature_width, rank, bias=False)
        self.video_projection = nn.Linear(feature_width, rank, bias=False)
        nn.init.xavier_uniform_(self.audio_projection.weight)
        nn.init.xavier_uniform_(self.video_projection.weight)

    def forward(self, audio: Tensor, candidate_videos: Tensor) -> Tensor:
        if (
            audio.ndim != 2
            or candidate_videos.ndim != 3
            or audio.shape[0] != candidate_videos.shape[0]
            or audio.shape[1] != self.feature_width
            or candidate_videos.shape[1] != 4
            or candidate_videos.shape[2] != self.feature_width
        ):
            raise ValueError(
                "expected audio [batch, feature_width] and videos [batch, 4, feature_width] shape"
            )
        if not torch.isfinite(audio).all() or not torch.isfinite(candidate_videos).all():
            raise ValueError("audio/video features must be finite")
        audio_latent = F.normalize(self.audio_projection(audio), p=2, dim=-1)
        video_latent = F.normalize(self.video_projection(candidate_videos), p=2, dim=-1)
        return (audio_latent[:, None, :] * video_latent).sum(dim=-1)
