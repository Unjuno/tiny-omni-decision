"""Small frozen-feature Decision readouts for episode-level AV matching."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BilinearAVMatchScorer(nn.Module):
    """Map audio and video summaries into a shared low-rank match space."""

    def __init__(self, *, audio_width: int = 768, video_width: int = 1536, rank: int = 32):
        super().__init__()
        if min(audio_width, video_width, rank) < 1:
            raise ValueError("feature widths and rank must be positive")
        self.audio_width = audio_width
        self.video_width = video_width
        self.rank = rank
        self.audio_projection = nn.Linear(audio_width, rank, bias=False)
        self.video_projection = nn.Linear(video_width, rank, bias=False)
        nn.init.xavier_uniform_(self.audio_projection.weight)
        nn.init.xavier_uniform_(self.video_projection.weight)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, audio: Tensor, video: Tensor) -> Tensor:
        if (
            audio.ndim != 2
            or video.ndim != 2
            or audio.shape[0] != video.shape[0]
            or audio.shape[1] != self.audio_width
            or video.shape[1] != self.video_width
        ):
            raise ValueError("audio and video summaries have invalid batch or feature widths")
        if (
            audio.device != video.device
            or audio.dtype != video.dtype
            or not audio.is_floating_point()
            or not torch.isfinite(audio).all()
            or not torch.isfinite(video).all()
        ):
            raise ValueError("audio and video summaries must be finite matching float tensors")
        audio_latent = F.normalize(self.audio_projection(audio), p=2, dim=-1)
        video_latent = F.normalize(self.video_projection(video), p=2, dim=-1)
        match_score = (audio_latent * video_latent).sum(-1) * math.sqrt(self.rank)
        return torch.stack((-match_score, match_score), dim=-1)


class SingleModalityMatchScorer(nn.Module):
    """Logistic unimodal control for the balanced AV-match task."""

    def __init__(self, feature_width: int):
        super().__init__()
        if feature_width < 1:
            raise ValueError("feature width must be positive")
        self.feature_width = feature_width
        self.projection = nn.Linear(feature_width, 1)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_width:
            raise ValueError("unimodal feature tensor has invalid shape")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("unimodal features must be finite floating-point values")
        score = self.projection(features).squeeze(-1)
        return torch.stack((-score, score), dim=-1)
