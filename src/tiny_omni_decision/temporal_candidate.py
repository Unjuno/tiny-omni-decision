"""Small candidate scorer over frozen text and ordered video features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalCandidateScorer(nn.Module):
    """Score each candidate using a question embedding and ordered video tokens."""

    def __init__(
        self,
        *,
        text_feature_size: int = 3072,
        video_feature_size: int = 768,
        temporal_hidden_size: int = 128,
        scorer_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if min(text_feature_size, video_feature_size, temporal_hidden_size, scorer_hidden_size) < 1:
            raise ValueError("feature and hidden sizes must be positive")
        self.video_feature_size = video_feature_size
        self.text_feature_size = text_feature_size
        self.temporal = nn.LSTM(
            input_size=video_feature_size,
            hidden_size=temporal_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(text_feature_size + 2 * temporal_hidden_size, scorer_hidden_size),
            nn.GELU(),
            nn.Linear(scorer_hidden_size, 1),
        )

    def forward(
        self,
        video: Tensor,
        candidate_features: Tensor,
        option_mask: Tensor | None = None,
    ) -> Tensor:
        if (
            video.ndim != 3
            or candidate_features.ndim != 3
            or video.shape[0] != candidate_features.shape[0]
            or video.shape[-1] != self.video_feature_size
            or candidate_features.shape[-1] != self.text_feature_size
            or candidate_features.shape[1] < 2
        ):
            raise ValueError(
                "expected video [batch,time,features] and candidates [batch,options,features]"
            )
        if not torch.isfinite(video).all() or not torch.isfinite(candidate_features).all():
            raise ValueError("video and candidate features must be finite")
        _, (hidden, _) = self.temporal(video)
        video_context = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        context = video_context.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        logits = self.scorer(torch.cat((context, candidate_features), dim=-1)).squeeze(-1)
        if option_mask is not None:
            if option_mask.shape != logits.shape or option_mask.dtype != torch.bool:
                raise ValueError("option_mask must be a boolean tensor matching the logits")
            if not option_mask.any(dim=1).all():
                raise ValueError("every example must have at least one valid option")
            logits = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)
        return logits


def temporal_candidate_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
