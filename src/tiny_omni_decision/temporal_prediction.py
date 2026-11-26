"""Small causal predictors over frozen frame-level video representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class NextFrameFeaturePredictor(nn.Module):
    """Predict each next frozen feature from the preceding feature sequence."""

    def __init__(self, feature_width: int = 768, hidden_width: int = 64) -> None:
        super().__init__()
        if (
            type(feature_width) is not int
            or type(hidden_width) is not int
            or feature_width < 1
            or hidden_width < 1
        ):
            raise ValueError("feature_width and hidden_width must be positive integers")
        self.feature_width = feature_width
        self.hidden_width = hidden_width
        self.input_norm = nn.LayerNorm(feature_width)
        self.temporal = nn.LSTM(feature_width, hidden_width, batch_first=True)
        self.output = nn.Linear(hidden_width, feature_width)

    def forward(self, features: Tensor) -> Tensor:
        """Return `[batch, frames - 1, width]` causal next-feature predictions."""
        reference = self.output.weight
        if (
            not isinstance(features, Tensor)
            or features.ndim != 3
            or features.shape[0] < 1
            or features.shape[1] < 2
            or features.shape[2] != self.feature_width
            or features.dtype != reference.dtype
            or features.device != reference.device
            or not torch.isfinite(features).all()
        ):
            raise ValueError("finite [batch, at least 2 frames, feature_width] input required")
        context = self.input_norm(features[:, :-1])
        hidden, _ = self.temporal(context)
        return self.output(hidden)


def cosine_prediction_loss(predictions: Tensor, features: Tensor) -> Tensor:
    """Mean cosine distance to the following frozen feature at every step."""
    if (
        not isinstance(predictions, Tensor)
        or not isinstance(features, Tensor)
        or features.ndim != 3
        or predictions.shape != (features.shape[0], features.shape[1] - 1, features.shape[2])
        or features.shape[0] < 1
        or features.shape[1] < 2
        or features.dtype != predictions.dtype
        or features.device != predictions.device
        or not torch.isfinite(features).all()
        or not torch.isfinite(predictions).all()
    ):
        raise ValueError("predictions and finite feature sequences have incompatible shapes")
    targets = F.normalize(features[:, 1:], dim=-1)
    predictions = F.normalize(predictions, dim=-1)
    return (1.0 - F.cosine_similarity(predictions, targets, dim=-1)).mean()


def per_sequence_cosine_error(predictions: Tensor, features: Tensor) -> Tensor:
    """Return one uncalibrated temporal surprise score per video sequence."""
    if (
        not isinstance(predictions, Tensor)
        or not isinstance(features, Tensor)
        or features.ndim != 3
        or predictions.shape != (features.shape[0], features.shape[1] - 1, features.shape[2])
        or features.shape[0] < 1
        or features.shape[1] < 2
        or features.dtype != predictions.dtype
        or features.device != predictions.device
        or not torch.isfinite(features).all()
        or not torch.isfinite(predictions).all()
    ):
        raise ValueError("predictions and finite feature sequences have incompatible shapes")
    targets = F.normalize(features[:, 1:], dim=-1)
    predictions = F.normalize(predictions, dim=-1)
    return (1.0 - F.cosine_similarity(predictions, targets, dim=-1)).mean(dim=1)
