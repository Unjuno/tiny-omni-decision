"""Tiny frozen-region correspondence scorer, not the Omni product model."""

import torch
from torch import nn


class MaskPairReadout(nn.Module):
    def __init__(self, feature_dim, *, hidden=64):
        super().__init__()
        if type(feature_dim) is not int or type(hidden) is not int or min(feature_dim, hidden) < 1:
            raise ValueError("positive pair and hidden widths required")
        self.feature_dim = feature_dim
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, features):
        if (
            features.ndim != 2
            or features.shape[1] != self.feature_dim
            or len(features) == 0
            or features.dtype != self.net[0].weight.dtype
            or not torch.isfinite(features).all()
        ):
            raise ValueError("nonempty aligned finite pair features required")
        same = self.net(features).squeeze(1)
        return torch.stack((torch.zeros_like(same), same), 1)
