"""Shared nonlinear spatial-cell aggregation with supplied candidate scoring."""

import math

import torch
from torch import nn


class CellwiseCandidateScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cells = nn.Linear(384, 128)
        self.cls = nn.Linear(384, 128)
        self.observation = nn.Linear(256, 128)
        self.candidate = nn.Linear(384, 128)

    def forward(self, features):
        if features.ndim < 2 or features.shape[-1] != 6912:
            raise ValueError("expected CLS384 + sixteen scaled cells384 + candidate384")
        # Existing cache readout scales flattened cells by1/4. Restore before shared map.
        cells = features[..., 384:6528].reshape(*features.shape[:-1], 16, 384) * 4
        pooled = torch.nn.functional.gelu(self.cells(cells)).sum(-2) / 4
        cls = torch.nn.functional.gelu(self.cls(features[..., :384]))
        visual = self.observation(torch.cat([cls, pooled], -1))
        candidate = self.candidate(features[..., 6528:])
        return (visual * candidate).sum(-1) / math.sqrt(128)
