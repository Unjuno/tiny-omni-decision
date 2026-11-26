"""Low-rank interaction of frozen observation and question/candidate features."""

import math

from torch import nn


class BilinearCandidateScorer(nn.Module):
    def __init__(self, observation_size, candidate_size, hidden_size=128):
        super().__init__()
        if min(observation_size, candidate_size, hidden_size) < 1:
            raise ValueError("positive dimensions required")
        self.observation_size = observation_size
        self.feature_size = observation_size + candidate_size
        self.observation = nn.Linear(observation_size, hidden_size)
        self.candidate = nn.Linear(candidate_size, hidden_size)
        self.scale = math.sqrt(hidden_size)

    def forward(self, features):
        if features.ndim < 2 or features.shape[-1] != self.feature_size:
            raise ValueError("candidate feature dimensions differ")
        visual = self.observation(features[..., : self.observation_size])
        text = self.candidate(features[..., self.observation_size :])
        return (visual * text).sum(-1) / self.scale
