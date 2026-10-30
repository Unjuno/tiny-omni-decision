"""A small global channel mixer for frozen MOMENT features."""

import torch
from torch import nn

from scripts.train_egoviz_vs_sync_readout import JointTemporalScorer


class ChannelWeightedScorer(JointTemporalScorer):
    def __init__(self):
        super().__init__()
        self.channel_logits = nn.Parameter(torch.zeros(6, 512))

    def forward(self, video: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        weights = self.channel_logits.softmax(dim=0)
        sensor = torch.einsum("bocpd,cd->bopd", candidates, weights)
        sensor = sensor.reshape(-1, 4, 8, 2, 512).mean(dim=3)
        video_embedding = nn.functional.normalize(self.video_projection(video), dim=-1)
        sensor_embedding = nn.functional.normalize(self.sensor_projection(sensor), dim=-1)
        return (video_embedding[:, None] * sensor_embedding).sum(dim=-1).mean(dim=-1)
