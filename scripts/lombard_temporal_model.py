"""One fixed capacity for ordered/broadcast native audiovisual slot experiments."""

import torch
from torch import nn


class TemporalSlotHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(768, 64), nn.GELU())
        self.temporal = nn.GRU(64, 64, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(450, 64), nn.GELU(), nn.Linear(64, 51))

    def forward(self, audio, video, present):
        batch = len(audio)
        if (
            audio.shape != (batch, 384)
            or video.shape != (batch, 16, 768)
            or present.shape != (batch, 2)
            or not all(torch.isfinite(v).all() for v in (audio, video, present))
            or not ((present == 0) | (present == 1)).all()
        ):
            raise ValueError("finite native observation dimensions and explicit presence required")
        sequence, _ = self.temporal(self.visual(video))
        visual = sequence[:, -1] * present[:, 1:2]
        audio = audio * present[:, :1]
        return self.readout(torch.cat((audio, visual, present), dim=-1))
