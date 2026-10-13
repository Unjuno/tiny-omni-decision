"""One small slot-time interaction block before the existing candidate readout."""

import torch
from torch import nn

from tiny_omni_decision.trajectory_readout import TrajectoryReadout


class RelationalSlotReadout(TrajectoryReadout):
    def __init__(self, input_dim=129, hidden=64):
        super().__init__(hidden=hidden, input_dim=input_dim)
        self.project = nn.Linear(input_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.interaction = nn.MultiheadAttention(hidden, 4, dropout=0, batch_first=True)
        self.restore = nn.Linear(hidden, input_dim)

    def forward(self, points, candidates, mask=None):
        if points.ndim != 3 or points.shape[-1] != self.input_dim or points.shape[1] == 0:
            raise ValueError("nonempty aligned slot features required")
        if not torch.isfinite(points).all():
            raise ValueError("nonfinite inputs")
        if mask is None:
            mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if mask.shape != points.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all():
            raise ValueError("aligned mask with at least one point required")
        latent = self.norm(self.project(points))
        mixed, _ = self.interaction(latent, latent, latent, key_padding_mask=~mask,
                                    need_weights=False)
        contextual = points + self.restore(mixed)
        return super().forward(contextual, candidates, mask)
