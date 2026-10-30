"""Small masked fusion building block; no pretrained encoder or quality claim."""

import torch
from torch import nn


class MaskedModalityFusion(nn.Module):
    """Project named feature vectors then learn a masked mixture.

    Input tensors are [batch,width]. Availability is an explicit [batch] bool
    tensor per configured modality. Callers establish timing/payload validity
    upstream. This vector control does not itself preserve within-stream time.
    """

    def __init__(self, dimensions, width=64):
        super().__init__()
        if not dimensions or width < 1 or any(d < 1 for d in dimensions.values()):
            raise ValueError("positive feature dimensions required")
        self.names = tuple(sorted(dimensions))
        self.dimensions = dict(dimensions)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dimensions[name], width) for name in self.names})
        self.gate = nn.Linear(width, 1)

    def forward(self, features, available):
        if set(features) != set(self.names) or set(available) != set(self.names):
            raise ValueError("all configured feature slots and explicit masks required")
        projected, masks = [], []
        batch_size = None
        for name in self.names:
            x, mask = features[name], available[name]
            if x.ndim != 2 or x.shape[1] != self.dimensions[name]:
                raise ValueError("feature shape mismatch")
            if mask.dtype != torch.bool or mask.shape != x.shape[:1]:
                raise ValueError("boolean batch mask required")
            if batch_size is not None and len(x) != batch_size:
                raise ValueError("batch sizes differ")
            batch_size = len(x)
            if not torch.isfinite(x[mask]).all():
                raise ValueError("nonfinite available feature")
            # Mask before projection: NaN/Inf in absent slots must not propagate.
            clean = torch.where(mask[:, None], x, torch.zeros_like(x))
            projected.append(torch.tanh(self.projections[name](clean)))
            masks.append(mask)
        valid = torch.stack(masks, dim=1)
        if not valid.any(dim=1).all():
            raise ValueError("all modalities missing; caller must abstain")
        tokens = torch.stack(projected, dim=1)
        logits = self.gate(tokens).squeeze(-1).masked_fill(~valid, float('-inf'))
        weights = logits.softmax(dim=1)
        fused = (tokens * weights[..., None]).sum(dim=1)
        return fused, weights
