"""Experimental vector fusion and option scoring; encoders remain external."""

import torch
from torch import nn

from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.masked_fusion import MaskedModalityFusion


class FusionDecisionScorer(nn.Module):
    """Shared scorer without option positions or a mandatory text criterion.

    Features: named [B,D] vectors; options: [B,K,option_dim]; criterion: [B,C].
    Typed/semantic embeddings are supplied by the caller. This module neither
    interprets IDs nor encodes raw media. Time/unit validation happens upstream.
    A missing required source still needs the external evidence/abstention gate.
    """

    def __init__(self, dimensions, option_dim, criterion_dim, width=64):
        super().__init__()
        if option_dim < 1 or criterion_dim < 1:
            raise ValueError("positive option and criterion dimensions required")
        self.option_dim, self.criterion_dim = option_dim, criterion_dim
        self.fusion = MaskedModalityFusion(dimensions, width)
        # Presence is observable separately from zero-valued evidence/criterion.
        size = width + len(dimensions) + option_dim + criterion_dim + 1
        self.scorer = FrozenFeatureCandidateScorer(size, width)

    def forward(self, features, available, options, option_valid,
                criterion=None, criterion_present=None):
        if options.ndim != 3 or options.shape[-1] != self.option_dim:
            raise ValueError("options must have shape [batch, candidates, option_dim]")
        if option_valid.dtype != torch.bool or option_valid.shape != options.shape[:2]:
            raise ValueError("boolean option mask required")
        if not (option_valid.sum(-1) >= 2).all():
            raise ValueError("at least two available candidates required per row")
        if not torch.isfinite(options[option_valid]).all():
            raise ValueError("nonfinite available option")
        batch, count = options.shape[:2]
        if criterion is None and criterion_present is None:
            criterion = options.new_zeros((batch, self.criterion_dim))
            criterion_present = torch.zeros(batch, dtype=torch.bool, device=options.device)
        elif criterion is None or criterion_present is None:
            raise ValueError("criterion values and explicit presence must be supplied together")
        if (criterion.shape != (batch, self.criterion_dim)
                or criterion_present.shape != (batch,) or criterion_present.dtype != torch.bool):
            raise ValueError("criterion shape or mask mismatch")
        if not torch.isfinite(criterion[criterion_present]).all():
            raise ValueError("nonfinite present criterion")
        fused, _ = self.fusion(features, available)
        if fused.shape[0] != batch:
            raise ValueError("feature and option batch sizes differ")
        presence = torch.stack([available[n] for n in self.fusion.names], -1).to(fused.dtype)
        clean_criterion = torch.where(criterion_present[:, None], criterion,
                                      torch.zeros_like(criterion))
        state = torch.cat((fused, presence, clean_criterion,
                           criterion_present[:, None].to(fused.dtype)), -1)
        clean_options = torch.where(option_valid[..., None], options, torch.zeros_like(options))
        candidate_features = torch.cat((state[:, None].expand(-1, count, -1), clean_options), -1)
        logits = self.scorer(candidate_features)
        if not torch.isfinite(logits[option_valid]).all():
            raise ValueError("nonfinite decision logits")
        return logits.masked_fill(~option_valid, float("-inf"))
