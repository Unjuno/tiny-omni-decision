"""Causal timestamp alignment and gated fusion for cached feature tokens."""

from __future__ import annotations

import math

import torch
from torch import nn


class TimestampedModalityFusion(nn.Module):
    """Align the latest non-stale token from each modality at query times.

    Inputs are frozen-feature sequences `[batch, time, width]` with explicit
    timestamps and validity masks. Alignment is causal: future tokens are never
    selected. The returned per-modality masks let callers abstain when a task's
    required inputs are unavailable. This module does not encode raw media or
    model dynamics between query steps.
    """

    def __init__(self, dimensions: dict[str, int], width: int = 64):
        super().__init__()
        if (
            not dimensions
            or type(width) is not int
            or width < 1
            or any(type(size) is not int or size < 1 for size in dimensions.values())
        ):
            raise ValueError("positive feature dimensions and fusion width required")
        self.names = tuple(sorted(dimensions))
        self.dimensions = dict(dimensions)
        self.projections = nn.ModuleDict({
            name: nn.Linear(self.dimensions[name] + 1, width) for name in self.names
        })
        self.gate = nn.Linear(width, 1)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        timestamps_ms: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        query_times_ms: torch.Tensor,
        *,
        max_age_ms: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return fused query tokens, modality weights, and aligned masks."""
        if not (
            set(features) == set(self.names)
            and set(timestamps_ms) == set(self.names)
            and set(available) == set(self.names)
        ):
            raise ValueError("all configured feature, timestamp and mask slots required")
        if (
            isinstance(max_age_ms, bool)
            or not isinstance(max_age_ms, (int, float))
            or not math.isfinite(max_age_ms)
            or max_age_ms < 0
        ):
            raise ValueError("max_age_ms must be finite and nonnegative")

        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        queries = torch.as_tensor(query_times_ms, device=device, dtype=dtype)
        if queries.ndim != 2 or queries.shape[1] < 1 or not torch.isfinite(queries).all():
            raise ValueError("finite query timestamps with shape [batch, queries] required")
        if queries.shape[1] > 1 and not (queries[:, 1:] > queries[:, :-1]).all():
            raise ValueError("query timestamps must strictly increase")

        projected = []
        aligned_masks = {}
        batch_size = queries.shape[0]
        for name in self.names:
            x = torch.as_tensor(features[name], device=device, dtype=dtype)
            stamps = torch.as_tensor(timestamps_ms[name], device=device, dtype=dtype)
            mask = torch.as_tensor(available[name], device=device)
            if x.ndim != 3 or x.shape[0] != batch_size or x.shape[2] != self.dimensions[name]:
                raise ValueError("feature shape mismatch")
            if stamps.shape != x.shape[:2] or mask.shape != x.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("timestamp or boolean validity mask shape mismatch")
            if x.shape[1] < 1 or not torch.isfinite(stamps).all():
                raise ValueError("feature timestamps must be nonempty and finite")
            if x.shape[1] > 1 and not (stamps[:, 1:] > stamps[:, :-1]).all():
                raise ValueError("feature timestamps must strictly increase")
            if not torch.isfinite(x[mask]).all():
                raise ValueError("nonfinite available feature")

            age = queries[:, None, :] - stamps[:, :, None]
            eligible = mask[:, :, None] & (age >= 0) & (age <= max_age_ms)
            present = eligible.any(dim=1)
            # Timestamps are strictly increasing, so the largest eligible
            # timestep is the latest causally available observation.
            timestep_indices = torch.arange(x.shape[1], device=device).view(1, -1, 1)
            indices = timestep_indices.expand_as(eligible).masked_fill(~eligible, -1)
            indices = indices.amax(dim=1).clamp_min(0)
            selected = x.gather(
                1, indices[:, :, None].expand(-1, -1, x.shape[-1])
            )
            selected_time = stamps.gather(1, indices)
            selected_age = torch.where(
                present,
                (queries - selected_time).clamp_min(0),
                torch.zeros_like(queries),
            )
            age_scale = max(float(max_age_ms), 1.0)
            age_feature = (selected_age / age_scale).unsqueeze(-1)
            clean = torch.where(present[:, :, None], selected, torch.zeros_like(selected))
            projected_feature = torch.cat((clean, age_feature), -1)
            projected.append(torch.tanh(self.projections[name](projected_feature)))
            aligned_masks[name] = present

        valid = torch.stack([aligned_masks[name] for name in self.names], dim=-1)
        tokens = torch.stack(projected, dim=-2)
        logits = self.gate(tokens).squeeze(-1).masked_fill(~valid, float("-inf"))
        any_valid = valid.any(dim=-1, keepdim=True)
        safe_logits = torch.where(any_valid, logits, torch.zeros_like(logits))
        weights = safe_logits.softmax(dim=-1) * valid.to(dtype)
        fused = (tokens * weights.unsqueeze(-1)).sum(dim=-2)
        return fused, weights, aligned_masks
