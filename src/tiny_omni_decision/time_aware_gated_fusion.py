"""Small causal fusion for timestamp-aligned, frozen modality features.

The module consumes caller-projected feature vectors. It does not load an
encoder, align clocks, or infer missingness. Those responsibilities stay in
the observation/cache layer so timing assumptions remain auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


@dataclass(frozen=True)
class TimeAwareFusionOutput:
    """Fused session state and per-step diagnostics in input time order."""

    fused: Tensor
    states: Tensor
    modality_weights: Tensor
    update_gates: Tensor


class TimeAwareGatedFusion(nn.Module):
    """Fuse causally aligned modality snapshots with a small recurrent gate.

    Args:
        hidden_size: Width shared by caller-projected modality features.
        modality_count: Number of modality slots in the frozen input contract.
        time_scale_seconds: Explicit scale used for log-relative time features.

    Inputs use ``features[B,T,M,H]``, ``timestamps_seconds[B,T]`` and a boolean
    ``available[B,T,M]``. Each T row is an as-of snapshot on one common clock;
    align asynchronous streams before calling this module. Valid timestamps
    must be nondecreasing. Padding rows have every modality unavailable.
    """

    def __init__(
        self,
        hidden_size: int,
        modality_count: int,
        time_scale_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or modality_count < 1:
            raise ValueError("hidden_size and modality_count must be positive")
        if time_scale_seconds <= 0 or not torch.isfinite(torch.tensor(time_scale_seconds)):
            raise ValueError("time_scale_seconds must be finite and positive")
        self.hidden_size = hidden_size
        self.modality_count = modality_count
        self.time_scale_seconds = float(time_scale_seconds)
        self.modality_embedding = nn.Embedding(modality_count, hidden_size)
        self.modality_gate = nn.Linear(hidden_size, 1, bias=False)
        self.time_projection = nn.Linear(2, hidden_size, bias=False)
        recurrent_input_size = hidden_size * 2 + 2
        self.update_gate = nn.Linear(recurrent_input_size, hidden_size)
        self.candidate = nn.Linear(recurrent_input_size, hidden_size)

    def forward(
        self,
        features: Tensor,
        timestamps_seconds: Tensor,
        available: Tensor,
    ) -> TimeAwareFusionOutput:
        if features.ndim != 4:
            raise ValueError("features must have shape [batch, time, modality, hidden]")
        batch, steps, modalities, width = features.shape
        if batch < 1 or steps < 1 or modalities != self.modality_count:
            raise ValueError("feature batch/time dimensions or modality count are invalid")
        if width != self.hidden_size:
            raise ValueError(f"feature width must be {self.hidden_size}")
        if timestamps_seconds.shape != (batch, steps):
            raise ValueError("timestamps_seconds must have shape [batch, time]")
        if available.shape != (batch, steps, modalities) or available.dtype != torch.bool:
            raise ValueError("available must be a boolean [batch, time, modality] mask")
        if timestamps_seconds.device != features.device or available.device != features.device:
            raise ValueError("features, timestamps and availability must share a device")
        if not features.is_floating_point():
            raise ValueError("features must use a floating-point dtype")
        if features.dtype != self.modality_embedding.weight.dtype:
            raise ValueError("feature dtype must match the fusion module parameter dtype")
        if not available.any(dim=(1, 2)).all():
            raise ValueError("each sample needs at least one available modality")
        if not torch.isfinite(features[available]).all():
            raise ValueError("available modality features must be finite")

        step_mask = available.any(dim=-1)
        # Keep subsecond deltas when features/weights use BF16 or FP16.
        timestamps = timestamps_seconds.to(dtype=torch.float32)
        if not torch.isfinite(timestamps[step_mask]).all():
            raise ValueError("timestamps for available steps must be finite")
        for row in range(batch):
            valid_times = timestamps[row, step_mask[row]]
            if (valid_times[1:] < valid_times[:-1]).any():
                raise ValueError("available observation timestamps must be nondecreasing")

        modality_ids = torch.arange(modalities, device=features.device)
        embeddings = self.modality_embedding(modality_ids)
        state = features.new_zeros((batch, self.hidden_size))
        first_time = features.new_zeros(batch)
        previous_time = features.new_zeros(batch)
        have_time = torch.zeros(batch, dtype=torch.bool, device=features.device)
        state_rows: list[Tensor] = []
        weight_rows: list[Tensor] = []
        gate_rows: list[Tensor] = []

        for step in range(steps):
            present = available[:, step]
            active = step_mask[:, step]
            time = timestamps[:, step]
            safe_time = torch.where(active, time, previous_time)
            first_time = torch.where(active & ~have_time, safe_time, first_time)
            delta = torch.where(active & have_time, safe_time - previous_time, 0.0)
            elapsed = torch.where(active, safe_time - first_time, 0.0)
            time_features = torch.stack(
                (
                    torch.log1p(delta.clamp_min(0) / self.time_scale_seconds),
                    torch.log1p(elapsed.clamp_min(0) / self.time_scale_seconds),
                ),
                dim=-1,
            ).to(dtype=features.dtype)

            safe_features = torch.where(
                present[..., None], features[:, step], torch.zeros_like(features[:, step])
            )
            modality_tokens = safe_features + embeddings[None, :, :]
            modality_logits = self.modality_gate(modality_tokens).squeeze(-1)
            masked_logits = modality_logits.masked_fill(~present, float("-inf"))
            # All-missing padding rows receive a temporary finite softmax input;
            # their returned weights are then forced to zero.
            masked_logits = torch.where(active[:, None], masked_logits, 0.0)
            weights = masked_logits.softmax(dim=-1)
            weights = torch.where(active[:, None], weights, torch.zeros_like(weights))
            event = (modality_tokens * weights[..., None]).sum(dim=1)
            event = event + self.time_projection(time_features)

            recurrent_input = torch.cat((state, event, time_features), dim=-1)
            update = self.update_gate(recurrent_input).sigmoid()
            proposal = self.candidate(recurrent_input).tanh()
            next_state = (1.0 - update) * state + update * proposal
            state = torch.where(active[:, None], next_state, state)
            previous_time = torch.where(active, safe_time, previous_time)
            have_time = have_time | active

            state_rows.append(torch.where(active[:, None], state, torch.zeros_like(state)))
            weight_rows.append(weights)
            gate_rows.append(torch.where(active[:, None], update, torch.zeros_like(update)))

        return TimeAwareFusionOutput(
            fused=state,
            states=torch.stack(state_rows, dim=1),
            modality_weights=torch.stack(weight_rows, dim=1),
            update_gates=torch.stack(gate_rows, dim=1),
        )


class TimeAwareGatedDecisionScorer(nn.Module):
    """Score typed candidate features against a fused temporal observation."""

    def __init__(
        self,
        hidden_size: int,
        modality_count: int,
        option_size: int,
        width: int = 128,
        time_scale_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        if option_size < 1 or width < 1:
            raise ValueError("option_size and width must be positive")
        self.option_size = option_size
        self.fusion = TimeAwareGatedFusion(
            hidden_size=hidden_size,
            modality_count=modality_count,
            time_scale_seconds=time_scale_seconds,
        )
        self.option_scorer = FrozenFeatureCandidateScorer(
            feature_size=hidden_size + option_size,
            hidden_size=width,
        )

    def forward(
        self,
        features: Tensor,
        timestamps_seconds: Tensor,
        available: Tensor,
        option_features: Tensor,
        option_valid: Tensor,
    ) -> Tensor:
        if option_features.ndim != 3 or option_features.shape[-1] != self.option_size:
            raise ValueError("option_features must have shape [batch, candidates, option_size]")
        if option_features.shape[0] != features.shape[0]:
            raise ValueError("observation and option batch sizes differ")
        if option_valid.dtype != torch.bool or option_valid.shape != option_features.shape[:2]:
            raise ValueError("option_valid must be a boolean [batch, candidates] mask")
        if not (option_valid.sum(dim=-1) >= 2).all():
            raise ValueError("at least two valid options are required per sample")
        if not torch.isfinite(option_features[option_valid]).all():
            raise ValueError("valid option features must be finite")

        observation = self.fusion(features, timestamps_seconds, available).fused
        safe_options = torch.where(
            option_valid[..., None], option_features, torch.zeros_like(option_features)
        )
        repeated_observation = observation[:, None, :].expand(
            -1, option_features.shape[1], -1
        )
        candidate_features = torch.cat((repeated_observation, safe_options), dim=-1)
        logits = self.option_scorer(candidate_features)
        if not torch.isfinite(logits[option_valid]).all():
            raise ValueError("valid option logits must be finite")
        return logits.masked_fill(~option_valid, float("-inf"))
