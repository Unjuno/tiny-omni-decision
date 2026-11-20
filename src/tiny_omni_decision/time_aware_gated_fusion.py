"""Small candidate-conditioned fusion for timestamped frozen feature streams.

This module is an untrained architecture candidate. It consumes encoded numeric
tokens; it does not load media, infer calibration, align clocks, or establish
that any task benefits from multiple modalities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TemporalFeatureStream:
    """One modality's batched tokens, timestamps, validity and presence."""

    values: Tensor
    timestamps_ms: Tensor
    valid: Tensor
    present: Tensor


class TimeAwareGatedFusion(nn.Module):
    """Fuse timestamped streams using candidate cross-attention and modality gates.

    ``streams`` must contain exactly the configured modality names. Missing
    streams are represented explicitly by ``present=False`` and an all-false
    validity row; a present zero-valued measurement remains distinguishable.
    Timestamps share a clock and are expressed in milliseconds. They are
    encoded relative to the latest valid token in each example, so absolute
    wall-clock values are not exposed to the learned block.

    The result is a candidate-aligned state tensor, not probabilities. A
    separately specified Decision scorer owns option scoring and JSON output.
    """

    def __init__(
        self,
        dimensions: Mapping[str, int],
        query_dim: int,
        *,
        width: int = 96,
        heads: int = 4,
        time_unit_ms: float = 1000.0,
        use_time: bool = True,
        use_gates: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(dimensions, Mapping) or not dimensions:
            raise ValueError("at least one named modality dimension is required")
        names = tuple(dimensions)
        if any(not isinstance(name, str) or not name.strip() or "." in name
               for name in names):
            raise ValueError("modality names must be nonempty strings")
        if any(type(size) is not int or size < 1 for size in dimensions.values()):
            raise ValueError("modality feature dimensions must be positive integers")
        if (type(query_dim) is not int or query_dim < 1 or type(width) is not int
                or width < 1 or type(heads) is not int or heads < 1 or width % heads):
            raise ValueError("positive query/width and a head count dividing width are required")
        if not math.isfinite(time_unit_ms) or time_unit_ms <= 0:
            raise ValueError("time_unit_ms must be finite and positive")
        if type(use_time) is not bool or type(use_gates) is not bool:
            raise ValueError("use_time and use_gates must be booleans")

        self.names = names
        self.dimensions = dict(dimensions)
        self.query_dim = query_dim
        self.width = width
        self.time_unit_ms = float(time_unit_ms)
        self._use_time = use_time
        self._use_gates = use_gates
        self.projections = nn.ModuleDict({
            name: nn.Linear(size, width) for name, size in self.dimensions.items()
        })
        self.modality = nn.Embedding(len(names), width)
        self.time_encoder = nn.Sequential(
            nn.Linear(9, width), nn.GELU(), nn.Linear(width, width)
        )
        self.query = nn.Sequential(nn.Linear(query_dim, width), nn.LayerNorm(width))
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.modality_summary = nn.ModuleDict({
            name: nn.Linear(width, width) for name in names
        })
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1), nn.Sigmoid()
            ) for name in names
        })
        self.norm = nn.LayerNorm(width)
        if not use_time:
            self.time_encoder.requires_grad_(False)
        if not use_gates:
            self.gates.requires_grad_(False)

    def get_extra_state(self) -> dict[str, object]:
        return {
            "schema": 1,
            "names": self.names,
            "dimensions": self.dimensions,
            "query_dim": self.query_dim,
            "width": self.width,
            "heads": self.attention.num_heads,
            "time_unit_ms": self.time_unit_ms,
            "use_time": self._use_time,
            "use_gates": self._use_gates,
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise ValueError("fusion checkpoint configuration differs")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("fusion checkpoint configuration missing or different")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def parameter_ledger(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters()
                        if parameter.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    def _validate(
        self,
        streams: Mapping[str, TemporalFeatureStream],
        queries: Tensor,
        option_valid: Tensor,
    ) -> tuple[int, Tensor, list[tuple[Tensor, Tensor, Tensor]]]:
        reference = self.modality.weight
        if not isinstance(streams, Mapping):
            raise ValueError("streams must be a modality-to-feature mapping")
        if set(streams) != set(self.names):
            raise ValueError("stream names must exactly match configured modalities")
        if (not isinstance(queries, Tensor) or queries.ndim != 3
                or queries.shape[1] < 2 or queries.shape[1] > 40
                or queries.shape[-1] != self.query_dim or queries.dtype != reference.dtype
                or queries.device != reference.device):
            raise ValueError("queries must be [batch, 2..40, query_dim] on model dtype/device")
        batch, candidates = queries.shape[:2]
        if (not isinstance(option_valid, Tensor) or option_valid.dtype != torch.bool
                or option_valid.shape != (batch, candidates)
                or option_valid.device != reference.device
                or not (option_valid.sum(-1) >= 2).all()):
            raise ValueError("boolean option mask with at least two valid choices is required")
        if not torch.isfinite(queries[option_valid]).all():
            raise ValueError("valid candidate queries must be finite")

        checked: list[tuple[Tensor, Tensor, Tensor]] = []
        total_tokens = 0
        any_present = torch.zeros(batch, dtype=torch.bool, device=reference.device)
        latest = torch.full((batch,), -torch.inf, dtype=torch.float64,
                            device=reference.device)
        for name in self.names:
            stream = streams[name]
            if not isinstance(stream, TemporalFeatureStream):
                raise ValueError("each configured stream must be a TemporalFeatureStream")
            values, times, valid, present = (
                stream.values, stream.timestamps_ms, stream.valid, stream.present
            )
            if (not all(isinstance(value, Tensor)
                        for value in (values, times, valid, present))
                    or values.ndim != 3 or values.shape[0] != batch
                    or values.shape[1] < 1 or values.shape[1] > 8192
                    or values.shape[2] != self.dimensions[name]
                    or times.shape != values.shape[:2] or times.dtype not in
                    (torch.float32, torch.float64)
                    or valid.shape != values.shape[:2] or valid.dtype != torch.bool
                    or present.shape != (batch,) or present.dtype != torch.bool):
                raise ValueError(f"{name} stream shapes, feature widths or masks are invalid")
            total_tokens += values.shape[1]
            if total_tokens > 8192:
                raise ValueError("8192 total token cap exceeded; no silent truncation")
            if any(value.device != reference.device
                   for value in (values, times, valid, present)):
                raise ValueError(f"{name} stream tensors must share the model device")
            if values.dtype != reference.dtype:
                raise ValueError(f"{name} features must match model dtype")
            if ((valid & ~present[:, None]).any()
                    or (present & ~valid.any(dim=1)).any()):
                raise ValueError(f"{name} presence and valid-token masks disagree")
            if not torch.isfinite(values[valid]).all() or not torch.isfinite(times[valid]).all():
                raise ValueError(f"{name} valid feature tokens and timestamps must be finite")
            for row in range(batch):
                row_times = times[row, valid[row]]
                if row_times.numel() > 1 and not (row_times[1:] > row_times[:-1]).all():
                    raise ValueError(f"{name} valid timestamps must be strictly increasing")
            safe_times = torch.where(valid, times.to(torch.float64), -torch.inf)
            latest = torch.maximum(latest, safe_times.max(dim=1).values)
            any_present |= present
            checked.append((values, times, valid & present[:, None]))
        if not any_present.all():
            raise ValueError("every example must contain at least one valid modality")
        return candidates, latest, checked

    def forward(
        self,
        streams: Mapping[str, TemporalFeatureStream],
        queries: Tensor,
        option_valid: Tensor,
        *,
        diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        candidates, latest, checked = self._validate(streams, queries, option_valid)
        reference = self.modality.weight
        query = self.query(torch.where(option_valid[..., None], queries,
                                       torch.zeros_like(queries)))
        token_parts: list[Tensor] = []
        valid_parts: list[Tensor] = []
        projected_parts: dict[str, Tensor] = {}
        mask_parts: dict[str, Tensor] = {}
        frequencies = reference.new_tensor((0.25, 1.0, 4.0, 16.0))

        for index, name in enumerate(self.names):
            values, times, mask = checked[index]
            safe_values = torch.where(mask[..., None], values, torch.zeros_like(values))
            safe_times = torch.where(mask, times.to(torch.float64), latest[:, None])
            relative_ms = (safe_times - latest[:, None]).to(reference.dtype)
            relative_seconds = relative_ms / 1000.0
            relative_scaled = relative_ms / self.time_unit_ms
            phase = relative_seconds[..., None] * frequencies * (2.0 * math.pi)
            time_features = torch.cat((
                torch.sign(relative_scaled)[..., None]
                * torch.log1p(relative_scaled.abs())[..., None],
                torch.sin(phase), torch.cos(phase),
            ), dim=-1)
            time_embedding = (self.time_encoder(time_features)
                              if self._use_time else torch.zeros_like(
                                  self.time_encoder(time_features)))
            projected = (self.projections[name](safe_values)
                         + self.modality.weight[index] + time_embedding)
            projected_parts[name] = projected
            mask_parts[name] = mask
            token_parts.append(projected)
            valid_parts.append(mask)

        memory = torch.cat(token_parts, dim=1)
        memory_valid = torch.cat(valid_parts, dim=1)
        attended, _ = self.attention(
            query, memory, memory, key_padding_mask=~memory_valid, need_weights=False
        )
        combined = attended
        gate_values: dict[str, Tensor] = {}
        for name in self.names:
            projected = projected_parts[name]
            mask = mask_parts[name]
            count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(projected.dtype)
            summary = (projected * mask[..., None]).sum(dim=1) / count
            expanded = summary[:, None, :].expand(-1, candidates, -1)
            if self._use_gates:
                gate = self.gates[name](torch.cat((
                    query, attended, query * attended, expanded,
                ), dim=-1))
            else:
                gate = query.new_ones((*query.shape[:2], 1))
            gate = gate * streams[name].present[:, None, None].to(gate.dtype)
            gate_values[name] = gate
            combined = combined + gate * self.modality_summary[name](expanded)

        fused = self.norm(combined)
        fused = torch.where(option_valid[..., None], fused, torch.zeros_like(fused))
        if not torch.isfinite(fused[option_valid]).all():
            raise ValueError("nonfinite fused candidate state")
        if diagnostics:
            return fused, {
                "gates": gate_values,
                "present": {name: streams[name].present for name in self.names},
                "valid_token_counts": {
                    name: mask.sum(dim=1) for name, mask in mask_parts.items()
                },
            }
        return fused
