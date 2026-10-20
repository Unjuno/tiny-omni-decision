"""Small causal fusion/readout for aligned frozen video, audio and text features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class TemporalDecisionOutput:
    logits: Tensor
    modality_weights: Tensor
    state: tuple[Tensor, Tensor]


def probability_json(logits: Tensor, options: Sequence[str], sample_id: str) -> dict[str, Any]:
    """Serialize one decision as an option-probability JSON object."""
    if logits.ndim != 1 or logits.numel() != len(options) or len(options) < 2:
        raise ValueError("one logit per supplied option is required")
    if (
        not sample_id
        or any(not option.strip() for option in options)
        or len(set(options)) != len(options)
    ):
        raise ValueError("sample ID and unique, nonempty options are required")
    if not torch.isfinite(logits).all():
        raise ValueError("decision logits must be finite")
    probabilities = torch.softmax(logits.float(), dim=-1).tolist()
    if not all(math.isfinite(value) for value in probabilities):
        raise ValueError("decision probabilities must be finite")
    prediction = max(range(len(options)), key=probabilities.__getitem__)
    return {
        "id": sample_id,
        "prediction": options[prediction],
        "probabilities": [
            {"option": option, "probability": float(probability)}
            for option, probability in zip(options, probabilities, strict=True)
        ],
    }


class CausalGatedTemporalDecision(nn.Module):
    """A small, causal, text-conditioned audio/video decision readout.

    Input features are frozen encoder outputs. The module only learns compact
    projections, per-time audio/video gates, a multiplicative interaction, and
    an LSTM state update. It emits two option logits at every time step.
    """

    def __init__(
        self,
        *,
        video_size: int = 768,
        audio_size: int = 384,
        text_size: int = 384,
        hidden_size: int = 128,
        modality_mode: str = "joint",
    ) -> None:
        super().__init__()
        if min(video_size, audio_size, text_size, hidden_size) < 1:
            raise ValueError("feature and hidden dimensions must be positive")
        if modality_mode not in {"joint", "video_only", "audio_only"}:
            raise ValueError("modality_mode must be joint, video_only, or audio_only")
        self.video_size = video_size
        self.audio_size = audio_size
        self.text_size = text_size
        self.hidden_size = hidden_size
        self.modality_mode = modality_mode
        self.video_projection = nn.Linear(video_size, hidden_size)
        self.audio_projection = nn.Linear(audio_size, hidden_size)
        self.text_projection = nn.Linear(text_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 3, 2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.temporal = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 2)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        text: Tensor,
        *,
        lengths: Tensor | None = None,
        video_present: Tensor | None = None,
        audio_present: Tensor | None = None,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> TemporalDecisionOutput:
        if video.ndim != 3 or audio.ndim != 3 or text.ndim != 2:
            raise ValueError("video/audio must be [batch,time,feature], text [batch,feature]")
        batch, steps, video_size = video.shape
        if (
            batch < 1
            or steps < 1
            or video_size != self.video_size
            or audio.shape != (batch, steps, self.audio_size)
            or text.shape != (batch, self.text_size)
        ):
            raise ValueError("video, audio, and text feature shapes do not match the model")
        if not all(torch.isfinite(value).all() for value in (video, audio, text)):
            raise ValueError("input features must be finite")

        def presence_mask(value: Tensor | None, name: str) -> Tensor:
            if value is None:
                return torch.ones((batch, steps), dtype=torch.bool, device=video.device)
            if value.shape != (batch, steps) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean [batch,time] tensor")
            return value.to(video.device)

        video_mask = presence_mask(video_present, "video_present").clone()
        audio_mask = presence_mask(audio_present, "audio_present").clone()
        if self.modality_mode == "video_only":
            audio_mask = torch.zeros_like(audio_mask)
        elif self.modality_mode == "audio_only":
            video_mask = torch.zeros_like(video_mask)
        if not torch.all(video_mask | audio_mask):
            raise ValueError("each time step requires at least one available modality")

        video_hidden = torch.tanh(self.video_projection(video)) * video_mask.unsqueeze(-1)
        audio_hidden = torch.tanh(self.audio_projection(audio)) * audio_mask.unsqueeze(-1)
        text_hidden = torch.tanh(self.text_projection(text)).unsqueeze(1).expand(-1, steps, -1)
        gate_logits = self.gate(torch.cat([video_hidden, audio_hidden, text_hidden], dim=-1))
        gate_logits = gate_logits.masked_fill(
            ~torch.stack([video_mask, audio_mask], dim=-1), torch.finfo(gate_logits.dtype).min
        )
        weights = torch.softmax(gate_logits, dim=-1)
        weighted_modalities = weights[..., 0:1] * video_hidden + weights[..., 1:2] * audio_hidden
        interaction = video_hidden * audio_hidden
        fused = self.fusion(torch.cat([weighted_modalities, text_hidden, interaction], dim=-1))

        if lengths is None:
            temporal_output, next_state = self.temporal(fused, state)
        else:
            if (
                lengths.shape != (batch,)
                or lengths.dtype not in (torch.int32, torch.int64)
                or torch.any(lengths < 1)
                or torch.any(lengths > steps)
            ):
                raise ValueError("lengths must contain one valid positive length per row")
            if state is not None:
                raise ValueError(
                    "packed variable-length input cannot be combined with carried state"
                )
            packed = pack_padded_sequence(
                fused, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, next_state = self.temporal(packed)
            temporal_output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=steps
            )
        logits = self.classifier(temporal_output)
        if lengths is not None:
            valid = torch.arange(steps, device=video.device).unsqueeze(0) < lengths.to(
                video.device
            ).unsqueeze(1)
            logits = logits.masked_fill(~valid.unsqueeze(-1), 0.0)
            weights = weights.masked_fill(~valid.unsqueeze(-1), 0.0)
        if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
            raise ValueError("temporal decision produced non-finite logits or gates")
        return TemporalDecisionOutput(logits=logits, modality_weights=weights, state=next_state)


class TaskConditionedTemporalDecision(nn.Module):
    """Text-free causal AV readout conditioned only on a typed task category."""

    def __init__(
        self,
        *,
        task_count: int,
        video_size: int = 768,
        audio_size: int = 384,
        hidden_size: int = 128,
        modality_mode: str = "joint",
    ) -> None:
        super().__init__()
        if min(task_count, video_size, audio_size, hidden_size) < 1:
            raise ValueError("task, feature, and hidden dimensions must be positive")
        if modality_mode not in {"joint", "video_only", "audio_only"}:
            raise ValueError("modality_mode must be joint, video_only, or audio_only")
        self.video_size = video_size
        self.audio_size = audio_size
        self.task_count = task_count
        self.hidden_size = hidden_size
        self.modality_mode = modality_mode
        self.video_projection = nn.Linear(video_size, hidden_size)
        self.audio_projection = nn.Linear(audio_size, hidden_size)
        self.task_embedding = nn.Embedding(task_count, hidden_size)
        self.gate = nn.Linear(hidden_size * 3, 2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.temporal = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 2)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        task_ids: Tensor,
        *,
        lengths: Tensor | None = None,
        video_present: Tensor | None = None,
        audio_present: Tensor | None = None,
    ) -> TemporalDecisionOutput:
        if video.ndim != 3 or audio.ndim != 3 or task_ids.ndim != 1:
            raise ValueError("video/audio must be [batch,time,feature] and task_ids [batch]")
        batch, steps, width = video.shape
        if (
            batch < 1
            or steps < 1
            or width != self.video_size
            or audio.shape != (batch, steps, self.audio_size)
            or task_ids.shape != (batch,)
            or task_ids.dtype not in (torch.int32, torch.int64)
            or torch.any(task_ids < 0)
            or torch.any(task_ids >= self.task_count)
        ):
            raise ValueError("video, audio, or task ID shape/value is invalid")
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise ValueError("input features must be finite")

        def presence_mask(value: Tensor | None, name: str) -> Tensor:
            if value is None:
                return torch.ones((batch, steps), dtype=torch.bool, device=video.device)
            if value.shape != (batch, steps) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean [batch,time] tensor")
            return value.to(video.device)

        video_mask = presence_mask(video_present, "video_present").clone()
        audio_mask = presence_mask(audio_present, "audio_present").clone()
        if self.modality_mode == "video_only":
            audio_mask.zero_()
        elif self.modality_mode == "audio_only":
            video_mask.zero_()
        if not torch.all(video_mask | audio_mask):
            raise ValueError("each time step requires at least one available modality")

        video_hidden = torch.tanh(self.video_projection(video)) * video_mask.unsqueeze(-1)
        audio_hidden = torch.tanh(self.audio_projection(audio)) * audio_mask.unsqueeze(-1)
        task_hidden = self.task_embedding(task_ids.to(video.device))
        task_sequence = task_hidden.unsqueeze(1).expand(-1, steps, -1)
        gate_logits = self.gate(torch.cat([video_hidden, audio_hidden, task_sequence], dim=-1))
        gate_logits = gate_logits.masked_fill(
            ~torch.stack([video_mask, audio_mask], dim=-1), torch.finfo(gate_logits.dtype).min
        )
        weights = torch.softmax(gate_logits, dim=-1)
        weighted = weights[..., :1] * video_hidden + weights[..., 1:] * audio_hidden
        interaction = video_hidden * audio_hidden
        fused = self.fusion(torch.cat([weighted, task_sequence, interaction], dim=-1))
        if lengths is None:
            temporal_output, next_state = self.temporal(fused)
        else:
            if (
                lengths.shape != (batch,)
                or lengths.dtype not in (torch.int32, torch.int64)
                or torch.any(lengths < 1)
                or torch.any(lengths > steps)
            ):
                raise ValueError("lengths must contain one valid positive length per row")
            packed = pack_padded_sequence(
                fused, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, next_state = self.temporal(packed)
            temporal_output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=steps
            )
        logits = self.classifier(temporal_output)
        if lengths is not None:
            valid = torch.arange(steps, device=video.device).unsqueeze(0) < lengths.to(
                video.device
            ).unsqueeze(1)
            logits = logits.masked_fill(~valid.unsqueeze(-1), 0.0)
            weights = weights.masked_fill(~valid.unsqueeze(-1), 0.0)
        if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
            raise ValueError("temporal decision produced non-finite logits or gates")
        return TemporalDecisionOutput(logits=logits, modality_weights=weights, state=next_state)


class TaskConditionedAVSensorTemporalDecision(nn.Module):
    """Task-conditioned AV/IMU fusion with per-window gating and causal memory."""

    def __init__(
        self,
        *,
        task_count: int,
        video_size: int = 768,
        audio_size: int = 384,
        sensor_size: int = 630,
        hidden_size: int = 128,
        modality_mode: str = "avs",
    ) -> None:
        super().__init__()
        if min(task_count, video_size, audio_size, sensor_size, hidden_size) < 1:
            raise ValueError("task, modality, and hidden dimensions must be positive")
        if modality_mode not in {"av", "sensor_only", "avs"}:
            raise ValueError("modality_mode must be av, sensor_only, or avs")
        self.video_size = video_size
        self.audio_size = audio_size
        self.sensor_size = sensor_size
        self.hidden_size = hidden_size
        self.modality_mode = modality_mode
        self.video_projection = nn.Linear(video_size, hidden_size)
        self.audio_projection = nn.Linear(audio_size, hidden_size)
        self.sensor_projection = nn.Linear(sensor_size, hidden_size)
        self.task_embedding = nn.Embedding(task_count, hidden_size)
        self.gate = nn.Linear(hidden_size * 4, 3)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.temporal = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 2)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        sensor: Tensor,
        task_ids: Tensor,
        *,
        lengths: Tensor | None = None,
    ) -> TemporalDecisionOutput:
        if video.ndim != 3 or audio.ndim != 3 or sensor.ndim != 3 or task_ids.ndim != 1:
            raise ValueError("video/audio/sensor must be [batch,time,feature], task_ids [batch]")
        batch, steps, width = video.shape
        if (
            batch < 1
            or steps < 1
            or width != self.video_size
            or audio.shape != (batch, steps, self.audio_size)
            or sensor.shape != (batch, steps, self.sensor_size)
            or task_ids.shape != (batch,)
            or task_ids.dtype not in (torch.int32, torch.int64)
            or torch.any(task_ids < 0)
            or torch.any(task_ids >= self.task_embedding.num_embeddings)
        ):
            raise ValueError("AV/sensor features or task IDs have invalid shape/value")
        if not all(torch.isfinite(value).all() for value in (video, audio, sensor)):
            raise ValueError("AV/sensor input features must be finite")

        active = {
            "av": torch.tensor([True, True, False], device=video.device),
            "sensor_only": torch.tensor([False, False, True], device=video.device),
            "avs": torch.tensor([True, True, True], device=video.device),
        }[self.modality_mode]
        video_hidden = torch.tanh(self.video_projection(video)) * active[0]
        audio_hidden = torch.tanh(self.audio_projection(audio)) * active[1]
        sensor_hidden = torch.tanh(self.sensor_projection(sensor)) * active[2]
        task_hidden = self.task_embedding(task_ids).unsqueeze(1).expand(-1, steps, -1)
        modality_hidden = torch.stack([video_hidden, audio_hidden, sensor_hidden], dim=2)
        gate_input = torch.cat([video_hidden, audio_hidden, sensor_hidden, task_hidden], dim=-1)
        gate_logits = self.gate(gate_input).masked_fill(
            ~active.view(1, 1, 3), torch.finfo(gate_input.dtype).min
        )
        weights = torch.softmax(gate_logits, dim=-1)
        weighted = (weights.unsqueeze(-1) * modality_hidden).sum(dim=2)
        interactions = torch.cat(
            [
                video_hidden * audio_hidden * (active[0] & active[1]),
                video_hidden * sensor_hidden * (active[0] & active[2]),
                audio_hidden * sensor_hidden * (active[1] & active[2]),
            ],
            dim=-1,
        )
        fused = self.fusion(torch.cat([weighted, task_hidden, interactions], dim=-1))
        valid = None
        if lengths is None:
            temporal_output, next_state = self.temporal(fused)
        else:
            if (
                lengths.shape != (batch,)
                or lengths.dtype not in (torch.int32, torch.int64)
                or torch.any(lengths < 1)
                or torch.any(lengths > steps)
            ):
                raise ValueError("lengths must contain one valid positive length per row")
            packed = pack_padded_sequence(
                fused, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, next_state = self.temporal(packed)
            temporal_output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=steps
            )
            valid = torch.arange(steps, device=video.device).unsqueeze(0) < lengths.to(
                video.device
            ).unsqueeze(1)
            temporal_output = temporal_output.masked_fill(~valid.unsqueeze(-1), 0.0)
            weights = weights.masked_fill(~valid.unsqueeze(-1), 0.0)
        logits = self.classifier(temporal_output)
        if valid is not None:
            logits = logits.masked_fill(~valid.unsqueeze(-1), 0.0)
        if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
            raise ValueError("AV/sensor temporal decision produced non-finite outputs")
        return TemporalDecisionOutput(logits=logits, modality_weights=weights, state=next_state)
