"""Small readout heads for controlled synthetic audio-video sequence matching."""

from __future__ import annotations

import math

import torch
from torch import nn

SEQUENCE_LENGTH = 8


def _validate_inputs(
    audio: torch.Tensor,
    video: torch.Tensor,
    query: torch.Tensor,
    audio_width: int,
    video_width: int,
    query_width: int,
) -> None:
    if audio.ndim != 3 or tuple(audio.shape[1:]) != (SEQUENCE_LENGTH, audio_width):
        raise ValueError(f"audio must have shape [batch,8,{audio_width}]")
    if video.ndim != 3 or tuple(video.shape[1:]) != (SEQUENCE_LENGTH, video_width):
        raise ValueError(f"video must have shape [batch,8,{video_width}]")
    if query.ndim != 2 or query.shape != (audio.shape[0], query_width):
        raise ValueError(f"query must have shape [batch,{query_width}]")
    if audio.shape[0] != video.shape[0] or audio.shape[0] != query.shape[0]:
        raise ValueError("audio, video, and query batch sizes must match")
    if audio.device != video.device or audio.device != query.device:
        raise ValueError("audio, video, and query must be on the same device")
    if audio.dtype != video.dtype or audio.dtype != query.dtype:
        raise ValueError("audio, video, and query must use the same dtype")


class TemporalLSTMReadout(nn.Module):
    """Read aligned audiovisual features with a two-layer temporal LSTM."""

    def __init__(
        self,
        audio_width: int,
        video_width: int,
        query_width: int,
        hidden_size: int = 96,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if min(audio_width, video_width, query_width, hidden_size, num_layers) < 1:
            raise ValueError("all readout dimensions must be positive")
        self.audio_width = audio_width
        self.video_width = video_width
        self.query_width = query_width
        self.lstm = nn.LSTM(
            audio_width + video_width + query_width,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(
        self, audio: torch.Tensor, video: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        _validate_inputs(
            audio, video, query, self.audio_width, self.video_width, self.query_width
        )
        repeated_query = query.unsqueeze(1).expand(-1, SEQUENCE_LENGTH, -1)
        sequence = torch.cat((audio, video, repeated_query), dim=-1)
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(hidden[-1])


class ConditionedTemporalAVReadout(nn.Module):
    """Use the query to gate aligned AV evidence before recurrent integration."""

    def __init__(
        self,
        audio_width: int,
        video_width: int,
        query_width: int,
        hidden_size: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if min(audio_width, video_width, query_width, hidden_size, num_layers) < 1:
            raise ValueError("all readout dimensions must be positive")
        self.audio_width = audio_width
        self.video_width = video_width
        self.query_width = query_width
        self.hidden_size = hidden_size
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_width, hidden_size), nn.LayerNorm(hidden_size)
        )
        self.video_projection = nn.Sequential(
            nn.Linear(video_width, hidden_size), nn.LayerNorm(hidden_size)
        )
        self.query_projection = nn.Sequential(
            nn.Linear(query_width, hidden_size), nn.LayerNorm(hidden_size)
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )
        self.temporal = nn.LSTM(
            hidden_size * 2,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(
        self, audio: torch.Tensor, video: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        _validate_inputs(
            audio, video, query, self.audio_width, self.video_width, self.query_width
        )
        audio_steps = self.audio_projection(audio)
        video_steps = self.video_projection(video)
        query_context = self.query_projection(query)
        repeated_query = query_context.unsqueeze(1).expand(-1, SEQUENCE_LENGTH, -1)
        gate_logits = self.modality_gate(
            torch.cat((audio_steps, video_steps, repeated_query), dim=-1)
        )
        modality_weights = torch.softmax(gate_logits, dim=-1)
        fused_steps = (
            modality_weights[..., 0:1] * audio_steps
            + modality_weights[..., 1:2] * video_steps
        )
        sequence = torch.cat((fused_steps, repeated_query), dim=-1)
        _, (hidden, _) = self.temporal(sequence)
        return self.classifier(hidden[-1])


class BypassConditionedTemporalAVReadout(nn.Module):
    """Keep both projected modalities visible beside a query-conditioned gate."""

    def __init__(
        self,
        audio_width: int,
        video_width: int,
        query_width: int,
        hidden_size: int = 112,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if min(audio_width, video_width, query_width, hidden_size, num_layers) < 1:
            raise ValueError("all readout dimensions must be positive")
        self.audio_width = audio_width
        self.video_width = video_width
        self.query_width = query_width
        self.hidden_size = hidden_size
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_width, hidden_size), nn.LayerNorm(hidden_size)
        )
        self.video_projection = nn.Sequential(
            nn.Linear(video_width, hidden_size), nn.LayerNorm(hidden_size)
        )
        self.query_projection = nn.Sequential(
            nn.Linear(query_width, hidden_size), nn.LayerNorm(hidden_size)
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )
        self.temporal = nn.LSTM(
            hidden_size * 4,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(
        self, audio: torch.Tensor, video: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        _validate_inputs(
            audio, video, query, self.audio_width, self.video_width, self.query_width
        )
        audio_steps = self.audio_projection(audio)
        video_steps = self.video_projection(video)
        repeated_query = self.query_projection(query).unsqueeze(1).expand(
            -1, SEQUENCE_LENGTH, -1
        )
        gate_logits = self.modality_gate(
            torch.cat((audio_steps, video_steps, repeated_query), dim=-1)
        )
        modality_weights = torch.softmax(gate_logits, dim=-1)
        gated_steps = (
            modality_weights[..., 0:1] * audio_steps
            + modality_weights[..., 1:2] * video_steps
        )
        sequence = torch.cat(
            (audio_steps, video_steps, gated_steps, repeated_query), dim=-1
        )
        _, (hidden, _) = self.temporal(sequence)
        return self.classifier(hidden[-1])


class ResidualGatedTemporalReadout(nn.Module):
    """Add identity-initialized, bounded modality gains before the raw-input LSTM."""

    def __init__(
        self,
        audio_width: int,
        video_width: int,
        query_width: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        gate_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        if min(
            audio_width,
            video_width,
            query_width,
            hidden_size,
            num_layers,
            gate_hidden_size,
        ) < 1:
            raise ValueError("all readout dimensions must be positive")
        self.audio_width = audio_width
        self.video_width = video_width
        self.query_width = query_width
        self.lstm = nn.LSTM(
            audio_width + video_width + query_width,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 2)
        self.audio_gate_projection = nn.Sequential(
            nn.Linear(audio_width, gate_hidden_size), nn.LayerNorm(gate_hidden_size)
        )
        self.video_gate_projection = nn.Sequential(
            nn.Linear(video_width, gate_hidden_size), nn.LayerNorm(gate_hidden_size)
        )
        self.query_gate_projection = nn.Sequential(
            nn.Linear(query_width, gate_hidden_size), nn.LayerNorm(gate_hidden_size)
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(gate_hidden_size * 3, gate_hidden_size),
            nn.GELU(),
            nn.Linear(gate_hidden_size, 2),
        )
        nn.init.zeros_(self.modality_gate[-1].weight)
        nn.init.zeros_(self.modality_gate[-1].bias)

    def forward(
        self, audio: torch.Tensor, video: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        _validate_inputs(
            audio, video, query, self.audio_width, self.video_width, self.query_width
        )
        query_steps = query.unsqueeze(1).expand(-1, SEQUENCE_LENGTH, -1)
        gate_context = torch.cat(
            (
                self.audio_gate_projection(audio),
                self.video_gate_projection(video),
                self.query_gate_projection(query_steps),
            ),
            dim=-1,
        )
        modality_weights = torch.softmax(self.modality_gate(gate_context), dim=-1)
        modality_gains = 1.0 + modality_weights - 0.5
        sequence = torch.cat(
            (
                audio * modality_gains[..., 0:1],
                video * modality_gains[..., 1:2],
                query_steps,
            ),
            dim=-1,
        )
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(hidden[-1])


class ConditionalAVReadout(nn.Module):
    """Bidirectionally exchange AV sequence evidence, query-pool, and gate."""

    def __init__(
        self,
        audio_width: int,
        video_width: int,
        query_width: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        cross_depth: int = 2,
    ) -> None:
        super().__init__()
        if min(audio_width, video_width, query_width, hidden_size, num_heads, cross_depth) < 1:
            raise ValueError("all readout dimensions must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        self.audio_width = audio_width
        self.video_width = video_width
        self.query_width = query_width
        self.hidden_size = hidden_size
        self.audio_projection = nn.Linear(audio_width, hidden_size)
        self.video_projection = nn.Linear(video_width, hidden_size)
        self.query_projection = nn.Linear(query_width, hidden_size)
        self.position_embedding = nn.Parameter(torch.empty(1, SEQUENCE_LENGTH, hidden_size))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.audio_reads_video = nn.ModuleList(
            nn.MultiheadAttention(hidden_size, num_heads, dropout=0.0, batch_first=True)
            for _ in range(cross_depth)
        )
        self.video_reads_audio = nn.ModuleList(
            nn.MultiheadAttention(hidden_size, num_heads, dropout=0.0, batch_first=True)
            for _ in range(cross_depth)
        )
        self.audio_norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(cross_depth))
        self.video_norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(cross_depth))
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(
        self, audio: torch.Tensor, video: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        _validate_inputs(
            audio, video, query, self.audio_width, self.video_width, self.query_width
        )
        query_embedding = self.query_projection(query)
        query_steps = query_embedding.unsqueeze(1) + self.position_embedding
        audio_steps = self.audio_projection(audio) + query_steps
        video_steps = self.video_projection(video) + query_steps
        for audio_attention, video_attention, audio_norm, video_norm in zip(
            self.audio_reads_video,
            self.video_reads_audio,
            self.audio_norms,
            self.video_norms,
            strict=True,
        ):
            audio_message = audio_attention(
                audio_steps, video_steps, video_steps, need_weights=False
            )[0]
            video_message = video_attention(
                video_steps, audio_steps, audio_steps, need_weights=False
            )[0]
            audio_steps = audio_norm(audio_steps + audio_message)
            video_steps = video_norm(video_steps + video_message)

        scale = math.sqrt(self.hidden_size)
        audio_weights = torch.softmax(
            (audio_steps * query_embedding.unsqueeze(1)).sum(-1) / scale, dim=-1
        )
        video_weights = torch.softmax(
            (video_steps * query_embedding.unsqueeze(1)).sum(-1) / scale, dim=-1
        )
        audio_context = (audio_steps * audio_weights.unsqueeze(-1)).sum(dim=1)
        video_context = (video_steps * video_weights.unsqueeze(-1)).sum(dim=1)
        joint_context = torch.cat((audio_context, video_context, query_embedding), dim=-1)
        gate_weights = torch.softmax(self.gate(joint_context), dim=-1)
        gated = torch.cat(
            (
                audio_context * gate_weights[:, 0:1],
                video_context * gate_weights[:, 1:2],
                query_embedding,
            ),
            dim=-1,
        )
        return self.classifier(gated)
