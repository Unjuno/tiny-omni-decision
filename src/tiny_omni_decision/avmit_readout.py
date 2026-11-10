"""Small option-scoring readouts over frozen AVMIT feature sequences."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from .native_feature_decision import NativeFeatureDecisionScorer


class _TemporalReadout(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, policy: str) -> None:
        super().__init__()
        if input_width < 1 or hidden_width < 1:
            raise ValueError("feature and hidden widths must be positive")
        if policy not in {"mean", "gru"}:
            raise ValueError("temporal policy must be 'mean' or 'gru'")
        self.input_width = input_width
        self.hidden_width = hidden_width
        self.policy = policy
        if policy == "mean":
            self.projection = nn.Linear(input_width, hidden_width)
            self.recurrent = None
        else:
            self.projection = None
            self.recurrent = nn.GRU(input_width, hidden_width, batch_first=True)

    def forward(self, sequence: Tensor, lengths: Tensor) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_width:
            raise ValueError("sequence must have shape [batch, time, feature_width]")
        if (
            lengths.dtype != torch.long
            or lengths.shape != (sequence.shape[0],)
            or lengths.device != sequence.device
            or not ((lengths >= 1) & (lengths <= sequence.shape[1])).all()
        ):
            raise ValueError("valid sequence lengths are required for every batch row")
        if not sequence.is_floating_point() or not torch.isfinite(sequence).all():
            raise ValueError("feature sequences must be finite floating point tensors")

        if self.policy == "mean":
            time = torch.arange(sequence.shape[1], device=sequence.device)
            valid = (time[None, :] < lengths[:, None]).to(sequence.dtype)
            pooled = (sequence * valid[..., None]).sum(dim=1) / lengths[:, None]
            return torch.tanh(self.projection(pooled))

        packed = pack_padded_sequence(
            sequence, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.recurrent(packed)
        return torch.tanh(hidden[-1])


class AVMITFeatureDecisionScorer(nn.Module):
    """Score same/different-label options from two frozen feature sequences."""

    OPTION_IDS = ("same_action", "different_action")

    def __init__(
        self,
        audio_width: int,
        video_width: int,
        *,
        temporal_policy: str,
        hidden_width: int = 64,
        scorer_width: int = 128,
    ) -> None:
        super().__init__()
        self.temporal_policy = temporal_policy
        self.hidden_width = hidden_width
        self.audio = _TemporalReadout(audio_width, hidden_width, temporal_policy)
        self.video = _TemporalReadout(video_width, hidden_width, temporal_policy)
        self.options = NativeFeatureDecisionScorer(
            feature_width=4 * hidden_width,
            option_width=2,
            hidden_width=scorer_width,
        )

    @staticmethod
    def option_features(batch_size: int, *, device: torch.device) -> tuple[Tensor, Tensor]:
        options = torch.eye(2, device=device, dtype=torch.float32)
        options = options.unsqueeze(0).expand(batch_size, -1, -1)
        valid = torch.ones((batch_size, 2), device=device, dtype=torch.bool)
        return options, valid

    def forward(
        self,
        audio: Tensor,
        audio_lengths: Tensor,
        video: Tensor,
        video_lengths: Tensor,
        option_features: Tensor,
        option_valid: Tensor,
    ) -> Tensor:
        if audio.shape[0] != video.shape[0] or audio.device != video.device:
            raise ValueError("audio and video batch sizes and devices must match")
        audio_state = self.audio(audio, audio_lengths)
        video_state = self.video(video, video_lengths)
        joint = torch.cat(
            (
                audio_state,
                video_state,
                torch.abs(audio_state - video_state),
                audio_state * video_state,
            ),
            dim=-1,
        )
        return self.options(joint, option_features, option_valid)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
