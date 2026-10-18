"""Text-free option scoring for frozen audio-event and object-crop features."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Condition = Literal["audio_only", "video_only", "joint"]


class AVEventSourceScorer(nn.Module):
    """Small Decision readout for event audio and variable object-crop options.

    This module consumes only frozen numeric features and supplied option order.
    It has no question, transcript, class-name, filename, or candidate-ID input.
    The audio-only condition is intentionally uniform: with unlabeled opaque
    option IDs and no option-side observations, audio alone cannot rank options.
    """

    def __init__(
        self,
        audio_size: int = 512,
        visual_size: int = 384,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if min(audio_size, visual_size, hidden_size) < 1:
            raise ValueError("feature and hidden dimensions must be positive")
        self.audio_size = audio_size
        self.visual_size = visual_size
        self.hidden_size = hidden_size
        self.audio_projection = nn.Linear(audio_size, hidden_size)
        self.visual_projection = nn.Linear(visual_size, hidden_size)
        self.video_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.joint_gate = nn.Linear(2 * hidden_size, hidden_size)
        self.joint_head = nn.Sequential(
            nn.Linear(5 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        *,
        condition: Condition,
        option_count: int | None = None,
        audio_features: Tensor | None = None,
        visual_features: Tensor | None = None,
        return_gate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Return one logit per supplied option in its caller-provided order."""
        if condition not in {"audio_only", "video_only", "joint"}:
            raise ValueError("condition must be audio_only, video_only, or joint")
        if condition == "audio_only":
            if visual_features is not None or audio_features is None:
                raise ValueError("audio_only requires audio and forbids visual features")
            if audio_features.shape != (self.audio_size,):
                raise ValueError("audio_only feature has the wrong shape")
            if not torch.isfinite(audio_features).all():
                raise ValueError("audio-only feature must be finite")
            if type(option_count) is not int or option_count < 2:
                raise ValueError("audio_only requires at least two supplied options")
            logits = audio_features.new_zeros(option_count, dtype=torch.float32)
            gate = None
        elif condition == "video_only":
            if audio_features is not None or visual_features is None:
                raise ValueError("video_only requires visual and forbids audio features")
            self._validate_visual(visual_features)
            visual = self.visual_projection(F.normalize(visual_features.float(), dim=-1))
            logits = self.video_head(visual).squeeze(-1).float()
            gate = None
        else:
            if audio_features is None or visual_features is None:
                raise ValueError("joint condition requires both modalities")
            self._validate_visual(visual_features)
            if audio_features.shape != (self.audio_size,):
                raise ValueError("joint audio feature has the wrong shape")
            if not torch.isfinite(audio_features).all():
                raise ValueError("joint audio feature must be finite")
            audio = self.audio_projection(F.normalize(audio_features.float(), dim=-1))
            visual = self.visual_projection(F.normalize(visual_features.float(), dim=-1))
            audio_per_option = audio.expand(visual.shape[0], -1)
            gate = torch.sigmoid(
                self.joint_gate(torch.cat((audio_per_option, visual), dim=-1))
            )
            gated_audio = gate * audio_per_option
            joint = torch.cat(
                (
                    visual,
                    gated_audio,
                    torch.abs(visual - gated_audio),
                    visual * gated_audio,
                    gate,
                ),
                dim=-1,
            )
            logits = self.joint_head(joint).squeeze(-1).float()
        if not torch.isfinite(logits).all():
            raise ValueError("AV event scorer produced non-finite logits")
        return (logits, gate) if return_gate else logits

    def _validate_visual(self, visual_features: Tensor) -> None:
        if (
            visual_features.ndim != 2
            or visual_features.shape[0] < 2
            or visual_features.shape[1] != self.visual_size
        ):
            raise ValueError("visual options must have shape [n_options>=2, visual_size]")
        if not torch.isfinite(visual_features).all():
            raise ValueError("visual option features must be finite")
