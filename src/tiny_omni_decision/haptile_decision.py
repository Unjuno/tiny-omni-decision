"""HapTile frozen-feature multimodal seven-way Decision readout."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .classification_json import classification_json
from .decision import FrozenFeatureCandidateScorer
from .haptile_action import ACTION_LABELS
from .video_sensor_temporal_decision import VideoSensorTemporalDecisionScorer

HAPTILE_MODES = (
    "text_only",
    "state_text",
    "vision_text",
    "tactile_text",
    "vision_tactile_text",
    "full",
)
VIDEO_STREAMS = ("front-camera", "wrist-camera", "tactile-left", "tactile-right")


class HapTileActionDecisionScorer(nn.Module):
    """Score supplied motion options from instruction, video and robot state.

    Video features are frozen VideoPrism outputs in the fixed four-stream order
    above. State features are causal, as-of aligned 25-channel histories. The
    task instruction and option text are encoded by a frozen MiniLM upstream.
    """

    def __init__(self) -> None:
        super().__init__()
        self.text_only_readout = FrozenFeatureCandidateScorer(1536, hidden_size=128)
        self.readout = VideoSensorTemporalDecisionScorer(
            video_feature_width=3072,
            sensor_channels=25,
            option_width=1536,
            context_width=384,
            sensor_hidden_size=32,
            width=64,
            heads=4,
            scorer_hidden_width=64,
        )

    @property
    def parameter_count(self) -> int:
        return self.readout.parameter_count

    @property
    def comparison_parameter_count(self) -> int:
        """Total trainables when retaining the separate text-only control head."""
        return self.parameter_count + sum(
            parameter.numel() for parameter in self.text_only_readout.parameters()
        )

    def forward(
        self,
        video_values: Tensor,
        video_timestamps_ms: Tensor,
        state_values: Tensor,
        state_timestamps_ms: Tensor,
        context_features: Tensor,
        option_features: Tensor,
        option_valid: Tensor,
        *,
        mode: str = "full",
    ) -> Tensor:
        if mode not in HAPTILE_MODES:
            raise ValueError(f"unsupported HapTile input ablation: {mode}")
        if (video_values.ndim != 4 or video_values.shape[1:] != (4, 8, 768)
                or state_values.ndim != 3 or state_values.shape[1:] != (8, 25)
                or context_features.ndim != 2 or context_features.shape[1] != 384
                or option_features.ndim != 3 or option_features.shape[1:] != (7, 1536)):
            raise ValueError("HapTile features do not match the frozen readout contract")
        batch = video_values.shape[0]
        if (state_values.shape[0] != batch or context_features.shape[0] != batch
                or option_features.shape[0] != batch or option_valid.shape != (batch, 7)):
            raise ValueError("HapTile input batches are not aligned")
        if mode == "text_only":
            return self.text_only_readout(option_features).masked_fill(
                ~option_valid, float("-inf")
            )

        # Convert [B, stream, time, channel] to one time-aligned multisensor token.
        video = video_values.permute(0, 2, 1, 3).contiguous().view(batch, 8, 3072)
        use_vision = mode in {"vision_text", "vision_tactile_text", "full"}
        use_tactile = mode in {"tactile_text", "vision_tactile_text", "full"}
        use_state = mode in {"state_text", "full"}
        if not use_vision:
            video[:, :, :1536] = 0
        if not use_tactile:
            video[:, :, 1536:] = 0

        present_video = use_vision or use_tactile
        video_valid = torch.full(
            (batch, 8), present_video, dtype=torch.bool, device=video.device
        )
        video_present = torch.full(
            (batch,), present_video, dtype=torch.bool, device=video.device
        )
        sensor_valid = torch.full(
            (batch, 8), use_state, dtype=torch.bool, device=state_values.device
        )
        sensor_present = torch.full(
            (batch,), use_state, dtype=torch.bool, device=state_values.device
        )
        logits = self.readout(
            video,
            video_timestamps_ms,
            video_valid,
            video_present,
            state_values,
            state_timestamps_ms,
            sensor_valid,
            sensor_present,
            option_features,
            option_valid,
            context_features=context_features,
        )
        return logits

    @torch.inference_mode()
    def decide(
        self,
        video_values: Tensor,
        video_timestamps_ms: Tensor,
        state_values: Tensor,
        state_timestamps_ms: Tensor,
        context_features: Tensor,
        option_features: Tensor,
        option_ids: list[str],
        class_ids: list[str],
        *,
        mode: str = "full",
        temperature: float = 1.0,
    ) -> dict[str, object]:
        """Return normalized JSON probabilities in the supplied option order."""
        if (len(option_ids) != 7 or len(class_ids) != 7
                or set(class_ids) != set(ACTION_LABELS)
                or len(set(option_ids)) != 7
                or any(not isinstance(value, str) or not value.strip()
                       for value in option_ids)):
            raise ValueError("the seven unique HapTile classes and option IDs are required")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        video_values = video_values.to(device=device, dtype=dtype)
        state_values = state_values.to(device=device, dtype=dtype)
        context_features = context_features.to(device=device, dtype=dtype)
        option_features = option_features.to(device=device, dtype=dtype)
        video_timestamps_ms = video_timestamps_ms.to(device=device)
        state_timestamps_ms = state_timestamps_ms.to(device=device, dtype=dtype)
        valid_options = torch.ones((1, 7), dtype=torch.bool, device=device)
        logits = self(
            video_values.unsqueeze(0),
            video_timestamps_ms.unsqueeze(0),
            state_values.unsqueeze(0),
            state_timestamps_ms.unsqueeze(0),
            context_features.reshape(1, 384),
            option_features.unsqueeze(0),
            valid_options,
            mode=mode,
        )[0].float() / temperature
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        options = [
            {"option_id": option_id, "class_id": class_id}
            for option_id, class_id in zip(option_ids, class_ids, strict=True)
        ]
        return classification_json(probabilities, class_ids, options)
