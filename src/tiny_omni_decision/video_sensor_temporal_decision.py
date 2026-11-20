"""Decision readout joining frozen video tokens with a causal sensor LSTM.

All visual features and typed candidate features are supplied by the caller.
This module does not load encoders or raw media and is an untrained architecture
candidate rather than evidence of task quality.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .classification_json import classification_json
from .decision import FrozenFeatureCandidateScorer
from .sensor_temporal_stem import SensorTemporalLSTMStem
from .time_aware_gated_fusion import TemporalFeatureStream, TimeAwareGatedFusion


class VideoSensorTemporalDecisionScorer(nn.Module):
    """Candidate-conditioned temporal fusion with explicit stream presence."""

    def __init__(
        self,
        video_feature_width: int,
        sensor_channels: int,
        option_width: int,
        *,
        sensor_hidden_size: int = 64,
        width: int = 96,
        heads: int = 4,
        scorer_hidden_width: int = 128,
        time_unit_ms: float = 1000.0,
    ) -> None:
        super().__init__()
        if min(video_feature_width, sensor_channels, option_width,
               sensor_hidden_size, scorer_hidden_width) < 1:
            raise ValueError("feature and hidden widths must be positive")
        if not math.isfinite(time_unit_ms) or time_unit_ms <= 0:
            raise ValueError("time_unit_ms must be finite and positive")
        self.video_feature_width = video_feature_width
        self.sensor_channels = sensor_channels
        self.option_width = option_width
        self.sensor_hidden_size = sensor_hidden_size
        self.width = width
        self.heads = heads
        self.scorer_hidden_width = scorer_hidden_width
        self.time_unit_ms = float(time_unit_ms)
        self.sensor_stem = SensorTemporalLSTMStem(
            sensor_channels, sensor_hidden_size, time_unit_ms=time_unit_ms
        )
        self.fusion = TimeAwareGatedFusion(
            {"video": video_feature_width, "sensor": sensor_hidden_size},
            option_width,
            width=width,
            heads=heads,
            time_unit_ms=time_unit_ms,
        )
        self.scorer = FrozenFeatureCandidateScorer(
            width + option_width, hidden_size=scorer_hidden_width
        )

    def _configuration(self) -> dict[str, object]:
        return {
            "schema": 1,
            "video_feature_width": self.video_feature_width,
            "sensor_channels": self.sensor_channels,
            "option_width": self.option_width,
            "sensor_hidden_size": self.sensor_hidden_size,
            "width": self.width,
            "heads": self.heads,
            "scorer_hidden_width": self.scorer_hidden_width,
            "time_unit_ms": self.time_unit_ms,
        }

    def get_extra_state(self) -> dict[str, object]:
        return self._configuration()

    def set_extra_state(self, state: object) -> None:
        if state != self._configuration():
            raise ValueError("video-sensor checkpoint configuration differs")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if state_dict.get("_extra_state") != self._configuration():
            raise ValueError("video-sensor checkpoint configuration missing or different")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        video_values: Tensor,
        video_timestamps_ms: Tensor,
        video_valid: Tensor,
        video_present: Tensor,
        sensor_values: Tensor,
        sensor_timestamps_ms: Tensor,
        sensor_valid: Tensor,
        sensor_present: Tensor,
        option_features: Tensor,
        option_valid: Tensor,
    ) -> Tensor:
        reference = self.fusion.modality.weight
        if (video_values.ndim != 3 or video_values.shape[-1] != self.video_feature_width
                or sensor_values.ndim != 3 or sensor_values.shape[-1] != self.sensor_channels
                or option_features.ndim != 3 or option_features.shape[-1] != self.option_width):
            raise ValueError("video, sensor or option feature shape does not match configuration")
        batch = video_values.shape[0]
        if sensor_values.shape[0] != batch or option_features.shape[0] != batch:
            raise ValueError("all feature batches must have the same size")
        if video_values.dtype != reference.dtype or sensor_values.dtype != reference.dtype:
            raise ValueError("video and sensor features must match model dtype")
        if option_features.dtype != reference.dtype:
            raise ValueError("option features must match model dtype")
        if any(value.device != reference.device for value in (
            video_values, video_timestamps_ms, video_valid, video_present,
            sensor_values, sensor_timestamps_ms, sensor_valid, sensor_present,
            option_features, option_valid,
        )):
            raise ValueError("all features, timestamps and masks must share model device")
        if sensor_timestamps_ms.dtype != sensor_values.dtype:
            raise ValueError("sensor values and timestamps must share dtype")
        if (sensor_valid.shape != sensor_values.shape[:2] or sensor_valid.dtype != torch.bool
                or sensor_present.shape != (batch,) or sensor_present.dtype != torch.bool):
            raise ValueError("sensor validity and presence masks have invalid shapes")
        if ((sensor_valid & ~sensor_present[:, None]).any()
                or (sensor_present & ~sensor_valid.any(dim=1)).any()):
            raise ValueError("sensor presence and valid-observation masks disagree")
        if sensor_valid.any() and not torch.isfinite(sensor_values[sensor_valid]).all():
            raise ValueError("valid sensor observations must be finite")

        rows = torch.nonzero(sensor_present, as_tuple=False).flatten()
        sensor_summary = sensor_values.new_zeros((batch, self.sensor_hidden_size))
        sensor_last_time = sensor_timestamps_ms.new_zeros((batch, 1))
        if rows.numel():
            summary = self.sensor_stem(
                sensor_values.index_select(0, rows),
                sensor_timestamps_ms.index_select(0, rows),
                sensor_valid.index_select(0, rows),
            )
            sensor_summary = sensor_summary.index_copy(0, rows, summary)
            last_times = torch.stack([
                sensor_timestamps_ms[row, sensor_valid[row]][-1] for row in rows
            ]).unsqueeze(-1)
            sensor_last_time = sensor_last_time.index_copy(0, rows, last_times)

        streams = {
            "video": TemporalFeatureStream(
                video_values, video_timestamps_ms, video_valid, video_present
            ),
            "sensor": TemporalFeatureStream(
                sensor_summary[:, None, :], sensor_last_time,
                sensor_present[:, None], sensor_present,
            ),
        }
        fused = self.fusion(streams, option_features, option_valid)
        clean_options = torch.where(
            option_valid[..., None], option_features, torch.zeros_like(option_features)
        )
        candidate_features = torch.cat((fused, clean_options), dim=-1)
        logits = self.scorer(candidate_features)
        if not torch.isfinite(logits[option_valid]).all():
            raise ValueError("video-sensor decision produced non-finite logits")
        return logits.masked_fill(~option_valid, float("-inf"))

    @torch.inference_mode()
    def decide(
        self,
        video_values: Tensor,
        video_timestamps_ms: Tensor,
        video_valid: Tensor,
        sensor_values: Tensor,
        sensor_timestamps_ms: Tensor,
        sensor_valid: Tensor,
        option_features: Tensor,
        option_ids: list[str],
        class_ids: list[str],
        *,
        temperature: float = 1.0,
    ) -> dict[str, object]:
        """Return probabilities in the supplied option order for one example."""
        if (video_values.ndim != 2 or sensor_values.ndim != 2
                or option_features.ndim != 2):
            raise ValueError("decide expects unbatched video, sensor and option matrices")
        if (len(option_ids) != len(option_features) or len(class_ids) != len(option_features)
                or len(set(option_ids)) != len(option_ids)
                or len(set(class_ids)) != len(class_ids)
                or any(not isinstance(value, str) or not value.strip()
                       for value in [*option_ids, *class_ids])):
            raise ValueError("unique nonempty option and class IDs must match option rows")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        video_values = video_values.to(device=device, dtype=dtype)
        option_features = option_features.to(device=device, dtype=dtype)
        sensor_values = sensor_values.to(device=device, dtype=dtype)
        video_timestamps_ms = video_timestamps_ms.to(device=device)
        sensor_timestamps_ms = sensor_timestamps_ms.to(device=device, dtype=dtype)
        video_valid = video_valid.to(device=device, dtype=torch.bool)
        sensor_valid = sensor_valid.to(device=device, dtype=torch.bool)
        video_present = video_valid.any().reshape(1)
        sensor_present = sensor_valid.any().reshape(1)
        logits = self(
            video_values.unsqueeze(0), video_timestamps_ms.unsqueeze(0),
            video_valid.unsqueeze(0), video_present,
            sensor_values.unsqueeze(0), sensor_timestamps_ms.unsqueeze(0),
            sensor_valid.unsqueeze(0), sensor_present,
            option_features.unsqueeze(0),
            torch.ones((1, len(option_features)), dtype=torch.bool, device=device),
        )[0].float() / temperature
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        options = [
            {"option_id": option_id, "class_id": class_id}
            for option_id, class_id in zip(option_ids, class_ids, strict=True)
        ]
        return classification_json(probabilities, class_ids, options)
