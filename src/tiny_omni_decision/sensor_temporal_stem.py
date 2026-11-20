"""Small causal sensor sequence encoder; normalization and calibration are upstream."""

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from .native_feature_decision import NativeFeatureDecisionScorer


class SensorTemporalLSTMStem(nn.Module):
    """Encode calibrated sensor windows into one causal state vector.

    ``values`` has shape ``[batch, time, channels]`` and should already use
    physically calibrated, training-derived normalized channel values.
    ``valid`` identifies real observations, not zero-valued measurements.
    Invalid padded values and timestamps are ignored. Log-scaled elapsed time
    between valid observations is appended as an input feature.

    This is an M-1 architecture primitive, not an end-to-end sensor Decision
    model. It does not infer units, calibration, clock alignment, or quality.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int = 64,
        *,
        time_unit_ms: float = 1000.0,
    ) -> None:
        super().__init__()
        if input_channels < 1 or hidden_size < 1:
            raise ValueError("input_channels and hidden_size must be positive")
        if not math.isfinite(time_unit_ms) or time_unit_ms <= 0:
            raise ValueError("time_unit_ms must be finite and positive")
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.time_unit_ms = float(time_unit_ms)
        self.lstm = nn.LSTM(
            input_size=input_channels + 1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, values: Tensor, timestamps_ms: Tensor, valid: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_channels:
            raise ValueError("values must have shape [batch, time, configured_channels]")
        if values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("nonempty batch and time dimensions are required")
        if timestamps_ms.shape != values.shape[:2] or valid.shape != values.shape[:2]:
            raise ValueError("timestamps and validity mask must match batch/time dimensions")
        if valid.dtype != torch.bool:
            raise ValueError("validity mask must be boolean")
        if (timestamps_ms.device != values.device or valid.device != values.device
                or timestamps_ms.dtype != values.dtype):
            raise ValueError("values, timestamps, and mask must share device; timestamps dtype too")

        sequences = []
        lengths = []
        for row in range(values.shape[0]):
            row_mask = valid[row]
            sequence = values[row, row_mask]
            times = timestamps_ms[row, row_mask]
            if sequence.shape[0] == 0:
                raise ValueError("every batch row requires at least one valid sensor observation")
            if not torch.isfinite(sequence).all() or not torch.isfinite(times).all():
                raise ValueError("valid sensor values and timestamps must be finite")
            elapsed = times[1:] - times[:-1]
            if (elapsed <= 0).any():
                raise ValueError("valid timestamps must be strictly increasing")
            delta = torch.cat((times.new_zeros(1), elapsed))
            time_feature = torch.log1p(delta / self.time_unit_ms).unsqueeze(-1)
            sequences.append(torch.cat((sequence, time_feature), dim=-1))
            lengths.append(sequence.shape[0])

        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]


class SensorTemporalDecisionScorer(nn.Module):
    """Causal sensor-window encoder with candidate-conditioned Decision JSON.

    ``option_features`` must come from a caller-owned typed option schema; IDs,
    labels, and free-form text are not model features. This class provides a
    runnable sensor-only M1 path while leaving calibration, normalization,
    freshness checks, units, and abstention policy to the caller.
    """

    def __init__(
        self,
        input_channels: int,
        option_width: int,
        *,
        hidden_size: int = 64,
        scorer_hidden_width: int = 128,
        time_unit_ms: float = 1000.0,
    ) -> None:
        super().__init__()
        self.stem = SensorTemporalLSTMStem(
            input_channels, hidden_size, time_unit_ms=time_unit_ms
        )
        self.readout = NativeFeatureDecisionScorer(
            hidden_size, option_width, hidden_width=scorer_hidden_width
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        values: Tensor,
        timestamps_ms: Tensor,
        valid: Tensor,
        option_features: Tensor,
        option_valid: Tensor,
    ) -> Tensor:
        state = self.stem(values, timestamps_ms, valid)
        return self.readout(state, option_features, option_valid)

    @torch.inference_mode()
    def decide(
        self,
        values: Tensor,
        timestamps_ms: Tensor,
        valid: Tensor,
        option_features: Tensor,
        option_ids: list[str],
        class_ids: list[str],
        *,
        temperature: float = 1.0,
    ) -> dict[str, object]:
        """Return caller-ordered probabilities for one sensor window."""
        if values.ndim != 2 or timestamps_ms.ndim != 1 or valid.ndim != 1:
            raise ValueError("decide expects [time, channels] values and [time] timestamps/mask")
        if timestamps_ms.shape[0] != values.shape[0] or valid.shape[0] != values.shape[0]:
            raise ValueError("sensor values, timestamps and mask must align")
        if option_features.ndim != 2:
            raise ValueError("option features must have shape [candidates, option_width]")
        state = self.stem(
            values.unsqueeze(0), timestamps_ms.unsqueeze(0), valid.unsqueeze(0)
        )[0]
        return self.readout.decide(
            state,
            option_features,
            option_ids,
            class_ids,
            temperature=temperature,
        )
