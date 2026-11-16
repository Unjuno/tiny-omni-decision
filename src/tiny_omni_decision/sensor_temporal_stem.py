"""Small causal sensor sequence encoder; normalization and calibration are upstream."""

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


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
