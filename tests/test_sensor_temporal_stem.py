from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)


def test_sensor_temporal_stem_is_causal_compact_and_finite():
    from tiny_omni_decision.sensor_temporal_stem import SensorTemporalLSTMStem

    torch.manual_seed(17)
    stem = SensorTemporalLSTMStem(input_channels=12, hidden_size=64).eval()
    values = torch.randn(3, 9, 12)
    timestamps = torch.arange(9, dtype=torch.float32).repeat(3, 1) * 20
    valid = torch.tensor(
        [[True] * 9, [True] * 5 + [False] * 4, [False, True] * 4 + [False]]
    )

    result = stem(values, timestamps, valid)

    assert result.shape == (3, 64)
    assert torch.isfinite(result).all()
    assert sum(parameter.numel() for parameter in stem.parameters()) < 1_000_000


def test_sensor_temporal_stem_ignores_invalid_padded_values_and_times():
    from tiny_omni_decision.sensor_temporal_stem import SensorTemporalLSTMStem

    torch.manual_seed(23)
    stem = SensorTemporalLSTMStem(input_channels=2, hidden_size=8).eval()
    values = torch.tensor([[[0.1, 0.2], [0.3, -0.4], [0.7, 0.8]]])
    timestamps = torch.tensor([[10.0, 30.0, 60.0]])
    valid = torch.tensor([[True, False, True]])
    padded_values = torch.tensor([[[0.1, 0.2], [float("nan"), float("inf")], [0.7, 0.8]]])
    padded_timestamps = torch.tensor([[10.0, float("nan"), 60.0]])

    expected = stem(values, timestamps, valid)
    actual = stem(padded_values, padded_timestamps, valid)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sensor_temporal_stem_propagates_training_gradients():
    from tiny_omni_decision.sensor_temporal_stem import SensorTemporalLSTMStem

    stem = SensorTemporalLSTMStem(input_channels=3, hidden_size=8)
    values = torch.randn(2, 4, 3, requires_grad=True)
    times = torch.tensor([[0.0, 10.0, 20.0, 30.0]]).repeat(2, 1)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])

    stem(values, times, valid).square().mean().backward()

    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert all(parameter.grad is not None for parameter in stem.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in stem.parameters())


@pytest.mark.parametrize(
    "timestamps,valid",
    [
        ([[1.0, 1.0]], [[True, True]]),
        ([[2.0, 1.0]], [[True, True]]),
        ([[1.0, 2.0]], [[False, False]]),
    ],
)
def test_sensor_temporal_stem_rejects_bad_valid_time_sequences(timestamps, valid):
    from tiny_omni_decision.sensor_temporal_stem import SensorTemporalLSTMStem

    stem = SensorTemporalLSTMStem(input_channels=1, hidden_size=4)
    with pytest.raises(ValueError):
        stem(torch.zeros(1, 2, 1), torch.tensor(timestamps), torch.tensor(valid))
