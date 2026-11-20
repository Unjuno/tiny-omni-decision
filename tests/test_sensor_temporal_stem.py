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


def test_sensor_temporal_decision_scorer_is_candidate_equivariant():
    from tiny_omni_decision.sensor_temporal_stem import SensorTemporalDecisionScorer

    torch.manual_seed(31)
    model = SensorTemporalDecisionScorer(
        input_channels=2, option_width=2, hidden_size=8, scorer_hidden_width=16
    ).eval()
    values = torch.randn(1, 5, 2)
    timestamps = torch.arange(5, dtype=torch.float32)[None] * 10
    valid = torch.ones(1, 5, dtype=torch.bool)
    options = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    option_valid = torch.ones(1, 3, dtype=torch.bool)
    order = torch.tensor([2, 0, 1])

    original = model(values, timestamps, valid, options, option_valid)
    permuted = model(values, timestamps, valid, options[:, order], option_valid[:, order])

    # Candidate batching may change FP32 reduction order by a few ULPs.
    torch.testing.assert_close(permuted, original[:, order], rtol=0, atol=1e-7)


def test_sensor_temporal_decision_scorer_returns_candidate_id_json():
    from tiny_omni_decision.sensor_temporal_stem import SensorTemporalDecisionScorer

    torch.manual_seed(37)
    model = SensorTemporalDecisionScorer(
        input_channels=2, option_width=1, hidden_size=8, scorer_hidden_width=16
    ).eval()

    result = model.decide(
        torch.randn(5, 2),
        torch.arange(5, dtype=torch.float32) * 25,
        torch.ones(5, dtype=torch.bool),
        torch.tensor([[0.0], [1.0], [2.0]]),
        ["caller-z", "caller-a", "caller-k"],
        ["stop", "walk", "wait"],
    )

    assert result["status"] == "selected"
    assert [item["option_id"] for item in result["options"]] == [
        "caller-z", "caller-a", "caller-k"
    ]
    assert sum(item["probability"] for item in result["options"]) == pytest.approx(1.0)


def test_sensor_temporal_decision_path_stays_below_one_million_parameters():
    from tiny_omni_decision.sensor_temporal_stem import SensorTemporalDecisionScorer

    model = SensorTemporalDecisionScorer(
        input_channels=12, option_width=8, hidden_size=64, scorer_hidden_width=128
    )

    assert model.stem.parameter_count == 20_224
    assert model.parameter_count == 29_697


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
