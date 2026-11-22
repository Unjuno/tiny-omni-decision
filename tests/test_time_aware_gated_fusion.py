import copy
import io

import pytest

torch = pytest.importorskip("torch")


def fixture(*, use_time=True, use_gates=True):
    from tiny_omni_decision.time_aware_gated_fusion import (
        TemporalFeatureStream,
        TimeAwareGatedFusion,
    )

    torch.manual_seed(2718)
    model = TimeAwareGatedFusion(
        {"audio": 2, "sensor": 3}, 5, width=16, heads=4,
        use_time=use_time, use_gates=use_gates,
    )
    streams = {
        "audio": TemporalFeatureStream(
            torch.randn(2, 4, 2),
            torch.tensor([[0., 10., 20., 30.], [5., 15., 25., 35.]]),
            torch.tensor([[True, True, True, True], [True, True, True, True]]),
            torch.tensor([True, True]),
        ),
        "sensor": TemporalFeatureStream(
            torch.randn(2, 3, 3),
            torch.tensor([[2., 17., 32.], [7., 22., 37.]]),
            torch.tensor([[True, True, True], [True, True, True]]),
            torch.tensor([True, True]),
        ),
    }
    queries = torch.randn(2, 3, 5)
    option_valid = torch.tensor([[True, True, False], [True, True, True]])
    return model, TemporalFeatureStream, streams, queries, option_valid


def test_candidate_states_are_finite_masked_and_candidate_equivariant():
    model, _, streams, queries, option_valid = fixture()
    output, diagnostics = model(streams, queries, option_valid, diagnostics=True)
    assert output.shape == (2, 3, 16)
    assert torch.isfinite(output[option_valid]).all()
    assert torch.equal(output[~option_valid], torch.zeros_like(output[~option_valid]))
    assert diagnostics["gates"]["audio"].shape == (2, 3, 1)
    assert diagnostics["gates"]["sensor"].shape == (2, 3, 1)
    assert model.parameter_ledger()["trainable_parameters"] == model.parameter_ledger()[
        "total_parameters"
    ]

    order = torch.tensor([2, 0, 1])
    permuted = model(streams, queries[:, order], option_valid[:, order])
    torch.testing.assert_close(permuted, output[:, order], rtol=0, atol=1e-6)


def test_default_audio_sensor_parameter_ledger_matches_report():
    from tiny_omni_decision.time_aware_gated_fusion import TimeAwareGatedFusion

    model = TimeAwareGatedFusion({"audio": 1, "sensor": 3}, 5)
    assert model.parameter_ledger() == {
        "total_parameters": 141_986,
        "trainable_parameters": 141_986,
    }


def test_padding_is_ignored_and_missing_is_distinct_from_zero():
    model, stream_type, streams, queries, option_valid = fixture()
    baseline = model(streams, queries, option_valid)
    padded = copy.copy(streams)
    audio = streams["audio"]
    padded["audio"] = stream_type(
        torch.cat((audio.values, torch.full((2, 2, 2), 1e8)), dim=1),
        torch.cat((audio.timestamps_ms, torch.full((2, 2), 1e8)), dim=1),
        torch.cat((audio.valid, torch.zeros(2, 2, dtype=torch.bool)), dim=1),
        audio.present,
    )
    torch.testing.assert_close(model(padded, queries, option_valid), baseline,
                               rtol=0, atol=5e-7)

    absent = copy.copy(streams)
    sensor = streams["sensor"]
    absent["sensor"] = stream_type(
        torch.zeros_like(sensor.values), sensor.timestamps_ms,
        torch.zeros_like(sensor.valid), torch.zeros_like(sensor.present),
    )
    zero_but_present = copy.copy(streams)
    zero_but_present["sensor"] = stream_type(
        torch.zeros_like(sensor.values), sensor.timestamps_ms,
        sensor.valid, sensor.present,
    )
    absent_output = model(absent, queries, option_valid)
    zero_output = model(zero_but_present, queries, option_valid)
    assert not torch.allclose(absent_output, zero_output, atol=1e-7, rtol=0)


def test_relative_timing_and_gate_ablations_change_the_candidate_state():
    model, stream_type, streams, queries, option_valid = fixture()
    baseline = model(streams, queries, option_valid)
    shifted = copy.copy(streams)
    sensor = streams["sensor"]
    shifted["sensor"] = stream_type(
        sensor.values, sensor.timestamps_ms + 500., sensor.valid, sensor.present
    )
    assert not torch.allclose(model(shifted, queries, option_valid), baseline,
                              atol=1e-7, rtol=0)

    fixed_gate, _, _, _, _ = fixture(use_gates=False)
    assert not torch.allclose(fixed_gate(streams, queries, option_valid), baseline,
                              atol=1e-7, rtol=0)
    assert all(not parameter.requires_grad
               for parameter in fixed_gate.gates.parameters())
    no_time, _, _, _, _ = fixture(use_time=False)
    assert all(not parameter.requires_grad
               for parameter in no_time.time_encoder.parameters())


def test_gradients_reach_both_streams_attention_and_gates():
    model, _, streams, queries, option_valid = fixture()
    output = model(streams, queries, option_valid)
    output[option_valid].square().mean().backward()
    for name in ("audio", "sensor"):
        assert streams[name].values.grad is None  # inputs are immutable leaf fixtures
        assert model.projections[name].weight.grad is not None
        assert torch.isfinite(model.projections[name].weight.grad).all()
        assert model.gates[name][0].weight.grad is not None
        assert model.gates[name][0].weight.grad.abs().sum() > 0
    assert model.attention.in_proj_weight.grad is not None


def test_checkpoint_roundtrip_and_configuration_gate():
    model, _, streams, queries, option_valid = fixture()
    expected = model(streams, queries, option_valid)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored, _, _, _, _ = fixture()
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored(streams, queries, option_valid), expected,
                               rtol=0, atol=0)

    mismatched, _, _, _, _ = fixture(use_time=False)
    before = {key: value.clone() for key, value in mismatched.state_dict().items()
              if isinstance(value, torch.Tensor)}
    with pytest.raises(ValueError, match="configuration"):
        mismatched.load_state_dict(state, strict=False)
    for key, value in before.items():
        torch.testing.assert_close(mismatched.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["all_missing", "inconsistent_presence", "time_order"])
def test_invalid_temporal_masks_fail_closed(case):
    model, stream_type, streams, queries, option_valid = fixture()
    broken = copy.copy(streams)
    if case == "all_missing":
        broken = {
            name: stream_type(stream.values, stream.timestamps_ms,
                              torch.zeros_like(stream.valid), torch.zeros_like(stream.present))
            for name, stream in streams.items()
        }
    elif case == "inconsistent_presence":
        audio = streams["audio"]
        broken["audio"] = stream_type(audio.values, audio.timestamps_ms,
                                      audio.valid, torch.tensor([False, True]))
    else:
        audio = streams["audio"]
        times = audio.timestamps_ms.clone()
        times[0, 2] = times[0, 1]
        broken["audio"] = stream_type(audio.values, times, audio.valid, audio.present)
    with pytest.raises(ValueError):
        model(broken, queries, option_valid)


@pytest.mark.parametrize("kwargs", [
    {"dimensions": {}},
    {"dimensions": {"audio": 0}},
    {"dimensions": {"": 2}},
    {"dimensions": {"not.a.module.key": 2}},
    {"dimensions": {"audio": 2}, "width": 10, "heads": 3},
    {"dimensions": {"audio": 2}, "use_gates": 1},
])
def test_invalid_fusion_configuration(kwargs):
    from tiny_omni_decision.time_aware_gated_fusion import TimeAwareGatedFusion

    arguments = {"dimensions": {"audio": 2}, "query_dim": 5, "width": 16, "heads": 4}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        TimeAwareGatedFusion(**arguments)
