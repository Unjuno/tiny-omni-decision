import io

import pytest

torch = pytest.importorskip("torch")


def build_fixture():
    from tiny_omni_decision.video_sensor_temporal_decision import (
        VideoSensorTemporalDecisionScorer,
    )

    torch.manual_seed(1701)
    model = VideoSensorTemporalDecisionScorer(
        video_feature_width=6, sensor_channels=3, option_width=4,
        context_width=2, sensor_hidden_size=5, width=8, heads=2, scorer_hidden_width=7,
    )
    video = torch.randn(2, 3, 6)
    video_times = torch.tensor([[0., 20., 40.], [10., 30., 50.]])
    video_valid = torch.tensor([[True, True, True], [True, True, True]])
    video_present = torch.tensor([True, True])
    sensor = torch.randn(2, 4, 3)
    sensor_times = torch.tensor([[1., 11., 21., 31.], [12., 22., 32., 42.]])
    sensor_valid = torch.tensor([[True, True, True, True], [True, True, True, True]])
    sensor_present = torch.tensor([True, True])
    options = torch.randn(2, 3, 4)
    option_valid = torch.tensor([[True, True, False], [True, True, True]])
    return (model, video, video_times, video_valid, video_present,
            sensor, sensor_times, sensor_valid, sensor_present, options, option_valid)


def test_video_sensor_logits_are_masked_candidate_equivariant_and_trainable():
    (model, video, video_times, video_valid, video_present,
     sensor, sensor_times, sensor_valid, sensor_present, options, option_valid) = build_fixture()
    logits = model(video, video_times, video_valid, video_present,
                   sensor, sensor_times, sensor_valid, sensor_present,
                   options, option_valid)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits[option_valid]).all()
    assert torch.isneginf(logits[~option_valid]).all()

    order = torch.tensor([2, 0, 1])
    permuted = model(video, video_times, video_valid, video_present,
                     sensor, sensor_times, sensor_valid, sensor_present,
                     options[:, order], option_valid[:, order])
    torch.testing.assert_close(permuted, logits[:, order], rtol=0, atol=1e-6)

    torch.logsumexp(logits[0, :2], dim=0).backward()
    assert model.sensor_stem.lstm.weight_ih_l0.grad is not None
    assert model.fusion.attention.in_proj_weight.grad is not None
    assert model.scorer.network[0].weight.grad is not None


def test_same_video_and_options_accept_distinct_typed_decision_contexts():
    (model, video, video_times, video_valid, video_present,
     sensor, sensor_times, sensor_valid, sensor_present, options, option_valid) = build_fixture()
    context = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    first = model(
        video, video_times, video_valid, video_present,
        sensor, sensor_times, sensor_valid, sensor_present,
        options, option_valid, context_features=context,
    )
    changed_context = context.clone()
    changed_context[0] = torch.tensor([1.0, 0.0])
    second = model(
        video, video_times, video_valid, video_present,
        sensor, sensor_times, sensor_valid, sensor_present,
        options, option_valid, context_features=changed_context,
    )
    assert not torch.allclose(first[0, :2], second[0, :2], rtol=0, atol=1e-7)


def test_missing_sensor_is_distinct_from_present_zero_and_json_keeps_option_ids():
    (model, video, video_times, video_valid, video_present,
     sensor, sensor_times, sensor_valid, sensor_present, options, option_valid) = build_fixture()
    absent_valid = sensor_valid.clone()
    absent_valid[0] = False
    absent_present = sensor_present.clone()
    absent_present[0] = False
    absent = model(video, video_times, video_valid, video_present,
                   sensor, sensor_times, absent_valid, absent_present,
                   options, option_valid)

    zero_sensor = torch.zeros_like(sensor)
    present_zero = model(video, video_times, video_valid, video_present,
                         zero_sensor, sensor_times, sensor_valid, sensor_present,
                         options, option_valid)
    assert not torch.allclose(absent[0, :2], present_zero[0, :2], rtol=0, atol=1e-7)

    result = model.decide(
        video[0], video_times[0], video_valid[0],
        sensor[0], sensor_times[0], sensor_valid[0],
        options[0, :2], ["opt-b", "opt-a"], ["B", "A"],
    )
    assert [option["option_id"] for option in result["options"]] == ["opt-b", "opt-a"]
    assert sum(option["probability"] for option in result["options"]) == pytest.approx(1.0)
    selected_index = next(i for i, option in enumerate(result["options"])
                          if option["option_id"] == result["selected_option_id"])
    assert result["options"][selected_index]["option_id"] in {"opt-b", "opt-a"}


def test_video_only_and_sensor_only_are_supported_but_both_missing_is_rejected():
    (model, video, video_times, video_valid, video_present,
     sensor, sensor_times, sensor_valid, sensor_present, options, option_valid) = build_fixture()
    no_sensor = torch.zeros_like(sensor_valid)
    no_sensor_present = torch.zeros_like(sensor_present)
    video_only = model(video, video_times, video_valid, video_present,
                       sensor, sensor_times, no_sensor, no_sensor_present,
                       options, option_valid)
    no_video = torch.zeros_like(video_valid)
    no_video_present = torch.zeros_like(video_present)
    sensor_only = model(video, video_times, no_video, no_video_present,
                        sensor, sensor_times, sensor_valid, sensor_present,
                        options, option_valid)
    assert torch.isfinite(video_only[option_valid]).all()
    assert torch.isfinite(sensor_only[option_valid]).all()
    with pytest.raises(ValueError, match="at least one"):
        model(video, video_times, no_video, no_video_present,
              sensor, sensor_times, no_sensor, no_sensor_present,
              options, option_valid)


def test_video_sensor_checkpoint_roundtrip_preserves_logits_and_config_gate():
    (model, video, video_times, video_valid, video_present,
     sensor, sensor_times, sensor_valid, sensor_present, options, option_valid) = build_fixture()
    expected = model(video, video_times, video_valid, video_present,
                     sensor, sensor_times, sensor_valid, sensor_present,
                     options, option_valid)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored, *_ = build_fixture()
    restored.load_state_dict(state, strict=True)
    actual = restored(video, video_times, video_valid, video_present,
                      sensor, sensor_times, sensor_valid, sensor_present,
                      options, option_valid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    mismatched = type(model)(
        video_feature_width=6, sensor_channels=3, option_width=4,
        context_width=2, sensor_hidden_size=6, width=8, heads=2, scorer_hidden_width=7,
    )
    with pytest.raises(ValueError, match="configuration"):
        mismatched.load_state_dict(state, strict=False)


def test_parameter_ledger_counts_only_this_candidate_readout():
    (model, *_rest) = build_fixture()
    assert model.parameter_ledger() == {
        "sensor_temporal_stem": 220,
        "temporal_fusion": 1338,
        "candidate_scorer": 113,
        "total_readout_parameters": 1671,
        "trainable_readout_parameters": 1671,
    }
