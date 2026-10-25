import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.synthetic_av_readouts import (  # noqa: E402
    BypassConditionedTemporalAVReadout,
    ConditionalAVReadout,
    ConditionedTemporalAVReadout,
    TemporalLSTMReadout,
)


def _inputs(batch_size=4):
    return (
        torch.randn(batch_size, 8, 384),
        torch.randn(batch_size, 8, 768),
        torch.nn.functional.one_hot(torch.arange(batch_size) % 2, num_classes=2).float(),
    )


def test_temporal_lstm_and_conditional_fusion_return_binary_logits_and_backpropagate():
    torch.manual_seed(17)
    audio, video, query = (value.requires_grad_() for value in _inputs())
    for model in (
        TemporalLSTMReadout(384, 768, 2),
        ConditionalAVReadout(384, 768, 2),
        ConditionedTemporalAVReadout(384, 768, 2),
        BypassConditionedTemporalAVReadout(384, 768, 2),
    ):
        logits = model(audio, video, query)
        assert logits.shape == (4, 2)
        logits.square().mean().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        assert audio.grad is not None and torch.count_nonzero(audio.grad) > 0
        assert video.grad is not None and torch.count_nonzero(video.grad) > 0
        assert query.grad is not None and torch.count_nonzero(query.grad) > 0
        audio.grad = video.grad = query.grad = None


def test_readout_parameter_counts_are_within_ten_percent():
    temporal = TemporalLSTMReadout(384, 768, 2)
    conditional = ConditionalAVReadout(384, 768, 2)
    conditioned_temporal = ConditionedTemporalAVReadout(384, 768, 2)
    bypass_conditioned_temporal = BypassConditionedTemporalAVReadout(384, 768, 2)
    def count(model):
        return sum(parameter.numel() for parameter in model.parameters())

    baseline = count(temporal)
    assert abs(baseline - count(conditional)) / baseline <= 0.1
    assert abs(baseline - count(conditioned_temporal)) / baseline <= 0.1
    assert count(conditioned_temporal) == 528388
    assert abs(baseline - count(bypass_conditioned_temporal)) / baseline <= 0.1
    assert count(bypass_conditioned_temporal) == 521476


def test_same_seed_produces_identical_initial_weights():
    torch.manual_seed(17)
    first = BypassConditionedTemporalAVReadout(384, 768, 2)
    torch.manual_seed(17)
    second = BypassConditionedTemporalAVReadout(384, 768, 2)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    )


def test_bypass_path_preserves_video_gradient_when_gate_collapses_to_audio():
    model = BypassConditionedTemporalAVReadout(384, 768, 2)
    with torch.no_grad():
        model.modality_gate[-1].bias.copy_(torch.tensor([20.0, -20.0]))
    audio, video, query = _inputs(batch_size=2)
    model(audio, video, query).square().mean().backward()
    video_projection_gradient = model.video_projection[0].weight.grad
    assert video_projection_gradient is not None
    assert torch.count_nonzero(video_projection_gradient) > 0
