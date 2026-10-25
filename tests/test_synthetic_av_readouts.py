import torch

from tiny_omni_decision.synthetic_av_readouts import (
    ConditionalAVReadout,
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
    for model in (TemporalLSTMReadout(384, 768, 2), ConditionalAVReadout(384, 768, 2)):
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
    def count(model):
        return sum(parameter.numel() for parameter in model.parameters())

    first, second = count(temporal), count(conditional)
    assert abs(first - second) / max(first, second) <= 0.1


def test_same_seed_produces_identical_initial_weights():
    torch.manual_seed(17)
    first = ConditionalAVReadout(384, 768, 2)
    torch.manual_seed(17)
    second = ConditionalAVReadout(384, 768, 2)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    )
