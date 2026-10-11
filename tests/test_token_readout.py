import pytest

torch = pytest.importorskip("torch")


def test_candidate_equivariance_and_frozen_tokens():
    from tiny_omni_decision.token_readout import VideoTokenReadout

    torch.manual_seed(17)
    model = VideoTokenReadout()
    tokens = torch.randn(2, 2048, 768)
    text = torch.randn(2, 3, 1536)
    values = model(tokens, text)
    torch.testing.assert_close(model(tokens, text.flip(1)), values.flip(1))
    values.sum().backward()
    assert tokens.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not torch.allclose(model(tokens.flip(1), text), values)


def test_mean_control_and_no_video():
    from tiny_omni_decision.token_readout import VideoTokenReadout

    tokens, text = torch.randn(1, 2048, 768), torch.randn(1, 2, 1536)
    model = VideoTokenReadout("mean")
    expected = (model.value(tokens.mean(1))[:, None] * model.query(text)).sum(-1) / model.scale
    torch.testing.assert_close(model(tokens, text), expected)
    model.mode = "no_video"
    torch.testing.assert_close(model(tokens, text), model(tokens * 5, text))
    with pytest.raises(ValueError):
        model(tokens[:, :1], text)
