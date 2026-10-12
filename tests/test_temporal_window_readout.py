import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("mode", ["mean", "attention"])
def test_padding_and_option_permutation(mode):
    from tiny_omni_decision.temporal_window_readout import TemporalWindowReadout

    torch.manual_seed(17)
    model = TemporalWindowReadout(mode)
    video = torch.randn(1, 3, 768)
    text = torch.randn(1, 3, 1536)
    mask = torch.ones(1, 3, dtype=torch.bool)
    expected = model(video, text, mask)
    padded = torch.cat([video, torch.randn(1, 2, 768)], 1)
    padded_mask = torch.tensor([[True, True, True, False, False]])
    torch.testing.assert_close(expected, model(padded, text, padded_mask))
    torch.testing.assert_close(expected.flip(1), model(video, text.flip(1), mask))
    expected.sum().backward()
    assert model.query.weight.grad.isfinite().all()
    assert model.value.weight.grad.isfinite().all()


def test_temporal_order_and_empty_mask():
    from tiny_omni_decision.temporal_window_readout import TemporalWindowReadout

    torch.manual_seed(17)
    video, text = torch.randn(1, 5, 768), torch.randn(1, 3, 1536)
    mask = torch.ones(1, 5, dtype=torch.bool)
    mean = TemporalWindowReadout("mean")
    torch.testing.assert_close(mean(video, text, mask), mean(video.flip(1), text, mask))
    attention = TemporalWindowReadout()
    assert not torch.allclose(
        attention(video, text, mask), attention(video.flip(1), text, mask)
    )
    with pytest.raises(ValueError, match="empty"):
        attention(video, text, ~mask)
