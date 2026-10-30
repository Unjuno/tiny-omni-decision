import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.supervised_trajectory_readout import (  # noqa: E402
    SupervisedTrajectoryReadout,
)


def test_shared_qa_cover_gradients_equivariance_and_reload():
    torch.manual_seed(17)
    model = SupervisedTrajectoryReadout(hidden=8)
    points, text = torch.randn(2, 7, 512), torch.randn(2, 6, 1536)
    qa, cover = model(points, text)
    assert qa.shape == (2, 6) and cover.shape == (2, 7)
    qa2, cover2 = model(points.flip(1), text.flip(1))
    torch.testing.assert_close(qa2, qa.flip(1))
    torch.testing.assert_close(cover2, cover.flip(1))
    qa.square().mean().backward(retain_graph=True)
    assert model.cover.point.weight.grad.abs().sum() > 0
    model.zero_grad()
    cover.square().mean().backward()
    assert model.cover.point.weight.grad.abs().sum() > 0
    restored = SupervisedTrajectoryReadout(hidden=8)
    restored.load_state_dict(model.state_dict())
    a, b = restored(points, text)
    torch.testing.assert_close(a, qa, rtol=0, atol=0)
    torch.testing.assert_close(b, cover, rtol=0, atol=0)


def test_padding_and_input_guards():
    model = SupervisedTrajectoryReadout()
    points, text = torch.randn(1, 2, 512), torch.randn(1, 3, 1536)
    qa, cover = model(points, text)
    padded = torch.cat([points, torch.full((1, 2, 512), 99.0)], dim=1)
    mask = torch.tensor([[True, True, False, False]])
    a, b = model(padded, text, mask)
    torch.testing.assert_close(a, qa)
    torch.testing.assert_close(b[:, :2], cover)
    assert torch.isneginf(b[:, 2:]).all()
    for bad in [
        torch.zeros(2, 3, 1536),
        torch.zeros(1, 0, 1536),
        torch.full((1, 3, 1536), float("nan")),
    ]:
        with pytest.raises(ValueError):
            model(points, bad)
