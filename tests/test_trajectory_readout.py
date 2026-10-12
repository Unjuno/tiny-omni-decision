import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.trajectory_readout import TrajectoryReadout  # noqa: E402


def test_equivariance_and_gradient():
    torch.manual_seed(17)
    model = TrajectoryReadout()
    points = torch.randn(2, 7, 512)
    candidates = torch.randn(2, 3, 1536)
    scores = model(points, candidates)
    torch.testing.assert_close(model(points.flip(1), candidates), scores)
    torch.testing.assert_close(model(points, candidates.flip(1)), scores.flip(1))
    scores.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


def test_padding_is_inert():
    model = TrajectoryReadout()
    points = torch.randn(1, 2, 512)
    candidates = torch.randn(1, 3, 1536)
    padded = torch.cat([points, torch.full((1, 3, 512), 999.0)], dim=1)
    torch.testing.assert_close(
        model(points, candidates),
        model(padded, candidates, torch.tensor([[True, True, False, False, False]])),
    )


def test_invalid_input_rejected():
    model = TrajectoryReadout()
    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 511), torch.randn(1, 3, 1536))
    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 512), torch.randn(1, 3, 1536), torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        model(torch.full((1, 2, 512), float("nan")), torch.randn(1, 3, 1536))
