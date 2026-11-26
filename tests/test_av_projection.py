import pytest

torch = pytest.importorskip("torch")

from scripts.smoke_av_correspondence import CorrespondenceProjection  # noqa: E402


def test_projection_equivariance_zero_and_gradient():
    torch.manual_seed(17)
    model = CorrespondenceProjection()
    a, v = torch.randn(4, 384), torch.randn(4, 4, 768)
    logits = model(a, v)
    assert sum(p.numel() for p in model.parameters()) == 73728
    torch.testing.assert_close(model(a, v.flip(1)).flip(1), logits)
    assert torch.equal(model(torch.zeros_like(a), v), torch.zeros(4, 4))
    assert torch.equal(model(a, torch.zeros_like(v)), torch.zeros(4, 4))
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
