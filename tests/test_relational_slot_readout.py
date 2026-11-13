import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.relational_slot_readout import RelationalSlotReadout  # noqa: E402


def test_relational_head_equivariance_gradients_and_parameter_budget():
    torch.manual_seed(17)
    model = RelationalSlotReadout()
    x = torch.randn(2, 14, 129)
    c = torch.randn(2, 3, 1536)
    result = model(x, c)
    assert result.shape == (2, 3)
    torch.testing.assert_close(model(x.flip(1), c), result)
    torch.testing.assert_close(model(x, c.flip(1)), result.flip(1))
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) < 200000


def test_padding_cannot_affect_prediction():
    model = RelationalSlotReadout().eval()
    x = torch.randn(1, 5, 129)
    c = torch.randn(1, 3, 1536)
    padded = torch.cat([x, torch.randn(1, 2, 129) * 100], dim=1)
    mask = torch.tensor([[True] * 5 + [False] * 2])
    torch.testing.assert_close(model(x, c), model(padded, c, mask))
