import pytest

torch = pytest.importorskip("torch")
from scripts.source_balanced_set import balanced_set_loss  # noqa: E402


def test_equal_positive_negative_gradient_mass_and_mask():
    logits = torch.zeros((1, 6), requires_grad=True)
    mask = torch.tensor([[True, True, True, True, False]])
    targets = torch.tensor([[True, False, False, False, False, False]])
    balanced_set_loss(logits, mask, targets).backward()
    assert torch.allclose(-logits.grad[0, 0], logits.grad[0, 1:4].sum())
    assert logits.grad[0, 4] == 0 and logits.grad[0, 5] > 0


def test_none_and_all_positive_are_finite():
    logits = torch.zeros((2, 3), requires_grad=True)
    mask = torch.tensor([[True, True], [True, True]])
    targets = torch.tensor([[False, False, True], [True, True, False]])
    loss = balanced_set_loss(logits, mask, targets)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, :2].abs().sum() == 0
    with pytest.raises(ValueError):
        balanced_set_loss(logits, mask, torch.tensor([[True, False, True], [True, True, False]]))
