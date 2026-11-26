import pytest

torch = pytest.importorskip("torch")


def test_cumulative_ordinal_loss_penalizes_distant_errors_more_than_adjacent_errors():
    from tiny_omni_decision.ordinal import cumulative_ordinal_nll

    target = torch.tensor([3])
    adjacent_wrong = torch.full((1, 10), -8.0)
    distant_wrong = torch.full((1, 10), -8.0)
    adjacent_wrong[0, 4] = 8.0
    distant_wrong[0, 9] = 8.0
    assert cumulative_ordinal_nll(adjacent_wrong, target) < cumulative_ordinal_nll(
        distant_wrong, target
    )


def test_cumulative_ordinal_loss_supports_edge_classes_and_gradients():
    from tiny_omni_decision.ordinal import cumulative_ordinal_nll

    logits = torch.randn(2, 10, requires_grad=True)
    targets = torch.tensor([0, 9])
    loss = cumulative_ordinal_nll(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_cumulative_ordinal_loss_rejects_bad_inputs():
    from tiny_omni_decision.ordinal import cumulative_ordinal_nll

    with pytest.raises(ValueError, match="batch"):
        cumulative_ordinal_nll(torch.randn(2, 10), torch.tensor([1]))
    with pytest.raises(ValueError, match="out of range"):
        cumulative_ordinal_nll(torch.randn(1, 10), torch.tensor([10]))
