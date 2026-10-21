import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision import mask_pair_readout as module  # noqa: E402


def test_binary_logits_have_gradients_and_exact_small_parameter_inventory():
    model = module.MaskPairReadout(9, hidden=2)
    logits = model(torch.ones(3, 9))
    assert logits.shape == (3, 2) and torch.equal(logits[:, 0], torch.zeros(3))
    assert sum(p.numel() for p in model.parameters()) == 23
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_feature_shape_and_nonfinite_values_are_rejected():
    model = module.MaskPairReadout(9)
    for x in (torch.ones(3, 8), torch.full((3, 9), float("nan")), torch.ones(0, 9)):
        with pytest.raises(ValueError):
            model(x)
