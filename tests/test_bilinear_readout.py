import pytest

torch = pytest.importorskip("torch")


def test_bilinear_candidate_equivariance_and_gradients():
    from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer

    model = BilinearCandidateScorer(768, 384, 128)
    assert sum(p.numel() for p in model.parameters()) == 147712
    values = torch.randn(2, 5, 1152)
    scores = model(values)
    torch.testing.assert_close(model(values.flip(1)), scores.flip(1))
    scores.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.randn(2, 1151))
