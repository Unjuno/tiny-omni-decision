import pytest

torch = pytest.importorskip("torch")


def test_cellwise_symmetry_options_and_gradients():
    from tiny_omni_decision.cellwise_readout import CellwiseCandidateScorer

    m = CellwiseCandidateScorer()
    assert sum(p.numel() for p in m.parameters()) == 180736
    x = torch.randn(2, 3, 6912)
    y = m(x)
    torch.testing.assert_close(m(x.flip(1)), y.flip(1))
    changed = x.clone()
    changed[..., 384:6528] = x[..., 384:6528].reshape(2, 3, 16, 384).flip(-2).reshape(2, 3, 6144)
    torch.testing.assert_close(m(changed), y, atol=1e-5, rtol=1e-5)
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with pytest.raises(ValueError):
        m(torch.randn(2, 6911))
