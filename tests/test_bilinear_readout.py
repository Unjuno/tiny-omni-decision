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


def test_video_head_capacity_and_zero_video_independence():
    from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer

    model = BilinearCandidateScorer(768, 1536, 128)
    assert sum(p.numel() for p in model.parameters()) == 295168
    values = torch.randn(3, 2, 2304)
    values[..., :768] = 0
    output = model(values)
    torch.testing.assert_close(model(values.flip(1)), output.flip(1))


def test_spatial_bilinear_capacity_and_candidate_equivariance():
    from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer

    model = BilinearCandidateScorer(6528, 384, 128)
    assert sum(p.numel() for p in model.parameters()) == 884992
    values = torch.randn(2, 10, 6912)
    torch.testing.assert_close(model(values.flip(1)), model(values).flip(1))
