import pytest

torch = pytest.importorskip("torch")


def test_fold_preserves_function_and_source():
    from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
    from tiny_omni_decision.global_readout import fold_global_readout

    model = BilinearCandidateScorer(13056, 384, 128).double()
    old = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    folded = fold_global_readout(model, condition="global")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert sum(p.numel() for p in folded.parameters()) == 246016
    summary, mean, options = torch.randn(2, 3, 768), torch.randn(2, 3, 768), torch.randn(2, 3, 384)
    wide = torch.cat((summary, mean.repeat(1, 1, 16) / 4, options), -1).double()
    narrow = torch.cat((summary, mean, options), -1).double()
    torch.testing.assert_close(model(wide), folded(narrow), atol=1e-12, rtol=1e-12)
    assert all(torch.equal(v, old[k]) for k, v in model.state_dict().items())


def test_fold_rejects_spatial_or_wrong_shape():
    from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
    from tiny_omni_decision.global_readout import fold_global_readout

    model = BilinearCandidateScorer(13056, 384, 128)
    with pytest.raises(ValueError):
        fold_global_readout(model, condition="spatial")
    with pytest.raises(ValueError):
        fold_global_readout(BilinearCandidateScorer(768, 384, 128), condition="global")
