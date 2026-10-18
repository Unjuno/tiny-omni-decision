import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("PIL")
pytest.importorskip("numpy")


def test_spatial_public_eval_emits_finite_normalized_probabilities():
    from scripts.evaluate_dinov2_clevr4_spatial_public_val import score_feature_batch
    from tiny_omni_decision.cellwise_readout import CellwiseCandidateScorer

    model = CellwiseCandidateScorer().eval()
    cells = torch.randn(3, 17, 384).numpy()
    options = {
        "color": torch.randn(4, 384),
        "shape": torch.randn(4, 384),
    }
    for condition in ("global", "spatial"):
        output = score_feature_batch(model, cells, options, condition, torch.device("cpu"))
        assert set(output) == {"color", "shape"}
        for probabilities in output.values():
            assert probabilities.shape == (3, 4)
            assert torch.isfinite(probabilities).all()
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(3))


def test_global_control_is_invariant_to_patch_cell_order():
    from scripts.evaluate_dinov2_clevr4_spatial_public_val import score_feature_batch
    from tiny_omni_decision.cellwise_readout import CellwiseCandidateScorer

    model = CellwiseCandidateScorer().eval()
    cells = torch.randn(2, 17, 384).numpy()
    options = {"color": torch.randn(4, 384)}
    original = score_feature_batch(model, cells, options, "global", torch.device("cpu"))
    permuted = cells.copy()
    permuted[:, 1:] = permuted[:, 1:][:, ::-1]
    reordered = score_feature_batch(model, permuted, options, "global", torch.device("cpu"))
    torch.testing.assert_close(original["color"], reordered["color"], atol=1e-6, rtol=1e-6)


def test_spatial_readout_is_permutation_invariant_but_distinct_from_global():
    from scripts.evaluate_dinov2_clevr4_spatial_public_val import score_feature_batch
    from tiny_omni_decision.cellwise_readout import CellwiseCandidateScorer

    model = CellwiseCandidateScorer().eval()
    cells = torch.randn(2, 17, 384).numpy()
    options = {"color": torch.randn(4, 384)}
    original = score_feature_batch(model, cells, options, "spatial", torch.device("cpu"))
    permuted = cells.copy()
    permuted[:, 1:] = permuted[:, 1:][:, ::-1]
    reordered = score_feature_batch(model, permuted, options, "spatial", torch.device("cpu"))
    torch.testing.assert_close(original["color"], reordered["color"], atol=1e-6, rtol=1e-6)
    global_output = score_feature_batch(model, cells, options, "global", torch.device("cpu"))
    assert not torch.allclose(original["color"], global_output["color"], atol=1e-6, rtol=1e-6)
