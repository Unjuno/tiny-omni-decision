import numpy as np
import pytest

from tiny_omni_decision.dinov2_features import (
    cls_patch_mean,
    readout_features,
    require_image_splits,
    spatial_cells,
)


def test_spatial_cells_row_major_and_mean():
    grid = np.arange(256, dtype=np.float32).reshape(1, 16, 16, 1)
    tokens = np.concatenate([np.zeros((1, 1, 384)), np.repeat(grid.reshape(1, 256, 1), 384, 2)], 1)
    cells = spatial_cells(tokens)
    assert cells.shape == (1, 17, 384)
    assert cells[0, 1, 0] == grid[0, :4, :4, 0].mean()
    assert cells[0, 2, 0] == grid[0, :4, 4:8, 0].mean()
    np.testing.assert_allclose(cells[:, 1:].mean(1), tokens[:, 1:].mean(1))


def test_spatial_aggregation_preserves_float32_reference_mean():
    x = np.random.default_rng(17).normal(16, 2, (3, 257, 384)).astype(np.float32)
    cells = spatial_cells(x)
    np.testing.assert_allclose(cells[:, 1:].mean(1), x[:, 1:].mean(1), rtol=0, atol=5e-6)


def test_global_control_discards_layout_with_equal_shape():
    cells = np.arange(17 * 384, dtype=np.float32).reshape(1, 17, 384)
    a = readout_features(cells, "global")
    b = readout_features(cells, "spatial")
    assert a.shape == b.shape == (1, 6528)
    np.testing.assert_allclose(a[:, 384:].reshape(1, 16, 384)[:, 0], cells[:, 1:].mean(1) / 4)
    np.testing.assert_array_equal(b[:, 384:], cells[:, 1:].reshape(1, -1) / 4)
    with pytest.raises(ValueError):
        readout_features(cells, "unknown")


def test_cls_patch_mean_separates_cls():
    x = np.ones((2, 257, 384), dtype=np.float32)
    x[:, 0] = 5
    out = cls_patch_mean(x)
    assert out.shape == (2, 768)
    np.testing.assert_array_equal(out[:, :384], 5)
    np.testing.assert_array_equal(out[:, 384:], 1)


def test_invalid_features_fail():
    for x in [np.zeros((2, 256, 384)), np.full((1, 257, 384), np.nan)]:
        with pytest.raises(ValueError):
            cls_patch_mean(x)


def test_split_identity_rejections():
    require_image_splits([{"id": "a", "sha256": "x"}], [{"id": "b", "sha256": "y"}])
    for other in [{"id": "a", "sha256": "y"}, {"id": "b", "sha256": "x"}]:
        with pytest.raises(ValueError):
            require_image_splits([{"id": "a", "sha256": "x"}], [other])
    with pytest.raises(ValueError):
        require_image_splits([{"id": "a", "sha256": "x"}] * 2, [{"id": "b", "sha256": "y"}])


def test_cellwise_candidate_scorer_parameter_count_and_candidate_shape():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.cellwise_readout import (
        CellwiseCandidateScorer,
        trainable_parameter_count,
    )

    scorer = CellwiseCandidateScorer()
    assert trainable_parameter_count(scorer) == 180736
    output = scorer(torch.zeros(2, 10, 6912))
    assert output.shape == (2, 10)


def test_cellwise_candidate_scorer_rejects_bad_width():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.cellwise_readout import CellwiseCandidateScorer

    with pytest.raises(ValueError):
        CellwiseCandidateScorer()(torch.zeros(1, 6911))
