from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.freeze_videomae_clevr4_spatial_inputs import (  # noqa: E402
    combine_train_selector_rows,
)
from tiny_omni_decision.videomae_spatial import (  # noqa: E402
    FEATURE_WIDTH,
    GRID_SIZE,
    SpatialGridClevr4Scorer,
    pool_spatiotemporal_tokens,
)


def test_spatial_pool_preserves_grid_cell_order_and_averages_time() -> None:
    first = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4, 1)
    second = first + 16
    tokens = torch.cat([first.reshape(1, 16, 1), second.reshape(1, 16, 1)], dim=1)

    pooled = pool_spatiotemporal_tokens(
        tokens, temporal_tokens=2, patch_height=4, patch_width=4, grid_size=2
    )

    assert pooled.tolist() == [[10.5, 12.5, 18.5, 20.5]]


def test_spatial_pool_rejects_mismatched_token_grid() -> None:
    with pytest.raises(ValueError, match="expected 16 tokens"):
        pool_spatiotemporal_tokens(
            torch.zeros((1, 15, 3)), temporal_tokens=1, patch_height=4, patch_width=4
        )


def test_spatial_grid_scorer_has_bounded_trainable_capacity() -> None:
    model = SpatialGridClevr4Scorer()
    count = sum(parameter.numel() for parameter in model.parameters())

    assert FEATURE_WIDTH == GRID_SIZE * GRID_SIZE * 384
    assert count == 789_962
    with torch.inference_mode():
        logits = model(torch.zeros((2, FEATURE_WIDTH)), torch.tensor([0, 2]))
    assert tuple(logits.shape) == (2, 10)


def test_spatial_grid_scorer_rejects_wrong_feature_width() -> None:
    model = SpatialGridClevr4Scorer()
    with pytest.raises(ValueError, match="6144"):
        model(torch.zeros((1, 384)), torch.tensor([0]))


def test_spatial_manifest_keeps_train_and_selector_order_without_reporting_rows() -> None:
    train = [{"id": "train-1", "media_sha256": "a"}, {"id": "train-2", "media_sha256": "b"}]
    selector = [{"id": "selector-1", "media_sha256": "c"}]

    combined = combine_train_selector_rows(train, selector)

    assert [row["id"] for row in combined] == ["train-1", "train-2", "selector-1"]


def test_spatial_manifest_rejects_cross_split_media_overlap() -> None:
    train = [{"id": "train-1", "media_sha256": "same-media"}]
    selector = [{"id": "selector-1", "media_sha256": "same-media"}]

    with pytest.raises(ValueError, match="media identities overlap"):
        combine_train_selector_rows(train, selector)
