from __future__ import annotations

import pytest

from tiny_omni_decision.pretrained_reuse import (
    validate_clevr4_splits,
    validate_validation_partition,
)


def test_validation_partition_preserves_prior_selection_and_new_holdout_order() -> None:
    selected, holdout = validate_validation_partition(
        ["v3", "v1", "v4", "v2"], ["v2", "v3"]
    )
    assert selected == ["v3", "v2"]
    assert holdout == ["v1", "v4"]
    assert set(selected).isdisjoint(holdout)


@pytest.mark.parametrize(
    ("validation_ids", "selection_ids"),
    [(["v1", "v1"], ["v1"]), (["v1"], ["v2"]), (["v1", "v2"], ["v1", "v1"])],
)
def test_validation_partition_fails_on_duplicate_or_missing_ids(
    validation_ids: list[str], selection_ids: list[str]
) -> None:
    with pytest.raises(ValueError):
        validate_validation_partition(validation_ids, selection_ids)


def test_full_split_validation_rejects_media_hash_overlap() -> None:
    train_rows = [{"id": "a", "split": "train"}]
    validation_rows = [{"id": "b", "split": "val"}]
    train_assets = [{"id": "a", "sha256": "same"}]
    validation_assets = [{"id": "b", "sha256": "same"}]
    with pytest.raises(ValueError, match="media hashes overlap"):
        validate_clevr4_splits(train_rows, validation_rows, train_assets, validation_assets)


def test_static_image_preprocessing_repeats_eight_finite_frames() -> None:
    np = pytest.importorskip("numpy")
    tf = pytest.importorskip("tensorflow")
    from scripts.extract_frozen_videoprism_clevr4_features import _preprocess_rgb_image

    pixels = np.full((360, 512, 3), 255, dtype=np.uint8)
    frames = _preprocess_rgb_image(tf.convert_to_tensor(pixels), tf)
    assert frames.shape == (8, 288, 288, 3)
    assert np.isfinite(frames).all()
    assert frames.min() == 1.0
    assert frames.max() == 1.0
