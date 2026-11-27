from __future__ import annotations

import pytest

from scripts.audit_physical_concepts_scene_similarity import scene_signature, screen_split
from tests.test_physical_concepts_tfrecord import _example

np = pytest.importorskip("numpy")


def test_scene_signature_is_finite_and_fixed_size() -> None:
    signature = scene_signature(_example(), np)

    assert signature.shape == (524,)
    assert np.isfinite(signature).all()
    assert np.linalg.norm(signature) == pytest.approx(1.0)


def test_split_screen_reports_nearest_pairs_without_filtering() -> None:
    train_ids = ["train-a", "train-b"]
    validation_ids = ["validation-a", "validation-b"]
    train = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    validation = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    result = screen_split(train_ids, train, validation_ids, validation, np=np, top_k=2)

    assert result["nearest_counts"]["0.99"] == {
        "train_to_validation": 2,
        "validation_to_train": 2,
    }
    assert result["examples_removed_or_reassigned"] == 0
    assert len(result["top_pairs"]) == 2
    assert all(pair["similarity"] <= 1.0 for pair in result["top_pairs"])


def test_split_screen_rejects_empty_split() -> None:
    with pytest.raises(ValueError, match="non-empty splits"):
        screen_split([], np.zeros((0, 2)), ["validation"], np.ones((1, 2)), np=np)


def test_split_screen_rejects_non_normalized_signatures() -> None:
    with pytest.raises(ValueError, match="unit normalized"):
        screen_split(
            ["train"],
            np.asarray([[2.0, 0.0]]),
            ["validation"],
            np.asarray([[1.0, 0.0]]),
            np=np,
        )
