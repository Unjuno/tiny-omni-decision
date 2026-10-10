from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")
compare_features = importlib.import_module(
    "scripts.compare_video_feature_order_sensitivity"
).compare_features


def test_compare_features_reports_small_order_perturbations() -> None:
    forward = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    reverse = np.asarray([[0.99, 0.01], [0.01, 0.99]])

    report = compare_features(forward, reverse, media_hashes=["a", "b"])

    assert report["paired_assets"] == 2
    assert report["feature_dimension"] == 2
    assert report["bitwise_equal"] is False
    assert report["cosine_similarity"]["mean"] == pytest.approx(0.999949, abs=1e-6)
    assert report["relative_l2_difference"]["mean"] > 0


@pytest.mark.parametrize(
    ("forward", "reverse", "hashes"),
    [
        (np.ones((2, 3)), np.ones((2, 4)), ["a", "b"]),
        (np.ones((2, 3)), np.ones((2, 3)), ["a"]),
        (np.ones((2, 3)), np.ones((2, 3)), ["a", "a"]),
        (np.asarray([[0.0, 0.0]]), np.asarray([[1.0, 1.0]]), ["a"]),
        (np.asarray([[np.nan, 1.0]]), np.ones((1, 2)), ["a"]),
    ],
)
def test_compare_features_rejects_invalid_pairs(
    forward: np.ndarray, reverse: np.ndarray, hashes: list[str]
) -> None:
    with pytest.raises(ValueError):
        compare_features(forward, reverse, media_hashes=hashes)
