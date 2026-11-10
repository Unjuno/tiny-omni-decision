from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from scripts.diagnose_videomae_clevr4_count_features import (
    _read_npy_header,
    paired_accuracy_bootstrap,
    ridge_scores,
    stratified_hash_folds,
)


def test_hash_folds_are_deterministic_and_stratified() -> None:
    ids = [f"image-{index:03d}" for index in range(40)]
    targets = np.repeat(np.arange(4), 10)

    first = stratified_hash_folds(ids, targets, n_folds=5)
    second = stratified_hash_folds(ids, targets, n_folds=5)

    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8
    for target in range(4):
        assert np.bincount(first[targets == target], minlength=5).tolist() == [2] * 5


def test_npy_header_reader_supports_numpy_v1_members() -> None:
    stream = BytesIO()
    np.save(stream, np.zeros((2, 3), dtype=np.float32))
    stream.seek(0)

    assert _read_npy_header(stream) == ((2, 3), False, np.dtype("float32"))


def test_ridge_diagnostic_recovers_linearly_separable_classes() -> None:
    rng = np.random.default_rng(17)
    prototypes = np.eye(4, dtype=np.float32) * 4
    fit_targets = np.repeat(np.arange(4), 20)
    query_targets = np.repeat(np.arange(4), 5)
    fit = prototypes[fit_targets] + rng.normal(0, 0.15, size=(len(fit_targets), 4)).astype(
        np.float32
    )
    query = prototypes[query_targets] + rng.normal(0, 0.15, size=(len(query_targets), 4)).astype(
        np.float32
    )

    scores = ridge_scores(fit, fit_targets, query, alpha=0.01, n_classes=4)

    assert scores.shape == (len(query), 4)
    assert float(np.mean(scores.argmax(axis=1) == query_targets)) > 0.98


def test_ridge_diagnostic_rejects_nonfinite_or_misaligned_features() -> None:
    with pytest.raises(ValueError, match="shapes"):
        ridge_scores(np.zeros((3, 2)), np.zeros(2, dtype=np.int64), np.zeros((1, 2)))
    bad = np.asarray([[0.0, np.nan], [1.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        ridge_scores(bad, np.asarray([0, 1]), np.zeros((1, 2)), n_classes=2)


def test_paired_bootstrap_is_deterministic_and_aligned() -> None:
    targets = np.asarray([0, 1, 1, 0, 1])
    baseline = np.asarray([0, 0, 1, 0, 0])
    candidate = np.asarray([0, 1, 1, 1, 0])

    result = paired_accuracy_bootstrap(targets, baseline, candidate, iterations=250)

    assert result == paired_accuracy_bootstrap(targets, baseline, candidate, iterations=250)
    assert result["point_difference_second_minus_first"] == pytest.approx(0.0)
    assert result["iterations"] == 250
