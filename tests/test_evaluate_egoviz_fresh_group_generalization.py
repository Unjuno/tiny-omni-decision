from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_egoviz_fresh_group_generalization import (
    grouped_metrics,
    paired_group_bootstrap,
    validate_matched_training_provenance,
)


def test_grouped_metrics_reports_per_category_counts() -> None:
    logits = np.asarray([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0]], dtype=float)
    targets = np.asarray([0, 2, 2])
    categories = np.asarray(["a", "a", "b"])

    result = grouped_metrics(logits, targets, categories)

    assert result["overall"]["count"] == 3
    assert result["by_category"]["a"]["count"] == 2
    assert result["by_category"]["a"]["accuracy"] == 0.5
    assert result["by_category"]["b"]["count"] == 1
    assert result["by_category"]["b"]["accuracy"] == 1.0


def test_paired_group_bootstrap_is_reproducible_and_grouped() -> None:
    candidate = np.asarray([[3, 0], [0, 3], [3, 0], [0, 3]], dtype=float)
    reference = np.asarray([[0, 3], [0, 3], [3, 0], [3, 0]], dtype=float)
    targets = np.asarray([0, 1, 0, 1])
    groups = np.asarray(["s1", "s1", "s2", "s2"])

    first = paired_group_bootstrap(candidate, reference, targets, groups, draws=100, seed=17)
    second = paired_group_bootstrap(candidate, reference, targets, groups, draws=100, seed=17)

    assert first == second
    assert first["group_count"] == 2
    assert first["candidate_minus_reference"]["accuracy"]["estimate"] == 0.5


def test_matched_training_provenance_requires_identical_inputs_and_order() -> None:
    linear = {
        "seed": 17,
        "input_sha256": {"train": "abc", "development": "def"},
        "sample_order_sha256": "order",
    }
    gru = {
        "seed": 17,
        "input_sha256": {"train": "abc", "development": "def"},
        "training": {"sample_order_sha256": "order"},
    }

    validate_matched_training_provenance(linear, gru)

    gru["input_sha256"]["train"] = "different"
    with pytest.raises(ValueError, match="input hashes differ"):
        validate_matched_training_provenance(linear, gru)


def test_matched_training_provenance_rejects_different_seed_or_order() -> None:
    linear = {"seed": 17, "input_sha256": {"train": "abc"}, "sample_order_sha256": "x"}
    gru = {
        "seed": 18,
        "input_sha256": {"train": "abc"},
        "training": {"sample_order_sha256": "y"},
    }

    with pytest.raises(ValueError, match="seeds differ"):
        validate_matched_training_provenance(linear, gru)

    gru["seed"] = 17
    with pytest.raises(ValueError, match="sample-order hashes differ"):
        validate_matched_training_provenance(linear, gru)
