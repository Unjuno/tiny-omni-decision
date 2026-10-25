import numpy as np
import pytest

from scripts.evaluate_openimages_frozen_probe_controls import (
    _paired_summary,
    fixed_derangement,
)


def test_fixed_derangement_is_reproducible_and_has_no_self_donors():
    first = fixed_derangement(168)
    second = fixed_derangement(168)

    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(168))
    assert not np.any(first == np.arange(168))


@pytest.mark.parametrize("count", [0, 1])
def test_fixed_derangement_requires_multiple_rows(count):
    with pytest.raises(ValueError, match="at least two"):
        fixed_derangement(count)


def test_paired_summary_counts_prediction_and_correctness_changes():
    reference = np.array([[0.8, 0.2], [0.3, 0.7], [0.7, 0.3]])
    comparison = np.array([[0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
    targets = np.array([0, 1, 0])

    result = _paired_summary(reference, comparison, targets)

    assert result["count"] == 3
    assert result["prediction_agreement_count"] == 2
    assert result["reference_correct_comparison_wrong"] == 1
    assert result["comparison_correct_reference_wrong"] == 0
    assert result["reference_minus_comparison_accuracy"] == pytest.approx(1 / 3)
