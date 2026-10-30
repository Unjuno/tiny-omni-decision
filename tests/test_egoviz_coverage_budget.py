import pytest

from scripts.egoviz_coverage_budget_policy import budget_geometry, evaluation_steps


def test_same_update_budget_counts_partial_pass_and_exposures():
    result = budget_geometry(examples=438, batch_size=32, updates=5632)
    assert result == {
        "batches_per_pass": 14,
        "full_passes": 402,
        "partial_pass_batches": 4,
        "sample_exposures": 176204,
    }
    assert evaluation_steps(5632)[-1] == 5632
    assert {176, 352, 704, 1408, 2816, 4224, 5632}.issubset(evaluation_steps(5632))


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError):
        budget_geometry(examples=0, batch_size=32, updates=5632)


def test_saved_probabilities_must_match_logits():
    import numpy as np

    from scripts.verify_egoviz_coverage_budget import check_probabilities

    logits = np.zeros((1, 4))
    check_probabilities([{"probabilities": [0.25] * 4}], logits)
    for values in ([0.4, 0.2, 0.2, 0.2], [float("nan")] * 4):
        with pytest.raises(ValueError):
            check_probabilities([{"probabilities": values}], logits)
