import pytest

from scripts.compare_robomnist_capacity_cv import _paired_metric_bootstrap
from tiny_omni_decision.robomnist_capacity_cv import (
    make_pair_folds,
    select_capacity_rank,
)


def _units(count=20):
    return [
        {
            "pair_unit_id": f"pair-{index:03d}",
            "condition": f"condition-{index % 2}",
            "group_a": f"group-a-{index:03d}",
            "group_b": f"group-b-{index:03d}",
        }
        for index in range(count)
    ]


def test_pair_folds_are_deterministic_complete_and_condition_balanced():
    units = _units()
    folds = make_pair_folds(units, fold_count=5, seed=17)

    assert folds == make_pair_folds(units, fold_count=5, seed=17)
    assert set(folds) == {unit["pair_unit_id"] for unit in units}
    total_counts = [sum(value == fold for value in folds.values()) for fold in range(5)]
    assert max(total_counts) - min(total_counts) <= 1
    for condition in {unit["condition"] for unit in units}:
        counts = [
            sum(
                folds[unit["pair_unit_id"]] == fold
                for unit in units
                if unit["condition"] == condition
            )
            for fold in range(5)
        ]
        assert max(counts) - min(counts) <= 1


def test_pair_folds_reject_source_group_reuse_across_pair_units():
    units = _units(5)
    units[1]["group_a"] = units[0]["group_a"]

    with pytest.raises(ValueError, match="reused across pair units"):
        make_pair_folds(units, fold_count=5, seed=17)


def test_pair_folds_require_enough_units_and_distinct_sources():
    with pytest.raises(ValueError, match="not enough pair units"):
        make_pair_folds(_units(4), fold_count=5, seed=17)
    units = _units(5)
    units[0]["group_b"] = units[0]["group_a"]
    with pytest.raises(ValueError, match="two distinct source groups"):
        make_pair_folds(units, fold_count=5, seed=17)


def test_rank_selection_uses_train_only_nll_gain_and_accuracy_guardrail():
    assert select_capacity_rank(
        {
            8: {"nll": 0.5, "accuracy": 0.82},
            32: {"nll": 0.53, "accuracy": 0.83},
        }
    ) == 8
    assert select_capacity_rank(
        {
            8: {"nll": 0.52, "accuracy": 0.83},
            32: {"nll": 0.53, "accuracy": 0.83},
        }
    ) == 32


def test_paired_group_bootstrap_is_deterministic_and_resamples_whole_pairs():
    import numpy as np

    targets = np.tile([0, 0, 1, 1], 200)
    pair_ids = np.repeat([f"pair-{index}" for index in range(200)], 4)
    reference = np.tile([[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.4, 0.6]], (200, 1))
    candidate = np.tile([[0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.4, 0.6]], (200, 1))

    first = _paired_metric_bootstrap(
        candidate, reference, targets, pair_ids, replicates=20, seed=17
    )
    second = _paired_metric_bootstrap(
        candidate, reference, targets, pair_ids, replicates=20, seed=17
    )

    assert first == second
    assert first["independent_pair_units"] == 200
    assert first["point_delta_candidate_minus_reference"]["accuracy"] == 0.0
    assert set(first["percentile_95_ci"]) == {"accuracy", "nll", "brier", "ece_15"}


def test_paired_group_bootstrap_rejects_misaligned_or_incomplete_units():
    import numpy as np

    probabilities = np.tile([[0.6, 0.4], [0.4, 0.6]], (400, 1))
    targets = np.tile([0, 1], 400)
    pair_ids = np.repeat([f"pair-{index}" for index in range(200)], 4)
    with pytest.raises(ValueError, match="shapes or values"):
        _paired_metric_bootstrap(probabilities[:-1], probabilities, targets, pair_ids)
    with pytest.raises(ValueError, match="exactly four rows"):
        _paired_metric_bootstrap(
            probabilities[:-1], probabilities[:-1], targets[:-1], pair_ids[:-1]
        )
    assert select_capacity_rank(
        {
            8: {"nll": 0.49, "accuracy": 0.80},
            32: {"nll": 0.52, "accuracy": 0.82},
        }
    ) == 32
