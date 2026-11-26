import copy

import pytest

from scripts.publish_binding_replay_coverage import (
    check_accounting,
    check_control_arrays,
    hypothesis,
)


def test_saved_control_raw_scores_must_match_prior_identical_condition():
    import numpy as np

    old = {"known": np.array([[0.0, 1.0]]), "focus": np.array([[1.0, 2.0]])}
    check_control_arrays(old, old)
    with pytest.raises(ValueError):
        check_control_arrays(old, {"known": old["known"]})
    with pytest.raises(ValueError):
        check_control_arrays(old, dict(old, focus=np.array([[1.0, 2.01]])))


def test_accounting_binds_every_recorded_curve_point_to_actual_consumption():
    data = [
        [dict(id="b", source="binding", group="bs", task="color", language_family=1)],
        [
            dict(id="o", source="original", group="os", task="choice"),
            dict(id="c", source="curriculum", group="cs", task="count"),
        ],
    ]
    order = [[[0, 0], [1, 0], [1, 1]]]
    record = dict(
        step=1,
        consumed=3,
        unique_questions=3,
        source_counts=dict(binding=1, original=1, curriculum=1),
        task_counts={"binding:color": 1, "original:choice": 1, "curriculum:count": 1},
        language_family_counts={"1": 1},
        unique_states_by_source=dict(binding=1, original=1, curriculum=1),
    )
    check_accounting([record], data, order)
    for key in ("source_counts", "unique_states_by_source", "language_family_counts"):
        wrong = copy.deepcopy(record)
        wrong[key] = {}
        with pytest.raises(ValueError):
            check_accounting([wrong], data, order)
    wrong = dict(record, consumed=4)
    with pytest.raises(ValueError):
        check_accounting([wrong], data, order)


def test_retention_hypothesis_requires_eligibility_and_paired_nll_improvement():
    better = dict(difference=[0.01, -0.02], lower=[-0.01, -0.03], upper=[0.03, -0.01])
    assert hypothesis(True, better) == "SUPPORT_WITHIN_FIXED_BUDGET"
    assert hypothesis(False, better) == "MIXED_RETENTION_TRADEOFF"
    assert hypothesis(True, dict(better, upper=[0.03, 0.01])) == "UNCERTAIN"
    assert (
        hypothesis(False, dict(better, difference=[0, -0.0])) == "NOT_SUPPORTED_WITHIN_FIXED_BUDGET"
    )
