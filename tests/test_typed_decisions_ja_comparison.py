from __future__ import annotations

import pytest

from scripts.analyze_typed_decisions_ja_comparison import compare


def _predictions(correct: bool) -> list[dict]:
    rows = []
    for workflow in ("alpha", "beta"):
        for index in range(3):
            target = "yes" if index % 2 == 0 else "no"
            prediction = target if correct else ("no" if target == "yes" else "yes")
            rows.append(
                {
                    "sample_id": f"{workflow}-{index}",
                    "workflow": workflow,
                    "question_key": "decision",
                    "question_type": "choice",
                    "target": target,
                    "gold_probabilities": {
                        "yes": 1.0 if target == "yes" else 0.0,
                        "no": 1.0 if target == "no" else 0.0,
                    },
                    "probabilities": {
                        "yes": 0.9 if prediction == "yes" else 0.1,
                        "no": 0.9 if prediction == "no" else 0.1,
                    },
                }
            )
    return rows


def test_comparison_pairs_rows_and_bootstraps_whole_states() -> None:
    result = compare(
        _predictions(correct=False),
        _predictions(correct=True),
        repetitions=1000,
        seed=17,
    )

    assert result["paired_decisions"] == 6
    assert result["paired_state_groups"] == 6
    assert result["state_groups_by_workflow"] == {"alpha": 3, "beta": 3}
    assert result["delta_second_minus_first"]["accuracy"] == pytest.approx(1.0)
    assert result["bootstrap"]["intervals"]["accuracy"]["percentile_95_ci"] == [1.0, 1.0]


def test_comparison_refuses_unpaired_validation_sets() -> None:
    second = _predictions(correct=True)
    second.pop()

    with pytest.raises(ValueError, match="identical validation decisions"):
        compare(_predictions(correct=False), second, repetitions=1000, seed=17)
