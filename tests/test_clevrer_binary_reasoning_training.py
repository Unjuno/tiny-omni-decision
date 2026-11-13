from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.train_frozen_clevrer_binary_reasoning import (  # noqa: E402
    QUESTION_TYPES,
    _metrics_by_scene_bootstrap,
    _sample_weights,
)


def _row(scene: int, kind: str) -> dict[str, object]:
    return {"scene_index": scene, "question_type": kind}


def test_sample_weights_balance_question_types_then_scenes():
    rows = [
        _row(1, "explanatory"),
        _row(1, "explanatory"),
        _row(2, "explanatory"),
        _row(1, "predictive"),
        _row(2, "predictive"),
        _row(1, "counterfactual"),
        _row(1, "counterfactual"),
        _row(2, "counterfactual"),
    ]

    weights = _sample_weights(rows)

    assert float(weights.sum()) == pytest.approx(1.0)
    for kind in QUESTION_TYPES:
        kind_weight = sum(
            float(weights[index]) for index, row in enumerate(rows) if row["question_type"] == kind
        )
        assert kind_weight == pytest.approx(1 / 3)
    assert float(weights[0]) == pytest.approx(1 / 12)
    assert float(weights[2]) == pytest.approx(1 / 6)


def test_scene_bootstrap_is_reproducible_and_reports_independent_groups():
    rows = [
        {"scene_index": scene, "question_type": kind}
        for scene in range(3)
        for kind in QUESTION_TYPES
    ]
    targets = [0, 1, 0, 1, 1, 0, 0, 0, 1]
    logits = [torch.tensor([0.6, -0.2]), torch.tensor([-0.1, 0.9])] * 4 + [torch.tensor([0.2, 0.4])]

    first = _metrics_by_scene_bootstrap(logits, targets, rows, replicates=100, seed=17)
    second = _metrics_by_scene_bootstrap(logits, targets, rows, replicates=100, seed=17)

    assert first == second
    assert first["independent_scenes"] == 3
    assert first["replicates"] == 100
    assert set(first["metrics"]) == {
        "overall_example_weighted",
        "macro_question_type",
        *QUESTION_TYPES,
    }
    for metrics in first["metrics"].values():
        for interval in metrics.values():
            assert len(interval) == 2
            assert interval[0] <= interval[1]
