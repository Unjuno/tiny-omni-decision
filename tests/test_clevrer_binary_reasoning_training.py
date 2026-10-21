from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.train_frozen_clevrer_binary_reasoning import (  # noqa: E402
    QUESTION_TYPES,
    _metrics_by_scene_bootstrap,
    _sample_weights,
    _validate_fetch_report,
    _validation_was_previously_used,
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


def test_validation_history_status_is_read_from_the_experiment_config():
    never_used = {"source": {"validation_previously_used": False}}
    previously_used = {"source": {"validation_previously_used": True}}
    assert _validation_was_previously_used(never_used) is False
    assert _validation_was_previously_used(previously_used) is True


def test_new_media_fetch_report_schema_passes_scene_and_asset_gates():
    _validate_fetch_report(
        {
            "source_revision": "pinned-revision",
            "cross_partition_media_hash_overlap": {"train_development": []},
        },
        source_revision="pinned-revision",
        train_scenes={1},
        validation_scenes={2},
        train_media={"train-hash"},
        validation_media={"validation-hash"},
    )


def test_fetch_report_gate_rejects_media_overlap_from_either_schema():
    with pytest.raises(ValueError, match="scene/media overlap"):
        _validate_fetch_report(
            {
                "revision": "pinned-revision",
                "scene_overlap": 0,
                "media_hash_overlap": 1,
            },
            source_revision="pinned-revision",
            train_scenes={1},
            validation_scenes={2},
            train_media={"train-hash"},
            validation_media={"validation-hash"},
        )
