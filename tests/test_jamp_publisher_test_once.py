from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.pretrained_reuse.evaluate_jamp_publisher_test_once import (  # noqa: E402
    audit_test_overlap,
    validation_selection_record,
)


def _row(num: str, group: str, premise: str, hypothesis: str) -> dict[str, str]:
    return {
        "num": num,
        "template_num": group,
        "premise": premise,
        "hypothesis": hypothesis,
        "gold_label": "entailment",
        "category": "temporal",
    }


def test_publisher_test_audit_reports_local_id_collisions_and_rejects_content_leaks() -> None:
    train = [_row("1", "train-group", "A happens", "B follows")]
    validation = [_row("2", "valid-group", "C happens", "D follows")]
    clean_test = [_row("3", "test-group", "E happens", "F follows")]

    audit = audit_test_overlap(train, validation, clean_test)
    assert audit["train_file_local_num_collisions"] == 0
    assert audit["validation_normalized_pair_overlap"] == 0
    assert audit["train_template_group_overlap"] == 0

    local_id_collision = audit_test_overlap(
        train, validation, [_row("1", "new-group", "new", "pair")]
    )
    assert local_id_collision["train_file_local_num_collisions"] == 1
    with pytest.raises(ValueError, match="shares normalized content"):
        audit_test_overlap(
            train,
            validation,
            [_row("4", "new-group", " A   happens ", "B follows")],
        )


def test_validation_only_selection_fails_closed_if_baseline_changes() -> None:
    frozen = {
        "candidate_reports": [
            {"candidate": "ruri-v3-30m", "selected_checkpoint": {"metrics": {"nll": 1.14}}},
            {
                "candidate": "line-distilbert-ja-first4",
                "selected_checkpoint": {"metrics": {"nll": 1.09}},
            },
            {
                "candidate": "tohoku-bert-base-ja-v3",
                "selected_checkpoint": {"metrics": {"nll": 1.10}},
            },
        ]
    }
    adapted = {"best_checkpoint": {"metrics": {"nll": 1.45}}}

    selection = validation_selection_record(frozen, adapted)

    assert selection["baseline"] == "line-distilbert-ja-first4"
    assert selection["publisher_test_metrics_used_for_selection"] is False
    frozen["candidate_reports"][1]["selected_checkpoint"]["metrics"]["nll"] = 1.5
    with pytest.raises(ValueError, match="selection rule changed"):
        validation_selection_record(frozen, adapted)
