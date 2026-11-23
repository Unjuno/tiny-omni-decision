from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.pretrained_reuse.evaluate_kuci_publisher_dev_once import (  # noqa: E402
    assert_no_training_group_leakage,
    audit_rows,
)


def _row(sample_id: int, *, context: str = "context", pair: str = "event-a/event-b") -> dict:
    return {
        "id": sample_id,
        "context": context,
        "choice_a": "choice a",
        "choice_b": "choice b",
        "choice_c": "choice c",
        "choice_d": "choice d",
        "label": "a",
        "agreement": 3,
        "core_event_pair": pair,
    }


def test_audit_reports_context_reuse_but_allows_nonoverlapping_groups() -> None:
    train = _row(1, context="shared context", pair="train/event")
    train["choice_d"] = "different choice"
    report = audit_rows(
        [_row(2, context=" shared   context ", pair="dev/event")],
        {"official_train": [train]},
    )

    overlap = report["references"]["official_train"]
    assert overlap["shared_normalized_content"] == 0
    assert overlap["shared_event_pairs"] == 0
    assert overlap["shared_normalized_contexts"] == 1
    assert_no_training_group_leakage(report)


@pytest.mark.parametrize(
    ("dev", "train", "message"),
    [
        (_row(2), _row(1), "content overlaps"),
        (_row(2, context="dev context", pair="shared/event"),
         _row(1, context="train context", pair="shared/event"), "event groups overlap"),
    ],
)
def test_audit_rejects_content_or_event_group_leakage(dev, train, message) -> None:
    report = audit_rows([dev], {"official_train": [train]})

    with pytest.raises(ValueError, match=message):
        assert_no_training_group_leakage(report)


def test_audit_rejects_duplicate_ids_and_duplicate_content() -> None:
    with pytest.raises(ValueError, match="duplicate publisher development IDs"):
        audit_rows([_row(1), _row(1, context="different")], {})

    with pytest.raises(ValueError, match="duplicate normalized question/choice content"):
        audit_rows([_row(1), _row(2)], {})
