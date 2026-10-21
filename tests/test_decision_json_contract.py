from __future__ import annotations

import json

import pytest

from tiny_omni_decision.decision_json import (
    decision_json_abstention_record,
    decision_json_record,
)


def test_decision_json_record_preserves_caller_ids_text_order_and_probabilities() -> None:
    record = decision_json_record(
        request_id="request-17",
        decision_type="choice",
        option_ids=["slow", "continue", "stop"],
        option_texts=["減速", "続行", "停止"],
        probabilities=[0.7, 0.2, 0.1],
    )

    assert record == {
        "schema_version": 1,
        "request_id": "request-17",
        "decision_type": "choice",
        "selected_option_id": "slow",
        "confidence": 0.7,
        "options": [
            {"id": "slow", "text": "減速", "probability": 0.7},
            {"id": "continue", "text": "続行", "probability": 0.2},
            {"id": "stop", "text": "停止", "probability": 0.1},
        ],
    }
    assert json.loads(json.dumps(record, ensure_ascii=False)) == record


def test_decision_json_record_supports_binary_and_ordinal_decisions() -> None:
    binary = decision_json_record(
        request_id="binary",
        decision_type="yes_no",
        option_ids=["no", "yes"],
        option_texts=["No", "Yes"],
        probabilities=[0.1, 0.9],
    )
    ordinal = decision_json_record(
        request_id="score",
        decision_type="ordinal_score",
        option_ids=["low", "mid", "high"],
        option_texts=["低", "中", "高"],
        probabilities=[0.2, 0.3, 0.5],
        ordinal_values=[0, 1, 2],
    )

    assert binary["selected_option_id"] == "yes"
    assert ordinal["expected_score"] == pytest.approx(1.3)


def test_decision_json_record_accepts_an_explicit_maximum_probability_tie() -> None:
    record = decision_json_record(
        request_id="tie",
        decision_type="choice",
        option_ids=["first", "second"],
        option_texts=["First", "Second"],
        probabilities=[0.5, 0.5],
        predicted_index=1,
    )

    assert record["selected_option_id"] == "second"


def test_abstention_json_has_no_fake_probability_distribution() -> None:
    record = decision_json_abstention_record(
        request_id="missing-sensor",
        decision_type="choice",
        option_ids=["walk", "sit"],
        option_texts=["walking", "sitting"],
        reason="incomplete_sensor_window",
    )

    assert record["status"] == "abstain"
    assert record["selected_option_id"] is None
    assert record["confidence"] is None
    assert [option["probability"] for option in record["options"]] == [None, None]
    assert json.loads(json.dumps(record, allow_nan=False)) == record


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": " "},
        {"decision_type": "unknown"},
        {"option_ids": ["same", "same"]},
        {"option_texts": ["only one"]},
        {"decision_type": "yes_no", "option_ids": ["a", "b", "c"]},
        {"probabilities": [0.2, 0.2]},
        {"probabilities": [float("nan"), 1.0]},
        {"probabilities": [-0.1, 1.1]},
        {"predicted_index": 1},
        {"confidence": 0.2},
        {
            "decision_type": "ordinal_score",
            "option_ids": ["a", "b", "c"],
            "option_texts": ["A", "B", "C"],
            "probabilities": [0.2, 0.3, 0.5],
            "ordinal_values": [0, 2, 1],
        },
        {"decision_type": "ordinal_score", "ordinal_values": [0]},
    ],
)
def test_decision_json_record_rejects_invalid_contract(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "request_id": "request",
        "decision_type": "choice",
        "option_ids": ["a", "b"],
        "option_texts": ["A", "B"],
        "probabilities": [0.8, 0.2],
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        decision_json_record(**values)  # type: ignore[arg-type]
