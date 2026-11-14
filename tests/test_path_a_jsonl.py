from __future__ import annotations

import io
import json

import pytest

from tiny_omni_decision.path_a_jsonl import check_parameter_budget, parse_request, run_jsonl
from tiny_omni_decision.runtime_dispatch import TypedDecisionSpec


def test_parse_text_request_builds_typed_decision_spec() -> None:
    task, payload, specs = parse_request(
        {
            "task": "text",
            "payload": {"state": "S", "question": "Q", "options": ["A", "B"]},
            "decision": {
                "request_id": "r-1",
                "decision_type": "choice",
                "option_ids": ["a", "b"],
                "option_texts": ["A", "B"],
            },
        }
    )

    assert task == "text"
    assert payload["options"] == ["A", "B"]
    assert specs == [TypedDecisionSpec("r-1", "choice", ("a", "b"), ("A", "B"))]


def test_parse_audio_request_accepts_one_contract_per_query() -> None:
    task, payload, specs = parse_request(
        {
            "task": "audio",
            "payload": {
                "audio_path": "sample.wav",
                "queries": [
                    {"question": "Q1", "options": ["No", "Yes"]},
                    {"question": "Q2", "options": ["Left", "Right"]},
                ],
            },
            "decisions": [
                {
                    "request_id": "r-1",
                    "decision_type": "yes_no",
                    "option_ids": ["no", "yes"],
                    "option_texts": ["No", "Yes"],
                },
                {
                    "request_id": "r-2",
                    "decision_type": "choice",
                    "option_ids": ["left", "right"],
                    "option_texts": ["Left", "Right"],
                },
            ],
        }
    )

    assert task == "audio"
    assert len(payload["queries"]) == 2
    assert [spec.request_id for spec in specs] == ["r-1", "r-2"]


@pytest.mark.parametrize(
    "input_value",
    [
        {},
        {"task": "sensor", "payload": {}, "decision": {}},
        {"task": "text", "payload": [], "decision": {}},
        {
            "task": "text",
            "payload": {"options": ["A", "B"]},
            "decision": {"request_id": "r", "option_ids": ["a", "b"]},
        },
    ],
)
def test_parse_request_rejects_malformed_or_unsupported_input(input_value: dict) -> None:
    with pytest.raises(ValueError):
        parse_request(input_value)


def test_parameter_budget_accepts_in_budget_manifest_and_rejects_over_budget() -> None:
    assert check_parameter_budget({"unique_runtime_parameters": 170_499_716}, 200_000_000) == (
        170_499_716
    )
    with pytest.raises(ValueError, match="exceeding"):
        check_parameter_budget({"unique_runtime_parameters": 200_000_001}, 200_000_000)


def test_run_jsonl_emits_only_typed_decision_records() -> None:
    class FakeDispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict, list[TypedDecisionSpec]]] = []

        def dispatch_typed(self, task, payload, *, decision_specs):
            self.calls.append((task, payload, decision_specs))
            return {"decision_json": {"request_id": decision_specs[0].request_id}}

    dispatcher = FakeDispatcher()
    source = io.StringIO(
        json.dumps(
            {
                "task": "image",
                "payload": {"image_path": "image.png", "taxonomy": "shape", "options": ["A", "B"]},
                "decision": {
                    "request_id": "image-1",
                    "decision_type": "choice",
                    "option_ids": ["a", "b"],
                    "option_texts": ["A", "B"],
                },
            }
        )
        + "\n"
    )
    sink = io.StringIO()

    run_jsonl(source, sink, dispatcher)

    assert json.loads(sink.getvalue()) == {"request_id": "image-1"}
    assert [call[0] for call in dispatcher.calls] == ["image"]


def test_run_jsonl_emits_all_audio_decisions_as_one_json_array() -> None:
    class FakeDispatcher:
        def dispatch_typed(self, task, payload, *, decision_specs):
            assert task == "audio"
            return {"decision_json": [{"request_id": item.request_id} for item in decision_specs]}

    request = {
        "task": "audio",
        "payload": {
            "audio_path": "sample.wav",
            "queries": [
                {"question": "first", "options": ["No", "Yes"]},
                {"question": "second", "options": ["Left", "Right"]},
            ],
        },
        "decisions": [
            {
                "request_id": "a",
                "decision_type": "yes_no",
                "option_ids": ["n", "y"],
                "option_texts": ["No", "Yes"],
            },
            {
                "request_id": "b",
                "decision_type": "choice",
                "option_ids": ["l", "r"],
                "option_texts": ["Left", "Right"],
            },
        ],
    }
    sink = io.StringIO()

    run_jsonl(io.StringIO(json.dumps(request) + "\n"), sink, FakeDispatcher())

    assert json.loads(sink.getvalue()) == [{"request_id": "a"}, {"request_id": "b"}]


def test_run_jsonl_rejects_invalid_json_with_line_number() -> None:
    with pytest.raises(ValueError, match="line 2"):
        run_jsonl(io.StringIO("\nnot-json\n"), io.StringIO(), object())
