from __future__ import annotations

import io
import json

from tiny_omni_decision.path_a_runtime_cli import run_jsonl


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], list[object]]] = []
        self.closed = False

    def dispatch_typed(self, task, payload, *, decision_specs):
        self.calls.append((task, payload, decision_specs))
        records = [
            {
                "request_id": spec.request_id,
                "selected_option_id": spec.option_ids[0],
                "options": [
                    {"id": option_id, "probability": 1.0 if index == 0 else 0.0}
                    for index, option_id in enumerate(spec.option_ids)
                ],
            }
            for spec in decision_specs
        ]
        return {"decision_json": records if task == "audio" else records[0]}

    def close(self) -> None:
        self.closed = True


def _request(task_id: str, task: str, payload: dict[str, object]) -> dict[str, object]:
    payload_options = payload.get("options")
    if isinstance(payload_options, list) and all(isinstance(item, str) for item in payload_options):
        options = [{"id": item, "text": item} for item in payload_options]
    else:
        options = [{"id": "wrong", "text": "wrong"}, {"id": "correct", "text": "correct"}]
    specs = [
        {
            "request_id": "req-1" if task != "audio" else f"req-1-{index}",
            "decision_type": "choice",
            "options": options,
        }
        for index in range(len(payload.get("queries", [])) or 1)
    ]
    return {
        "request_id": "req-1",
        "task_id": task_id,
        "payload": payload,
        "decision_specs": specs,
    }


def test_jsonl_dispatches_scoped_text_request_and_returns_typed_json() -> None:
    dispatcher = FakeDispatcher()
    request = _request(
        "typed-decisions-synth-text-v0",
        "text",
        {"state": "door is open", "question": "which option applies?", "options": ["yes", "no"]},
    )
    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()

    run_jsonl(stdin, stdout, dispatcher)

    response = json.loads(stdout.getvalue())
    assert response["status"] == "ok"
    assert response["task_id"] == "typed-decisions-synth-text-v0"
    assert response["decision"]["request_id"] == "req-1"
    assert dispatcher.calls[0][0] == "text"


def test_jsonl_audio_keeps_one_typed_record_per_query() -> None:
    dispatcher = FakeDispatcher()
    payload = {
        "audio_path": "sample.wav",
        "expected_media_sha256": "a" * 64,
        "queries": [
            {"question": "is command left?", "options": ["left", "right"]},
            {"question": "is command up?", "options": ["up", "down"]},
        ],
    }
    request = _request("path-a-audio-multiquery-runtime-v0", "audio", payload)
    request["decision_specs"] = [
        {
            "request_id": f"req-1-{index}",
            "decision_type": "choice",
            "options": [{"id": label, "text": label} for label in options],
        }
        for index, options in enumerate((("left", "right"), ("up", "down")))
    ]
    stdout = io.StringIO()

    run_jsonl(io.StringIO(json.dumps(request) + "\n"), stdout, dispatcher)

    response = json.loads(stdout.getvalue())
    assert len(response["decision"]) == 2
    assert [item["request_id"] for item in response["decision"]] == ["req-1-0", "req-1-1"]


def test_jsonl_rejects_media_without_sha256_before_dispatch() -> None:
    dispatcher = FakeDispatcher()
    request = _request(
        "path-a-image-videoprism-clevr4-runtime-v0",
        "image",
        {"image_path": "sample.png", "taxonomy": "color", "options": ["red", "blue"]},
    )
    stdout = io.StringIO()

    run_jsonl(io.StringIO(json.dumps(request) + "\n"), stdout, dispatcher)

    response = json.loads(stdout.getvalue())
    assert response["status"] == "error"
    assert "expected_media_sha256" in response["error"]["message"]
    assert dispatcher.calls == []


def test_jsonl_rejects_video_options_that_do_not_match_frozen_head_order() -> None:
    dispatcher = FakeDispatcher()
    request = _request(
        "clevrer-videoprism-binary-feature-decision-v0",
        "video",
        {
            "video_path": "sample.mp4",
            "expected_media_sha256": "b" * 64,
            "question": "did it happen?",
            "candidate_statement": "the object moved",
        },
    )
    request["decision_specs"][0]["options"] = [
        {"id": "yes", "text": "correct"},
        {"id": "no", "text": "wrong"},
    ]
    stdout = io.StringIO()

    run_jsonl(io.StringIO(json.dumps(request) + "\n"), stdout, dispatcher)

    response = json.loads(stdout.getvalue())
    assert response["status"] == "error"
    assert "('wrong', 'correct')" in response["error"]["message"]
    assert dispatcher.calls == []


def test_jsonl_reports_bad_line_and_continues_to_next_request() -> None:
    dispatcher = FakeDispatcher()
    request = _request(
        "typed-decisions-synth-text-v0",
        "text",
        {"state": "door is open", "question": "which option applies?", "options": ["yes", "no"]},
    )
    stdout = io.StringIO()

    run_jsonl(io.StringIO("not-json\n" + json.dumps(request) + "\n"), stdout, dispatcher)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [item["status"] for item in responses] == ["error", "ok"]
