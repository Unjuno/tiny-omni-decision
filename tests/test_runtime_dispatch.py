from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tiny_omni_decision.runtime_dispatch import (
    JsonLineWorker,
    PathARuntimeDispatcher,
    TypedDecisionSpec,
)


def test_json_line_worker_keeps_process_resident_for_multiple_requests(tmp_path: Path) -> None:
    worker_source = tmp_path / "worker.py"
    worker_source.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True, 'profile': 'test'}), flush=True)\n"
        "for line in sys.stdin:\n"
        " request = json.loads(line)\n"
        " if request.get('command') == 'shutdown':\n"
        "  print(json.dumps({'stopped': True}), flush=True)\n"
        "  break\n"
        " print(json.dumps({'echo': request['value']}), flush=True)\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "worker.log"

    worker = JsonLineWorker(
        [sys.executable, str(worker_source)],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=log_path,
    )
    try:
        assert worker.ready["profile"] == "test"
        assert worker.request({"value": 17})["echo"] == 17
        assert worker.request({"value": "same-process"})["echo"] == "same-process"
    finally:
        worker.close()

    assert worker._process.poll() == 0


def test_dispatch_routes_each_modality_to_its_runtime_profile() -> None:
    class RecordingWorker:
        def __init__(self, profile: str) -> None:
            self.profile = profile
            self.requests: list[dict[str, object]] = []

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            self.requests.append(payload)
            task = payload["task"]
            if task == "encode_video":
                return {"video_feature": [0.25] * 768, "encoder_metrics": {"seconds": 1}}
            if task == "image":
                return {"task": "image", "profile": self.profile}
            return {"task": task, "profile": self.profile}

    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = RecordingWorker("torch")
    dispatcher._videoprism_worker = RecordingWorker("videoprism")

    image = dispatcher.dispatch("image", {"image_path": "image.png"})
    text = dispatcher.dispatch("text", {"state": "state", "question": "q", "options": []})
    audio = dispatcher.dispatch("audio", {"audio_path": "audio.wav", "queries": []})
    video = dispatcher.dispatch(
        "video",
        {"video_path": "video.mp4", "question": "q", "candidate_statement": "candidate"},
    )

    assert image["profile"] == "videoprism"
    assert text["profile"] == "torch"
    assert audio["profile"] == "torch"
    assert video["video_encoder"] == {"seconds": 1}
    assert dispatcher._videoprism_worker.requests[-1]["task"] == "encode_video"
    video_score = dispatcher._torch_worker.requests[-1]
    assert video_score["task"] == "video_score"
    assert video_score["payload"]["video_feature"] == [0.25] * 768


def test_dispatch_rejects_unknown_task_before_starting_workers() -> None:
    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = None
    dispatcher._videoprism_worker = None

    with pytest.raises(ValueError, match="unsupported Path A task"):
        dispatcher.dispatch("sensor", {})

    assert dispatcher._torch_worker is None
    assert dispatcher._videoprism_worker is None


def test_dispatch_typed_wraps_existing_text_scores_without_reordering_options() -> None:
    class RecordingWorker:
        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            self.requests.append(payload)
            return {
                "logits": [0.0, 1.0],
                "probabilities": [0.2, 0.8],
                "prediction_index": 1,
                "confidence": 0.8,
            }

    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = RecordingWorker()
    dispatcher._videoprism_worker = None
    spec = TypedDecisionSpec(
        request_id="choice-17",
        decision_type="choice",
        option_ids=("candidate-b", "candidate-a"),
        option_texts=("B", "A"),
    )

    result = dispatcher.dispatch_typed(
        "text",
        {"state": "state", "question": "q", "options": ["B", "A"]},
        decision_specs=[spec],
    )

    assert result["decision_json"]["selected_option_id"] == "candidate-a"
    assert [item["id"] for item in result["decision_json"]["options"]] == [
        "candidate-b",
        "candidate-a",
    ]
    assert result["probabilities"] == [0.2, 0.8]


def test_dispatch_typed_validates_audio_spec_count_before_starting_workers() -> None:
    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = None
    dispatcher._videoprism_worker = None

    with pytest.raises(ValueError, match="one decision spec per audio query"):
        dispatcher.dispatch_typed(
            "audio",
            {"audio_path": "audio.wav", "queries": [{"options": ["a", "b"]}]},
            decision_specs=[],
        )

    assert dispatcher._torch_worker is None
    assert dispatcher._videoprism_worker is None


def test_dispatch_typed_wraps_each_audio_query_with_its_own_contract() -> None:
    class RecordingWorker:
        def request(self, payload: dict[str, object]) -> dict[str, object]:
            return {
                "decisions": [
                    {
                        "probabilities": [0.9, 0.1],
                        "prediction_index": 0,
                        "confidence": 0.9,
                    },
                    {
                        "probabilities": [0.2, 0.8],
                        "prediction_index": 1,
                        "confidence": 0.8,
                    },
                ],
                "encoder_metrics": {"cache_hits": 1},
            }

    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = RecordingWorker()
    dispatcher._videoprism_worker = None
    specs = [
        TypedDecisionSpec("audio-1", "yes_no", ("no", "yes"), ("No", "Yes")),
        TypedDecisionSpec("audio-2", "choice", ("left", "right"), ("Left", "Right")),
    ]

    result = dispatcher.dispatch_typed(
        "audio",
        {
            "audio_path": "audio.wav",
            "queries": [
                {"question": "first?", "options": ["No", "Yes"]},
                {"question": "second?", "options": ["Left", "Right"]},
            ],
        },
        decision_specs=specs,
    )

    assert [item["decision_json"]["request_id"] for item in result["decisions"]] == [
        "audio-1",
        "audio-2",
    ]
    assert [item["decision_json"]["selected_option_id"] for item in result["decisions"]] == [
        "no",
        "right",
    ]
    assert result["encoder_metrics"] == {"cache_hits": 1}
