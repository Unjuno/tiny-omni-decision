from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import tiny_omni_decision.runtime_dispatch as runtime_dispatch_module
from tiny_omni_decision.runtime_dispatch import (
    JsonLineWorker,
    PathARuntimeDispatcher,
    TypedDecisionSpec,
)
from tiny_omni_decision.speech_commands_m1_contract import SPEECH_COMMANDS_M1_LABELS


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


def test_dispatch_typed_keyword_m1_preserves_caller_option_order_and_ids() -> None:
    class RecordingWorker:
        startup_wall_seconds = 0.0

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            assert payload["task"] == "audio_keyword"
            probabilities = [0.01] * len(SPEECH_COMMANDS_M1_LABELS)
            probabilities[SPEECH_COMMANDS_M1_LABELS.index("go")] = 0.91
            return {
                "probabilities": probabilities,
                "prediction_index": 9,
                "confidence": 0.91,
                "keyword_metrics": {"encoder_calls": 1},
            }

    labels = list(reversed(SPEECH_COMMANDS_M1_LABELS))
    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = RecordingWorker()
    dispatcher._videoprism_worker = None
    dispatcher.speech_commands_m1_readout = Path("m1-readout.pt")

    result = dispatcher.dispatch_typed(
        "audio_keyword",
        {"audio_path": "train.wav", "expected_media_sha256": "a" * 64},
        decision_specs=[
            TypedDecisionSpec(
                "speech-command-1",
                "choice",
                tuple(f"id-{label}" for label in labels),
                tuple(labels),
            )
        ],
    )

    assert result["decision_json"]["selected_option_id"] == "id-go"
    assert [item["text"] for item in result["decision_json"]["options"]] == labels
    assert result["keyword_metrics"] == {"encoder_calls": 1}


def test_dispatch_typed_keyword_m1_rejects_subsets_before_worker_start() -> None:
    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = None
    dispatcher._videoprism_worker = None
    dispatcher.speech_commands_m1_readout = Path("m1-readout.pt")

    with pytest.raises(ValueError, match="complete ten-label"):
        dispatcher.dispatch_typed(
            "audio_keyword",
            {"audio_path": "train.wav", "expected_media_sha256": "a" * 64},
            decision_specs=[TypedDecisionSpec("choice", "choice", ("yes", "no"), ("yes", "no"))],
        )

    assert dispatcher._torch_worker is None
    assert dispatcher._videoprism_worker is None


def test_dispatch_typed_rejects_reordered_video_options_before_starting_workers() -> None:
    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = None
    dispatcher._videoprism_worker = None

    with pytest.raises(ValueError, match="video decision option texts"):
        dispatcher.dispatch_typed(
            "video",
            {"video_path": "video.mp4", "question": "q", "candidate_statement": "s"},
            decision_specs=[
                TypedDecisionSpec("video-1", "choice", ("yes", "no"), ("correct", "wrong"))
            ],
        )

    assert dispatcher._torch_worker is None
    assert dispatcher._videoprism_worker is None


def test_dispatch_typed_video_binds_correct_probability_to_caller_id() -> None:
    class TorchWorker:
        def request(self, payload: dict[str, object]) -> dict[str, object]:
            assert payload["task"] == "video_score"
            return {
                "probabilities": [0.1, 0.9],
                "prediction_index": 1,
                "confidence": 0.9,
            }

    class VideoWorker:
        def request(self, payload: dict[str, object]) -> dict[str, object]:
            assert payload["task"] == "encode_video"
            return {"video_feature": [0.0] * 768, "encoder_metrics": {}}

    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = TorchWorker()
    dispatcher._videoprism_worker = VideoWorker()
    result = dispatcher.dispatch_typed(
        "video",
        {"video_path": "video.mp4", "question": "q", "candidate_statement": "s"},
        decision_specs=[
            TypedDecisionSpec(
                "video-1",
                "choice",
                ("candidate-wrong", "candidate-correct"),
                ("wrong", "correct"),
            )
        ],
    )

    assert result["decision_json"]["selected_option_id"] == "candidate-correct"
    assert [option["probability"] for option in result["decision_json"]["options"]] == [0.1, 0.9]


def test_dispatch_typed_perception_video_text_preserves_caller_option_order() -> None:
    class VideoMAEWorker:
        def request(self, payload: dict[str, object]) -> dict[str, object]:
            assert payload["task"] == "video_text"
            return {
                "probabilities": [0.15, 0.7, 0.15],
                "prediction_index": 1,
                "confidence": 0.7,
            }

    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher._request_id = 0
    dispatcher._torch_worker = None
    dispatcher._videoprism_worker = None
    dispatcher._videomae_worker = VideoMAEWorker()
    dispatcher.perception_video_readout = Path("readout.pt")
    spec = TypedDecisionSpec(
        "perception-1",
        "choice",
        ("third", "first", "second"),
        ("C: third answer", "A: first answer", "B: second answer"),
    )

    result = dispatcher.dispatch_typed(
        "video_text_videomae",
        {
            "video_path": "video.mp4",
            "question": "Which answer applies?",
            "options": list(spec.option_texts),
            "expected_media_sha256": "a" * 64,
        },
        decision_specs=[spec],
    )

    assert result["decision_json"]["selected_option_id"] == "first"
    assert [row["id"] for row in result["decision_json"]["options"]] == [
        "third",
        "first",
        "second",
    ]
    assert [row["probability"] for row in result["decision_json"]["options"]] == [
        0.15,
        0.7,
        0.15,
    ]


def test_videomae_worker_receives_perception_readout_and_assembly_minilm_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assembly = tmp_path / "assembly"
    text_model = assembly / "components/audio/models/text-minilm-en"
    text_model.mkdir(parents=True)
    model_root = tmp_path / "videomae"
    dependency_root = tmp_path / "dependencies"
    readout = tmp_path / "perception-readout.pt"
    for path in (model_root, dependency_root):
        path.mkdir()
    readout.write_bytes(b"pinned-readout-placeholder")
    captured: dict[str, object] = {}

    class StubWorker:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

    monkeypatch.setattr(runtime_dispatch_module, "JsonLineWorker", StubWorker)
    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher.videomae_model_dir = model_root
    dispatcher.videomae_dependency_dir = dependency_root
    dispatcher.videomae_readout = None
    dispatcher.perception_video_readout = readout
    dispatcher.assembly_dir = assembly
    dispatcher.worker_log_dir = tmp_path / "logs"
    dispatcher.device = "cpu"
    dispatcher._videomae_worker = None
    dispatcher._project_root = tmp_path
    dispatcher._worker_env = {}

    dispatcher._ensure_videomae()

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--perception-video-readout") + 1] == str(readout)
    assert command[command.index("--text-model-dir") + 1] == str(text_model)
    assert "--readout" not in command


def test_torch_worker_does_not_receive_videomae_only_perception_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assembly = tmp_path / "assembly"
    assembly.mkdir()
    (assembly / "components/audio/models/text-minilm-en").mkdir(parents=True)
    readout = tmp_path / "perception-readout.pt"
    readout.write_bytes(b"pinned-readout-placeholder")
    captured: dict[str, object] = {}

    class StubWorker:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

    monkeypatch.setattr(runtime_dispatch_module, "JsonLineWorker", StubWorker)
    dispatcher = object.__new__(PathARuntimeDispatcher)
    dispatcher.assembly_dir = assembly
    dispatcher.perception_video_readout = readout
    dispatcher.speech_commands_m1_readout = None
    dispatcher.worker_log_dir = tmp_path / "logs"
    dispatcher.device = "cpu"
    dispatcher._torch_worker = None
    dispatcher._project_root = tmp_path
    dispatcher._worker_env = {}

    dispatcher._ensure_torch()

    command = captured["command"]
    assert isinstance(command, list)
    worker_args = command[command.index("scripts.path_a_torch_worker") + 1 :]
    assert "--assembly" in worker_args
    assert "--device" in worker_args
    assert "--perception-video-readout" not in worker_args
    assert "--text-model-dir" not in worker_args


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
