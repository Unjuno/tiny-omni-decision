"""JSONL entry point for the existing task-scoped Path A runtime bundle."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, TextIO

from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher, TypedDecisionSpec
from tiny_omni_decision.speech_commands_m1_contract import SPEECH_COMMANDS_M1_LABELS

TASKS = {
    "typed-decisions-synth-text-v0": "text",
    "path-a-image-videoprism-clevr4-runtime-v0": "image",
    "path-a-image-videomaev2-clevr4-100k-runtime-v0": "image_videomae",
    "path-a-audio-multiquery-runtime-v0": "audio",
    "path-a-audio-speech-commands-m1-v0": "audio_keyword",
    "clevrer-videoprism-binary-feature-decision-v0": "video",
}
MEDIA_TASKS = {"image", "image_videomae", "audio", "audio_keyword", "video"}
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _validate_payload(task: str, payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if task == "text":
        _required_text(payload.get("state"), "payload.state")
        _required_text(payload.get("question"), "payload.question")
        if not isinstance(payload.get("options"), list):
            raise ValueError("payload.options must be an array")
    elif task in {"image", "image_videomae"}:
        _required_text(payload.get("image_path"), "payload.image_path")
        _required_text(payload.get("taxonomy"), "payload.taxonomy")
        if not isinstance(payload.get("options"), list):
            raise ValueError("payload.options must be an array")
    elif task == "audio":
        _required_text(payload.get("audio_path"), "payload.audio_path")
        queries = payload.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError("payload.queries must be a nonempty array")
        for index, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(f"payload.queries[{index}] must be an object")
            _required_text(query.get("question"), f"payload.queries[{index}].question")
            if not isinstance(query.get("options"), list):
                raise ValueError(f"payload.queries[{index}].options must be an array")
    elif task == "audio_keyword":
        _required_text(payload.get("audio_path"), "payload.audio_path")
    elif task == "video":
        _required_text(payload.get("video_path"), "payload.video_path")
        _required_text(payload.get("question"), "payload.question")
        _required_text(payload.get("candidate_statement"), "payload.candidate_statement")
    else:  # pragma: no cover - guarded by the task registry
        raise ValueError(f"unsupported Path A task: {task}")

    if task in MEDIA_TASKS:
        media_hash = payload.get("expected_media_sha256")
        if not isinstance(media_hash, str) or not SHA256_PATTERN.fullmatch(media_hash):
            raise ValueError(
                "payload.expected_media_sha256 must be a 64-character SHA-256 hex digest"
            )
    return payload


def _decision_specs(value: object) -> list[TypedDecisionSpec]:
    if not isinstance(value, list) or not value:
        raise ValueError("decision_specs must be a nonempty array")
    specs: list[TypedDecisionSpec] = []
    for index, raw_spec in enumerate(value):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"decision_specs[{index}] must be an object")
        options = raw_spec.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError(f"decision_specs[{index}].options must be a nonempty array")
        option_ids: list[str] = []
        option_texts: list[str] = []
        for option_index, option in enumerate(options):
            if not isinstance(option, dict):
                raise ValueError(
                    f"decision_specs[{index}].options[{option_index}] must be an object"
                )
            option_ids.append(
                _required_text(
                    option.get("id"), f"decision_specs[{index}].options[{option_index}].id"
                )
            )
            option_texts.append(
                _required_text(
                    option.get("text"), f"decision_specs[{index}].options[{option_index}].text"
                )
            )
        ordinal_values = raw_spec.get("ordinal_values")
        if ordinal_values is not None and not isinstance(ordinal_values, list):
            raise ValueError(f"decision_specs[{index}].ordinal_values must be an array")
        specs.append(
            TypedDecisionSpec(
                request_id=_required_text(
                    raw_spec.get("request_id"), f"decision_specs[{index}].request_id"
                ),
                decision_type=_required_text(
                    raw_spec.get("decision_type"), f"decision_specs[{index}].decision_type"
                ),
                option_ids=tuple(option_ids),
                option_texts=tuple(option_texts),
                ordinal_values=tuple(ordinal_values) if ordinal_values is not None else None,
            )
        )
    return specs


def _parse_request(request: object) -> tuple[str, str, dict[str, Any], list[TypedDecisionSpec]]:
    if not isinstance(request, dict):
        raise ValueError("each JSONL line must contain an object")
    request_id = _required_text(request.get("request_id"), "request_id")
    task_id = _required_text(request.get("task_id"), "task_id")
    task = TASKS.get(task_id)
    if task is None:
        raise ValueError(f"unsupported task_id: {task_id}")
    payload = _validate_payload(task, request.get("payload"))
    specs = _decision_specs(request.get("decision_specs"))
    if task == "audio":
        if len(specs) != len(payload["queries"]):
            raise ValueError("audio requires one decision_spec per query")
    elif task == "audio_keyword":
        if len(specs) != 1:
            raise ValueError("audio_keyword requires exactly one decision_spec")
        if specs[0].decision_type != "choice":
            raise ValueError("audio_keyword requires decision_type 'choice'")
        if len(specs[0].option_texts) != len(SPEECH_COMMANDS_M1_LABELS) or set(
            specs[0].option_texts
        ) != set(SPEECH_COMMANDS_M1_LABELS):
            raise ValueError(
                "audio_keyword requires all ten fixed Speech Commands labels exactly once"
            )
    elif len(specs) != 1:
        raise ValueError("non-audio tasks require exactly one decision_spec")
    if task == "video" and specs[0].option_texts != ("wrong", "correct"):
        raise ValueError("video decision option texts must be ('wrong', 'correct')")
    return request_id, task_id, payload, specs


def run_jsonl(
    input_stream: TextIO,
    output_stream: TextIO,
    dispatcher: PathARuntimeDispatcher,
) -> None:
    """Read task-scoped requests and emit one success/error JSON object per line."""

    for line_number, line in enumerate(input_stream, start=1):
        if not line.strip():
            continue
        request: object = None
        try:
            request = json.loads(line)
            request_id, task_id, payload, specs = _parse_request(request)
            task = TASKS[task_id]
            result = dispatcher.dispatch_typed(task, payload, decision_specs=specs)
            runtime = {
                key: result[key]
                for key in (
                    "dispatch_wall_seconds",
                    "worker_startup_wall_seconds",
                    "worker_inference_seconds",
                    "encoder_metrics",
                    "keyword_metrics",
                    "video_encoder",
                    "image_encoder",
                    "feature_shape",
                    "cuda_allocated_bytes_after_request",
                    "cuda_peak_allocated_bytes",
                )
                if key in result
            }
            worker_metadata = getattr(dispatcher, "worker_metadata", {})
            if isinstance(worker_metadata, Mapping):
                loaded_workers = {
                    name: {key: value for key, value in metadata.items() if key != "process_id"}
                    for name, metadata in worker_metadata.items()
                    if isinstance(metadata, Mapping) and metadata.get("ready") is True
                }
                if loaded_workers:
                    runtime["worker_metadata"] = loaded_workers
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "request_id": request_id,
                        "task_id": task_id,
                        "decision": result["decision_json"],
                        "runtime": runtime,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                file=output_stream,
                flush=True,
            )
        except Exception as error:
            request_id = request.get("request_id") if isinstance(request, Mapping) else None
            print(
                json.dumps(
                    {
                        "status": "error",
                        "request_id": request_id,
                        "line": line_number,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                file=output_stream,
                flush=True,
            )
