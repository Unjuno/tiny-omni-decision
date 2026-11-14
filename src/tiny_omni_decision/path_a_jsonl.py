"""JSONL entry point for the frozen-feature Path A decision runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from .runtime_dispatch import TypedDecisionSpec


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON array of strings")
    return tuple(value)


def _decision_spec(value: Any) -> TypedDecisionSpec:
    if not isinstance(value, dict):
        raise ValueError("each decision contract must be a JSON object")
    required = {"request_id", "decision_type", "option_ids", "option_texts"}
    allowed = required | {"ordinal_values"}
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing or unknown:
        raise ValueError(
            f"decision contract keys invalid (missing={sorted(missing)}, "
            f"unknown={sorted(unknown)})"
        )
    request_id = value["request_id"]
    decision_type = value["decision_type"]
    if not isinstance(request_id, str) or not isinstance(decision_type, str):
        raise ValueError("request_id and decision_type must be strings")
    ordinal_values = value.get("ordinal_values")
    if ordinal_values is not None and (
        not isinstance(ordinal_values, list)
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in ordinal_values
        )
    ):
        raise ValueError("ordinal_values must be a JSON array of numbers")
    return TypedDecisionSpec(
        request_id=request_id,
        decision_type=decision_type,
        option_ids=_string_tuple(value["option_ids"], "option_ids"),
        option_texts=_string_tuple(value["option_texts"], "option_texts"),
        ordinal_values=tuple(ordinal_values) if ordinal_values is not None else None,
    )


def parse_request(value: Any) -> tuple[str, dict[str, Any], list[TypedDecisionSpec]]:
    """Validate a JSONL request and bind its typed output contract."""

    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    task = value.get("task")
    payload = value.get("payload")
    if not isinstance(task, str) or task not in {"text", "image", "audio", "video"}:
        raise ValueError("task must be one of text, image, audio, video")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if task == "audio":
        if set(value) != {"task", "payload", "decisions"}:
            raise ValueError("audio request keys must be task, payload, and decisions")
        raw_specs = value["decisions"]
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("audio decisions must be a nonempty JSON array")
    else:
        if set(value) != {"task", "payload", "decision"}:
            raise ValueError("request keys must be task, payload, and decision")
        raw_specs = [value["decision"]]
    return task, payload, [_decision_spec(item) for item in raw_specs]


def check_parameter_budget(manifest: dict[str, Any], maximum: int) -> int:
    """Return the verified package parameter count or fail closed over budget."""

    if isinstance(maximum, bool) or maximum < 1:
        raise ValueError("max-parameters must be a positive integer")
    try:
        count = int(manifest["unique_runtime_parameters"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("assembly manifest has no valid unique parameter count") from error
    if count < 1:
        raise ValueError("assembly manifest unique parameter count must be positive")
    if count > maximum:
        raise ValueError(
            f"runtime has {count} unique parameters, exceeding the configured limit {maximum}"
        )
    return count


def run_jsonl(
    source: Iterable[str], sink: TextIO, dispatcher: Any
) -> None:
    """Dispatch each nonblank JSONL request and emit only typed decision JSON."""

    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
        task, payload, specs = parse_request(request)
        result = dispatcher.dispatch_typed(task, payload, decision_specs=specs)
        record = result.get("decision_json")
        if record is None:
            raise RuntimeError(f"runtime returned no typed decision JSON for {task}")
        sink.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        sink.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--worker-log-dir", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--max-parameters", type=int, default=200_000_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    from .runtime_dispatch import PathARuntimeDispatcher

    try:
        metadata_path = args.metadata_output.resolve()
        if metadata_path.exists():
            raise FileExistsError(f"refusing to overwrite runtime metadata: {metadata_path}")
        dispatcher = PathARuntimeDispatcher(
            args.assembly,
            videoprism_python=args.videoprism_python,
            videoprism_source=args.videoprism_source,
            worker_log_dir=args.worker_log_dir,
            device=args.device,
        )
        with dispatcher:
            unique_parameters = check_parameter_budget(
                dispatcher.manifest, args.max_parameters
            )
            run_jsonl(sys.stdin, sys.stdout, dispatcher)
            metadata = {
                "schema_version": 1,
                "device": args.device,
                "assembly_manifest_sha256": hashlib.sha256(
                    (args.assembly / "package-manifest.json").read_bytes()
                ).hexdigest(),
                "unique_runtime_parameters": unique_parameters,
                "package_file_bytes_excluding_manifest": int(
                    dispatcher.manifest["package_file_bytes_excluding_manifest"]
                ),
                "package_target_met": bool(dispatcher.manifest["package_target_met"]),
                "workers": dispatcher.worker_metadata,
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(metadata, stream, indent=2, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"path-a-decision-json: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
