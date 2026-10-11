"""Measure four-modality Path A dispatch with frozen development examples."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tiny_omni_decision.dataset import iter_local_rows, normalize_jsonl, sha256_file
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher
from tiny_omni_decision.schema import DatasetManifest


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_manifested_view(source: Path, target: Path) -> dict[str, Any]:
    """Create a clean hard-link view without modifying the source assembly."""
    manifest_path = source / "package-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if target.exists():
        raise FileExistsError(f"refusing to overwrite runtime package view: {target}")
    target.mkdir(parents=True)
    linked = 0
    copied = 0
    copied_bytes = 0
    for entry in manifest["files"]:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("package manifest contains an unsafe relative path")
        source_file = (source / relative).resolve(strict=True)
        if not source_file.is_relative_to(source):
            raise ValueError("package manifest file resolves outside source assembly")
        target_file = target / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_file, target_file)
            linked += 1
        except OSError:
            shutil.copy2(source_file, target_file)
            copied += 1
            copied_bytes += source_file.stat().st_size
    shutil.copy2(manifest_path, target / "package-manifest.json")
    return {
        "source_path": str(source),
        "view_path": str(target),
        "manifest_sha256": sha256_file(target / "package-manifest.json"),
        "manifest_file_count": len(manifest["files"]),
        "hardlinked_file_count": linked,
        "copied_file_count": copied,
        "copied_bytes": copied_bytes,
    }


def _choose_text(artifact_root: Path, manifest_path: Path) -> dict[str, Any]:
    run_dir = artifact_root / "text-probe-v0"
    reference_path = run_dir / "validation-predictions.jsonl"
    references = _jsonl(reference_path)
    reference = references[0]
    validation_path = artifact_root / "text" / "validation.jsonl"
    manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
    examples = list(
        normalize_jsonl(iter_local_rows(validation_path), manifest, "typed-decisions-synth")
    )
    example_by_id = {row.id: row for row in examples}
    example = next(
        (
            row for row in examples if row.id == reference["id"]
        ),
        None,
    )
    if example is None:
        raise ValueError(f"text validation state not found for {reference['id']}")
    if list(example.options) != reference["options"]:
        raise ValueError("text runtime options differ from the saved validation prediction")
    state_id = example.source_record_id.split(":", 1)[0]
    alternate_reference = next(
        (
            row
            for row in references[1:]
            if str(row.get("source_record_id", "")).split(":", 1)[0] == state_id
        ),
        reference,
    )
    alternate = example_by_id.get(str(alternate_reference["id"]), example)
    return {
        "task": "text",
        "payload": {
            "state": str(example.state),
            "question": str(example.question),
            "options": list(example.options),
        },
        "reference": {
            "sample_id": str(reference["id"]),
            "prediction_index": example.options.index(str(reference["prediction"])),
            "probabilities": reference["probabilities"],
        },
        "warm_payload": {
            "state": str(alternate.state),
            "question": str(alternate.question),
            "options": list(alternate.options),
        },
        "warm_reference": {
            "sample_id": str(alternate_reference["id"]),
            "prediction_index": alternate.options.index(str(alternate_reference["prediction"])),
            "probabilities": alternate_reference["probabilities"],
        },
        "source_files": [validation_path, reference_path, manifest_path],
    }


def _choose_audio(artifact_root: Path) -> dict[str, Any]:
    run_dir = artifact_root / "audio-multiquery-v0"
    reference_path = run_dir / "validation-multiquery-predictions.jsonl"
    references = _jsonl(reference_path)
    sample_id = str(references[0]["sample_id"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        grouped[str(row["sample_id"])].append(row)
    sample = grouped[sample_id]
    sample_dir = artifact_root / "speech-commands-sample-seed17-128-64-retry2"
    media_path = sample_dir / "media" / "validation" / f"{sample_id}.wav"
    audio_config = run_dir / "effective-config.yaml"
    if not media_path.is_file():
        raise FileNotFoundError(media_path)
    return {
        "task": "audio",
        "payload": {
            "audio_path": str(media_path),
            "queries": [
                {"question": row["question"], "options": row["options"]} for row in sample
            ],
        },
        "reference": {
            "sample_id": sample_id,
            "decisions": [
                {
                    "prediction_index": int(row["prediction_index"]),
                    "probabilities": row["probabilities"],
                    "question_type": row["question_type"],
                }
                for row in sample
            ],
        },
        "source_files": [media_path, reference_path, audio_config],
    }


def _choose_image(artifact_root: Path) -> dict[str, Any]:
    reference_path = (
        artifact_root
        / "image-videoprism-clevr4-runtime-validation-v2"
        / "runtime-predictions.jsonl"
    )
    references = _jsonl(reference_path)
    reference = references[0]
    alternate = next(
        (row for row in references[1:] if row["image_id"] == reference["image_id"]), reference
    )
    sample_dir = artifact_root / "clevr4-full-coverage-seed17"
    image_path = sample_dir / "val" / "images" / f"{reference['image_id']}.png"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return {
        "task": "image",
        "payload": {
            "image_path": str(image_path),
            "expected_media_sha256": sha256_file(image_path),
            "taxonomy": str(reference["taxonomy"]),
            "options": reference["options"],
        },
        "reference": {
            "sample_id": f"{reference['image_id']}:{reference['taxonomy']}",
            "prediction_index": int(reference["reference_prediction_index"]),
            "probabilities": reference["probabilities"],
        },
        "warm_payload": {
            "image_path": str(image_path),
            "expected_media_sha256": sha256_file(image_path),
            "taxonomy": str(alternate["taxonomy"]),
            "options": alternate["options"],
        },
        "warm_reference": {
            "sample_id": f"{alternate['image_id']}:{alternate['taxonomy']}",
            "prediction_index": int(alternate["reference_prediction_index"]),
            "probabilities": alternate["probabilities"],
        },
        "source_files": [image_path, reference_path],
    }


def _choose_video(artifact_root: Path) -> dict[str, Any]:
    run_dir = artifact_root / "clevrer-binary-reasoning-v0"
    rows_path = run_dir / "validation-binary-decisions.jsonl"
    prediction_path = run_dir / "validation-predictions.jsonl"
    rows = _jsonl(rows_path)
    predictions = {str(row["id"]): row for row in _jsonl(prediction_path)}
    row = rows[0]
    reference = predictions[str(row["id"])]
    alternate_row = next(
        (
            candidate
            for candidate in rows[1:]
            if candidate["media_sha256"] == row["media_sha256"]
            and candidate["id"] != row["id"]
        ),
        row,
    )
    alternate_reference = predictions[str(alternate_row["id"])]
    sample_dir = artifact_root / "clevrer-expanded-seed17-256train-200val"
    video_path = sample_dir / str(row["media_path"])
    if sha256_file(video_path) != str(row["media_sha256"]):
        raise ValueError("CLEVRER video differs from the frozen validation media hash")
    return {
        "task": "video",
        "payload": {
            "video_path": str(video_path),
            "expected_media_sha256": str(row["media_sha256"]),
            "question": str(row.get("source_question", row["question"])),
            "candidate_statement": str(row["candidate_statement"]),
        },
        "reference": {
            "sample_id": str(row["id"]),
            "prediction_index": int(reference["prediction_index"]),
            "probabilities": reference["probabilities"],
        },
        "warm_payload": {
            "video_path": str(video_path),
            "expected_media_sha256": str(alternate_row["media_sha256"]),
            "question": str(alternate_row.get("source_question", alternate_row["question"])),
            "candidate_statement": str(alternate_row["candidate_statement"]),
        },
        "warm_reference": {
            "sample_id": str(alternate_row["id"]),
            "prediction_index": int(alternate_reference["prediction_index"]),
            "probabilities": alternate_reference["probabilities"],
        },
        "source_files": [video_path, rows_path, prediction_path],
    }


def _compare_output(
    task_spec: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    reference = task_spec["reference"]
    if task_spec["task"] == "audio":
        decisions = response["decisions"]
        reference_decisions = reference["decisions"]
        if len(decisions) != len(reference_decisions):
            raise ValueError("audio multi-query result count differs from the reference")
        comparisons = [
            {
                "question_type": expected["question_type"],
                "runtime_prediction_index": int(actual["prediction_index"]),
                "reference_prediction_index": int(expected["prediction_index"]),
                "prediction_match": int(actual["prediction_index"])
                == int(expected["prediction_index"]),
                "max_probability_absolute_delta": max(
                    abs(float(left) - float(right))
                    for left, right in zip(
                        actual["probabilities"], expected["probabilities"], strict=True
                    )
                ),
            }
            for actual, expected in zip(decisions, reference_decisions, strict=True)
        ]
        return {
            "sample_id": reference["sample_id"],
            "comparisons": comparisons,
            "prediction_mismatches": sum(not row["prediction_match"] for row in comparisons),
            "max_probability_absolute_delta": max(
                row["max_probability_absolute_delta"] for row in comparisons
            ),
        }

    probabilities = [float(value) for value in response["probabilities"]]
    reference_probabilities = [float(value) for value in reference["probabilities"]]
    if len(probabilities) != len(reference_probabilities):
        raise ValueError(f"{task_spec['task']} output size differs from the reference")
    prediction_index = int(response["prediction_index"])
    reference_index = int(reference["prediction_index"])
    return {
        "sample_id": reference["sample_id"],
        "runtime_prediction_index": prediction_index,
        "reference_prediction_index": reference_index,
        "prediction_mismatches": int(prediction_index != reference_index),
        "max_probability_absolute_delta": max(
            abs(left - right)
            for left, right in zip(probabilities, reference_probabilities, strict=True)
        ),
    }


def _measure_task(
    task_spec: dict[str, Any],
    dispatcher: PathARuntimeDispatcher,
    *,
    is_service_cold_start: bool,
) -> dict[str, Any]:
    task = str(task_spec["task"])
    first = dispatcher.dispatch(task, task_spec["payload"])
    warm_spec = {
        **task_spec,
        "payload": task_spec.get("warm_payload", task_spec["payload"]),
        "reference": task_spec.get("warm_reference", task_spec["reference"]),
    }
    warm = dispatcher.dispatch(task, warm_spec["payload"])
    first_comparison = _compare_output(task_spec, first)
    warm_comparison = _compare_output(warm_spec, warm)
    def request_resources(response: dict[str, Any]) -> dict[str, Any]:
        if task == "image":
            return response.get("encoder_metrics", {})
        if task == "video":
            return response.get("video_encoder", {})
        return {
            key: response[key]
            for key in (
                "cuda_allocated_bytes_after_request",
                "cuda_peak_allocated_bytes_since_consolidation",
                "rss_after_request_bytes",
                "peak_working_set_bytes",
            )
            if key in response
        }

    return {
        "task": task,
        "is_service_cold_start": is_service_cold_start,
        "input": {
            "sample_id": task_spec["reference"]["sample_id"],
            "warm_sample_id": warm_spec["reference"]["sample_id"],
            "source_files": {
                str(path): sha256_file(path) for path in task_spec["source_files"]
            },
        },
        "first_request": {
            "dispatch_wall_seconds": first["dispatch_wall_seconds"],
            "worker_startup_wall_seconds": first["worker_startup_wall_seconds"],
            "worker_inference_seconds": first.get("worker_inference_seconds"),
            "encoder_metrics": first.get("video_encoder", first.get("encoder_metrics")),
            "resource_metrics": request_resources(first),
            "comparison": first_comparison,
        },
        "warm_request": {
            "dispatch_wall_seconds": warm["dispatch_wall_seconds"],
            "worker_startup_wall_seconds": warm["worker_startup_wall_seconds"],
            "worker_inference_seconds": warm.get("worker_inference_seconds"),
            "encoder_metrics": warm.get("video_encoder", warm.get("encoder_metrics")),
            "resource_metrics": request_resources(warm),
            "comparison": warm_comparison,
        },
        "parity_passed": all(
            result["prediction_mismatches"] == 0
            and result["max_probability_absolute_delta"] <= 1e-5
            for result in (first_comparison, warm_comparison)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    assembly = args.assembly.resolve(strict=True)
    artifact_root = args.artifact_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite dispatcher benchmark: {output_dir}")
    videoprism_python = args.videoprism_python.resolve(strict=True)
    videoprism_source = args.videoprism_source.resolve(strict=True)
    manifest_path = assembly / "package-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    specs = [
        _choose_video(artifact_root),
        _choose_image(artifact_root),
        _choose_text(artifact_root, repo / "manifests/candidates/typed-synth-validation.yaml"),
        _choose_audio(artifact_root),
    ]
    output_dir.mkdir(parents=True)
    view = _make_manifested_view(assembly, output_dir / "runtime-package-view")
    report_started = time.perf_counter()
    results = []
    dispatcher = None
    try:
        dispatcher = PathARuntimeDispatcher(
            Path(view["view_path"]),
            videoprism_python=videoprism_python,
            videoprism_source=videoprism_source,
            worker_log_dir=output_dir / "worker-logs",
            device=args.device,
        )
        bundle_verification_seconds = dispatcher.bundle_verification_seconds
        for index, spec in enumerate(specs):
            results.append(
                _measure_task(
                    spec,
                    dispatcher,
                    is_service_cold_start=index == 0,
                )
            )
        worker_metadata = dispatcher.worker_metadata
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed_runtime_benchmark",
            "error_type": type(error).__name__,
            "error": str(error),
            "source_commit": _git(repo, "rev-parse", "HEAD"),
            "assembly_manifest_sha256": manifest_hash,
            "runtime_package_view": view,
            "sealed_audit_loaded": False,
            "training_performed": False,
            "completed_tasks": results,
        }
        failure_path = output_dir / "dispatcher-benchmark-failure.json"
        failure_path.write_text(
            json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if dispatcher is not None:
            dispatcher.close()
    report = {
        "schema_version": 1,
        "status": "passed_parity"
        if all(result["parity_passed"] for result in results)
        else "failed_parity",
        "scope": (
            "first and warm request timing on frozen development-validation inputs: "
            "same Video scene with a second candidate, same image with a second taxonomy, "
            "same text state with a second question, and four Audio queries; "
            "not a quality evaluation"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": _git(repo, "rev-parse", "HEAD"),
        "source_branch": _git(repo, "branch", "--show-current"),
        "source_worktree_status": _git(repo, "status", "--short"),
        "implementation_sha256": {
            "runtime_dispatch.py": sha256_file(
                repo / "src/tiny_omni_decision/runtime_dispatch.py"
            ),
            "path_a_torch_worker.py": sha256_file(repo / "scripts/path_a_torch_worker.py"),
            "path_a_videoprism_worker.py": sha256_file(
                repo / "scripts/path_a_videoprism_worker.py"
            ),
            "benchmark_path_a_dispatcher.py": sha256_file(Path(__file__).resolve()),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "requested_torch_device": args.device,
            "videoprism_python": str(videoprism_python),
            "videoprism_source_commit": _git(videoprism_source, "rev-parse", "HEAD"),
        },
        "assembly": {
            "path": str(assembly),
            "manifest_sha256": manifest_hash,
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            "runtime_package_view": view,
            "verification_seconds": bundle_verification_seconds,
        },
        "validation_sources_are_read_only": True,
        "sealed_audit_loaded": False,
        "training_performed": False,
        "runtime_workers": worker_metadata,
        "runtime_resource_summary": {
            "torch_rss_after_load_bytes": worker_metadata["torch"][
                "rss_after_load_bytes"
            ],
            "videoprism_rss_after_load_bytes": worker_metadata["videoprism"][
                "rss_after_load_bytes"
            ],
            "combined_rss_after_load_bytes": sum(
                int(worker_metadata[name]["rss_after_load_bytes"])
                for name in ("torch", "videoprism")
            ),
            "torch_startup_peak_cuda_allocated_bytes": worker_metadata["torch"][
                "startup_peak_cuda_allocated_bytes"
            ],
            "torch_cuda_allocated_bytes_after_load": worker_metadata["torch"][
                "cuda_allocated_bytes_after_load"
            ],
            "torch_peak_working_set_after_load_bytes": worker_metadata["torch"][
                "peak_working_set_after_load_bytes"
            ],
        },
        "modalities": results,
        "total_sequential_benchmark_wall_seconds": time.perf_counter() - report_started,
    }
    report_path = output_dir / "dispatcher-benchmark.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_hash = sha256_file(report_path)
    report_path.with_suffix(".sha256").write_text(report_hash + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_path": str(report_path),
                "report_sha256": report_hash,
                "modalities": [
                    {
                        "task": row["task"],
                        "parity_passed": row["parity_passed"],
                        "first_request_seconds": row["first_request"][
                            "dispatch_wall_seconds"
                        ],
                        "warm_request_seconds": row["warm_request"][
                            "dispatch_wall_seconds"
                        ],
                    }
                    for row in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if report["status"] != "passed_parity":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
