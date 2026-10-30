"""Measure Path A dispatch latency across distinct saved validation assets."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.benchmark_path_a_dispatcher import (  # noqa: E402
    _git,
    _jsonl,
    _make_manifested_view,
    _measure_task,
)
from tiny_omni_decision.dataset import iter_local_rows, normalize_jsonl, sha256_file  # noqa: E402
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher  # noqa: E402
from tiny_omni_decision.schema import DatasetManifest  # noqa: E402


def _choose_videos(root: Path, count: int) -> list[dict[str, Any]]:
    run = root / "clevrer-binary-reasoning-v0"
    rows_path = run / "validation-binary-decisions.jsonl"
    predictions_path = run / "validation-predictions.jsonl"
    rows = _jsonl(rows_path)
    predictions = {str(row["id"]): row for row in _jsonl(predictions_path)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["media_sha256"])].append(row)
    sample_dir = root / "clevrer-expanded-seed17-256train-200val"
    specs = []
    for media_sha, scene_rows in groups.items():
        if len(scene_rows) < 2:
            continue
        first, alternate = scene_rows[:2]
        video_path = sample_dir / str(first["media_path"])
        if sha256_file(video_path) != media_sha:
            raise ValueError(f"CLEVRER media hash mismatch for {video_path}")
        if first["scene_group_id"] != alternate["scene_group_id"]:
            raise ValueError("rows with identical video hash have different scene groups")
        specs.append(
            {
                "task": "video",
                "payload": {
                    "video_path": str(video_path),
                    "expected_media_sha256": media_sha,
                    "question": str(first.get("source_question", first["question"])),
                    "candidate_statement": str(first["candidate_statement"]),
                },
                "reference": {
                    "sample_id": str(first["id"]),
                    "prediction_index": int(predictions[str(first["id"])]["prediction_index"]),
                    "probabilities": predictions[str(first["id"])]["probabilities"],
                },
                "warm_payload": {
                    "video_path": str(video_path),
                    "expected_media_sha256": media_sha,
                    "question": str(alternate.get("source_question", alternate["question"])),
                    "candidate_statement": str(alternate["candidate_statement"]),
                },
                "warm_reference": {
                    "sample_id": str(alternate["id"]),
                    "prediction_index": int(
                        predictions[str(alternate["id"])]["prediction_index"]
                    ),
                    "probabilities": predictions[str(alternate["id"])]["probabilities"],
                },
                "asset_id": media_sha,
                "source_files": [video_path, rows_path, predictions_path],
            }
        )
        if len(specs) == count:
            break
    if len(specs) < count:
        raise ValueError(f"requested {count} Video scenes but only found {len(specs)}")
    return specs


def _choose_images(root: Path, count: int) -> list[dict[str, Any]]:
    reference_path = (
        root / "image-videoprism-clevr4-runtime-validation-v2" / "runtime-predictions.jsonl"
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(reference_path):
        groups[str(row["image_id"])].append(row)
    sample_dir = root / "clevr4-full-coverage-seed17"
    specs = []
    for image_id, rows in groups.items():
        if len(rows) < 2:
            continue
        first, alternate = rows[:2]
        image_path = sample_dir / "val" / "images" / f"{image_id}.png"
        media_sha = sha256_file(image_path)
        for row in (first, alternate):
            if row["image_id"] != image_id:
                raise ValueError("image taxonomy rows have different image identities")
        specs.append(
            {
                "task": "image",
                "payload": {
                    "image_path": str(image_path),
                    "expected_media_sha256": media_sha,
                    "taxonomy": str(first["taxonomy"]),
                    "options": first["options"],
                },
                "reference": {
                    "sample_id": f"{image_id}:{first['taxonomy']}",
                    "prediction_index": int(first["reference_prediction_index"]),
                    "probabilities": first["probabilities"],
                },
                "warm_payload": {
                    "image_path": str(image_path),
                    "expected_media_sha256": media_sha,
                    "taxonomy": str(alternate["taxonomy"]),
                    "options": alternate["options"],
                },
                "warm_reference": {
                    "sample_id": f"{image_id}:{alternate['taxonomy']}",
                    "prediction_index": int(alternate["reference_prediction_index"]),
                    "probabilities": alternate["probabilities"],
                },
                "asset_id": image_id,
                "source_files": [image_path, reference_path],
            }
        )
        if len(specs) == count:
            break
    if len(specs) < count:
        raise ValueError(f"requested {count} Image assets but only found {len(specs)}")
    return specs


def _choose_texts(root: Path, manifest_path: Path, count: int) -> list[dict[str, Any]]:
    run = root / "text-probe-v0"
    reference_path = run / "validation-predictions.jsonl"
    validation_path = root / "text" / "validation.jsonl"
    refs = _jsonl(reference_path)
    manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
    examples = list(
        normalize_jsonl(iter_local_rows(validation_path), manifest, "typed-decisions-synth")
    )
    by_id = {row.id: row for row in examples}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in refs:
        groups[str(row["source_record_id"]).split(":", 1)[0]].append(row)
    specs = []
    for state_id, rows in groups.items():
        if len(rows) < 2:
            continue
        first_ref, alternate_ref = rows[:2]
        first = by_id[str(first_ref["id"])]
        alternate = by_id[str(alternate_ref["id"])]
        if first.source_record_id.split(":", 1)[0] != state_id:
            raise ValueError("normalized Text state identity differs from saved reference")
        if (
            list(first.options) != first_ref["options"]
            or list(alternate.options) != alternate_ref["options"]
        ):
            raise ValueError("normalized Text option order differs from saved references")
        specs.append(
            {
                "task": "text",
                "payload": {
                    "state": str(first.state),
                    "question": str(first.question),
                    "options": list(first.options),
                },
                "reference": {
                    "sample_id": str(first_ref["id"]),
                    "prediction_index": first.options.index(str(first_ref["prediction"])),
                    "probabilities": first_ref["probabilities"],
                },
                "warm_payload": {
                    "state": str(alternate.state),
                    "question": str(alternate.question),
                    "options": list(alternate.options),
                },
                "warm_reference": {
                    "sample_id": str(alternate_ref["id"]),
                    "prediction_index": alternate.options.index(str(alternate_ref["prediction"])),
                    "probabilities": alternate_ref["probabilities"],
                },
                "asset_id": state_id,
                "source_files": [validation_path, reference_path, manifest_path],
            }
        )
        if len(specs) == count:
            break
    if len(specs) < count:
        raise ValueError(f"requested {count} Text states but only found {len(specs)}")
    return specs


def _choose_audio(root: Path, count: int) -> list[dict[str, Any]]:
    run = root / "audio-multiquery-v0"
    reference_path = run / "validation-multiquery-predictions.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(reference_path):
        grouped[str(row["sample_id"])].append(row)
    sample_dir = root / "speech-commands-sample-seed17-128-64-retry2"
    specs = []
    for sample_id, rows in grouped.items():
        media_path = sample_dir / "media" / "validation" / f"{sample_id}.wav"
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        specs.append(
            {
                "task": "audio",
                "payload": {
                    "audio_path": str(media_path),
                    "queries": [
                        {"question": row["question"], "options": row["options"]}
                        for row in rows
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
                        for row in rows
                    ],
                },
                "asset_id": sample_id,
                "source_files": [
                    media_path,
                    reference_path,
                    run / "effective-config.yaml",
                ],
            }
        )
        if len(specs) == count:
            break
    if len(specs) < count:
        raise ValueError(f"requested {count} Audio clips but only found {len(specs)}")
    return specs


def _distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def nearest_rank(percentile: float) -> float:
        index = max(0, min(len(ordered) - 1, int(percentile * len(ordered) + 0.999999) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90_nearest_rank": nearest_rank(0.9),
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--samples-per-modality", type=int, default=8)
    args = parser.parse_args()
    if args.samples_per_modality < 2:
        raise ValueError("at least two distinct assets per modality are required")

    assembly = args.assembly.resolve(strict=True)
    artifact_root = args.artifact_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite latency run: {output_dir}")
    videoprism_python = args.videoprism_python.resolve(strict=True)
    videoprism_source = args.videoprism_source.resolve(strict=True)
    manifest_path = assembly / "package-manifest.json"
    manifest_hash = sha256_file(manifest_path)
    count = args.samples_per_modality
    specs_by_task = {
        "video": _choose_videos(artifact_root, count),
        "image": _choose_images(artifact_root, count),
        "text": _choose_texts(
            artifact_root,
            REPO / "manifests/candidates/typed-synth-validation.yaml",
            count,
        ),
        "audio": _choose_audio(artifact_root, count),
    }

    output_dir.mkdir(parents=True)
    view = _make_manifested_view(assembly, output_dir / "runtime-package-view")
    dispatcher = None
    results = []
    started = time.perf_counter()
    try:
        dispatcher = PathARuntimeDispatcher(
            Path(view["view_path"]),
            videoprism_python=videoprism_python,
            videoprism_source=videoprism_source,
            worker_log_dir=output_dir / "worker-logs",
            device=args.device,
        )
        cold_start_pending = True
        for _task, task_specs in specs_by_task.items():
            for spec in task_specs:
                result = _measure_task(
                    spec,
                    dispatcher,
                    is_service_cold_start=cold_start_pending,
                )
                cold_start_pending = False
                result["asset_id"] = spec["asset_id"]
                results.append(result)
        workers = dispatcher.worker_metadata
        verification_seconds = dispatcher.bundle_verification_seconds
    except Exception as error:
        failure = {
            "status": "failed_runtime_benchmark",
            "error_type": type(error).__name__,
            "error": str(error),
            "source_commit": _git(REPO, "rev-parse", "HEAD"),
            "assembly_manifest_sha256": manifest_hash,
            "runtime_package_view": view,
            "completed_requests": results,
            "sealed_audit_loaded": False,
            "training_performed": False,
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if dispatcher is not None:
            dispatcher.close()

    summaries = {}
    for task, _task_specs in specs_by_task.items():
        rows = [row for row in results if row["task"] == task]
        first_times = [float(row["first_request"]["dispatch_wall_seconds"]) for row in rows]
        warm_times = [float(row["warm_request"]["dispatch_wall_seconds"]) for row in rows]
        steady_first_times = [
            float(row["first_request"]["dispatch_wall_seconds"])
            for row in rows
            if not row["is_service_cold_start"]
        ]
        summaries[task] = {
            "unique_asset_count": len({row["asset_id"] for row in rows}),
            "first_request_dispatch_seconds": _distribution(first_times),
            "first_request_steady_service_seconds": _distribution(steady_first_times),
            "same_asset_warm_request_dispatch_seconds": _distribution(warm_times),
            "requests_with_exact_saved_prediction_parity": sum(
                row["parity_passed"] for row in rows
            ),
            "requests_total": len(rows),
        }

    report = {
        "schema_version": 1,
        "status": (
            "passed_parity" if all(row["parity_passed"] for row in results) else "failed_parity"
        ),
        "scope": (
            "steady and first-service latency distributions over distinct saved "
            "development-validation assets; same-asset second query/taxonomy/candidate "
            "tests feature reuse; not a quality evaluation"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": _git(REPO, "rev-parse", "HEAD"),
        "source_branch": _git(REPO, "branch", "--show-current"),
        "source_worktree_status": _git(REPO, "status", "--short"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "input_selection": {
            "selection_policy": (
                "first rows in saved validation order with distinct media/state identities"
            ),
            "samples_per_modality": count,
            "asset_id_semantics": {
                "video": "video media SHA-256 (CLEVRER scene identity)",
                "image": "CLEVR-4 image_id",
                "text": "normalized source state identity",
                "audio": "Speech Commands sample_id",
            },
        },
        "host": {
            "platform": __import__("platform").platform(),
            "python": __import__("platform").python_version(),
            "requested_torch_device": args.device,
            "videoprism_python": str(videoprism_python),
            "videoprism_source_commit": _git(videoprism_source, "rev-parse", "HEAD"),
        },
        "assembly": {
            "path": str(assembly),
            "manifest_sha256": manifest_hash,
            "runtime_package_view": view,
            "verification_seconds": verification_seconds,
        },
        "validation_sources_are_read_only": True,
        "sealed_audit_loaded": False,
        "training_performed": False,
        "runtime_workers": workers,
        "runtime_resource_summary": {
            "torch_rss_after_load_bytes": workers["torch"]["rss_after_load_bytes"],
            "videoprism_rss_after_load_bytes": workers["videoprism"]["rss_after_load_bytes"],
            "combined_rss_after_load_bytes": sum(
                int(workers[name]["rss_after_load_bytes"]) for name in ("torch", "videoprism")
            ),
            "torch_cuda_allocated_bytes_after_load": workers["torch"][
                "cuda_allocated_bytes_after_load"
            ],
            "torch_startup_peak_cuda_allocated_bytes": workers["torch"][
                "startup_peak_cuda_allocated_bytes"
            ],
        },
        "latency_summary": summaries,
        "measurements": results,
        "total_sequential_wall_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "latency-distribution.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(report_path)
    report_path.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "sha256": digest,
                "latency_summary": summaries,
                "total_sequential_wall_seconds": report["total_sequential_wall_seconds"],
            },
            indent=2,
        )
    )
    if report["status"] != "passed_parity":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
