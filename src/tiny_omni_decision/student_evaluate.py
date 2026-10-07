"""Run resumable, hash-checked validation for the EmbeddingGemma 2 student."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_structured_file
from .schema import BaseModelManifest
from .student_eval import (
    compare_student_predictions,
    evaluate_student_examples,
    load_fixed_validation_snapshot,
    validate_local_media_paths,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verify_local_model_files(model_path: Path, manifest: BaseModelManifest) -> dict[str, str]:
    verified: dict[str, str] = {}
    for entry in manifest.files:
        name = entry.get("path")
        if not name or ("sha256" not in entry and "size_bytes" not in entry):
            continue
        path = model_path / name
        if not path.is_file():
            raise ValueError(f"pinned model file is missing: {path}")
        digest = _sha256(path)
        if "sha256" in entry and digest != entry["sha256"]:
            raise ValueError(f"pinned model file hash mismatch: {name}")
        if "size_bytes" in entry and path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"pinned model file size mismatch: {name}")
        verified[name] = digest

    notes = manifest.notes
    note_hashes = {
        "config.json": notes.get("inspected_config_sha256"),
        "processor_config.json": notes.get("inspected_processor_config_sha256"),
        "preprocessor_config.json": notes.get("inspected_preprocessor_config_sha256"),
    }
    for name, expected_digest in note_hashes.items():
        if expected_digest is None:
            continue
        path = model_path / name
        if not path.is_file() or _sha256(path) != expected_digest:
            raise ValueError(f"pinned model file hash mismatch: {name}")
        verified[name] = expected_digest

    if not verified:
        raise ValueError("model manifest has no locally verifiable pinned files")
    return verified


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--validation-snapshot", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=7.0)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if (
        args.threads < 1
        or not math.isfinite(args.minimum_free_vram_gib)
        or args.minimum_free_vram_gib <= 0
    ):
        raise SystemExit("threads and minimum-free-vram-gib must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise SystemExit("temperature must be finite and positive")
    manifest = BaseModelManifest.model_validate(load_structured_file(args.model_manifest))
    verified_files = _verify_local_model_files(args.model_path, manifest)
    examples, teacher_predictions = load_fixed_validation_snapshot(
        args.validation_snapshot,
        args.selection_manifest,
        source_validation_path=args.source_validation,
        teacher_predictions_path=args.teacher_predictions,
    )
    media_counts = validate_local_media_paths(examples, args.data_root)

    import torch
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        required_bytes = int(args.minimum_free_vram_gib * 1024**3)
        if free_bytes < required_bytes:
            raise SystemExit(
                f"insufficient free VRAM: {free_bytes} bytes free, "
                f"{required_bytes} bytes required by the safety threshold"
            )
    else:
        free_bytes = None
        total_bytes = None

    identity = {
        "base_repo_id": manifest.repo_id,
        "base_revision": manifest.revision,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip(),
        "evaluator_code_sha256": {
            "student_eval.py": _sha256(Path(__file__).with_name("student_eval.py")),
            "student_evaluate.py": _sha256(Path(__file__)),
            "student.py": _sha256(Path(__file__).with_name("student.py")),
            "decision_math.py": _sha256(Path(__file__).with_name("decision_math.py")),
            "schema.py": _sha256(Path(__file__).with_name("schema.py")),
        },
        "model_path": str(args.model_path.resolve()),
        "model_manifest_sha256": _sha256(args.model_manifest),
        "verified_model_files_sha256": verified_files,
        "validation_snapshot_path": str(args.validation_snapshot.resolve()),
        "validation_snapshot_sha256": _sha256(args.validation_snapshot),
        "selection_manifest_path": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": _sha256(args.selection_manifest),
        "source_validation_path": str(args.source_validation.resolve()),
        "source_validation_sha256": _sha256(args.source_validation),
        "teacher_predictions_path": str(args.teacher_predictions.resolve()),
        "teacher_predictions_sha256": _sha256(args.teacher_predictions),
        "data_root": str(args.data_root.resolve()),
        "temperature": args.temperature,
        "ece_bins": 15,
        "device": args.device,
        "threads": args.threads,
        "dtype": "bfloat16",
        "attention_backend": "eager",
        "validation_count": len(examples),
        "validation_media_counts": media_counts,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run-metadata.json"
    partial_path = args.output_dir / "predictions.partial.jsonl"
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    if predictions_path.exists():
        raise SystemExit(f"completed predictions already exist: {predictions_path}")
    if run_path.exists():
        run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
        if run_metadata.get("identity") != identity:
            raise SystemExit("existing run metadata does not match this evaluation request")
        run_metadata.setdefault("resume_count", 0)
        run_metadata["resume_count"] += 1
    elif partial_path.exists():
        raise SystemExit("partial predictions exist without their run metadata; refusing resume")
    else:
        run_metadata = {
            "identity": identity,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "state": "in_progress",
            "resume_count": 0,
        }

    partial_bytes = partial_path.read_bytes() if partial_path.exists() else b""
    if partial_bytes and not partial_bytes.endswith(b"\n"):
        raise SystemExit("partial prediction file has a truncated final record")
    existing = [json.loads(line) for line in partial_bytes.splitlines() if line]
    run_metadata["partial_records_before_run"] = len(existing)
    run_metadata["python_version"] = platform.python_version()
    run_metadata["platform"] = platform.platform()
    run_metadata["cpu"] = platform.processor()
    run_metadata["torch_version"] = torch.__version__
    run_metadata["transformers_version"] = importlib.metadata.version("transformers")
    run_metadata["command_line"] = [sys.executable, *sys.argv]
    run_metadata["cuda_runtime_version"] = torch.version.cuda
    run_metadata["gpu_name"] = torch.cuda.get_device_name() if args.device == "cuda" else None
    run_metadata["gpu_free_bytes_at_start"] = free_bytes
    run_metadata["gpu_total_bytes"] = total_bytes
    _write_json_atomic(run_path, run_metadata)

    started = datetime.now(UTC)
    model = AutoModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    append_mode = "ab" if partial_bytes else "wb"
    saved_count = len(existing)
    with partial_path.open(append_mode) as output_stream:

        def save_prediction(record: dict[str, Any]) -> None:
            nonlocal saved_count
            output_stream.write(
                (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
            output_stream.flush()
            os.fsync(output_stream.fileno())
            saved_count += 1
            if saved_count % 16 == 0 or saved_count == len(examples):
                print(f"saved {saved_count}/{len(examples)}: {record['sample_id']}")

        student_metrics, student_predictions = evaluate_student_examples(
            model,
            processor,
            examples,
            data_root=args.data_root,
            temperature=args.temperature,
            ece_bins=15,
            existing_predictions=existing,
            on_prediction=save_prediction,
        )

    comparison = compare_student_predictions(
        student_predictions, teacher_predictions, ece_bins=15
    )
    output_stream_path = args.output_dir / "predictions.jsonl"
    _write_json_atomic(
        metrics_path,
        {
            "run_identity": identity,
            "student_metrics": student_metrics,
            "paired_teacher_comparison": comparison,
            "teacher_evaluation_scope": (
                "same frozen validation sample IDs and option order; not final evaluation"
            ),
        },
    )
    os.replace(partial_path, output_stream_path)
    run_metadata.update(
        {
            "state": "complete",
            "ended_at_utc": datetime.now(UTC).isoformat(),
            "last_attempt_duration_seconds": (datetime.now(UTC) - started).total_seconds(),
            "prediction_count": len(student_predictions),
            "predictions_sha256": _sha256(output_stream_path),
        }
    )
    _write_json_atomic(run_path, run_metadata)
    print(f"evaluation complete: {args.output_dir}")


if __name__ == "__main__":
    main()
