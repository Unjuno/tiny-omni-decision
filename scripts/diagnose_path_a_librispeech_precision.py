"""Compare frozen Audio runtime probabilities on the fixed LibriSpeech inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import transformers

from tiny_omni_decision.audio_runtime import (
    AUDIO_WEIGHTS_SHA256,
    READOUT_SHA256,
    TEXT_WEIGHTS_SHA256,
    DecisionQuery,
    SpeechCommandsMultiQueryRuntime,
    _file_sha256,
)
from tiny_omni_decision.bundle_validation import verify_bundle

EXAMPLE_MANIFEST_SHA256 = "1f17d86a15faecc3ba3e13cd406dfd0119ef17850d1a7206956a70f4eb801923"
BUNDLE_MANIFEST_SHA256 = "574640d88d6ddbbdb2d12f2466a248ce13c8d9eb3ac542d099b2f08981b690f4"
ORDERED_ID_SHA256 = "709687d926f46907a3c5d447d8921b51cb8f1188163f25fd777b11dfae34384c"
CONDITIONS = ("random_distractors", "hard_distractors")
QUESTION = "Which exact transcript matches the spoken audio?"
PARITY_THRESHOLD = 1e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_fields(source: dict[str, Any]) -> dict[str, Any]:
    """Project the fixed artifact onto inference inputs; ignore all labels."""
    options_by_condition = source["options_by_condition"]
    if set(options_by_condition) != set(CONDITIONS):
        raise ValueError("unexpected LibriSpeech distractor conditions")
    projected_options = {
        condition: [str(option) for option in options_by_condition[condition]]
        for condition in CONDITIONS
    }
    if any(len(options) != 4 or len(set(options)) != 4 for options in projected_options.values()):
        raise ValueError("each condition must contain four unique options")
    return {
        "sample_id": str(source["id"]),
        "audio_path": str(source["audio_path"]),
        "audio_sha256": str(source["audio_sha256"]),
        "duration_seconds": float(source["duration_seconds"]),
        "options_by_condition": projected_options,
    }


def _load_requests(
    manifest_path: Path, source_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != EXAMPLE_MANIFEST_SHA256:
        raise ValueError(f"fixed example manifest hash mismatch: {manifest_hash}")
    rows = [
        _request_fields(json.loads(line))
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["sample_id"] for row in rows]
    if len(rows) != 1938 or len(set(ids)) != len(ids):
        raise ValueError("expected 1,938 unique fixed LibriSpeech inputs")
    order_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if order_hash != ORDERED_ID_SHA256:
        raise ValueError(f"fixed LibriSpeech ID order mismatch: {order_hash}")

    duration_buckets = {"<=5s": 0, "<=10s": 0, "<=20s": 0, ">20s": 0}
    media_hashes: set[str] = set()
    for row in rows:
        audio_path = (source_root / row["audio_path"]).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        actual_hash = _file_sha256(audio_path)
        if actual_hash != row["audio_sha256"]:
            raise ValueError(f"fixed audio hash mismatch for {row['sample_id']}")
        with av.open(str(audio_path)) as container:
            stream = next(stream for stream in container.streams if stream.type == "audio")
            if stream.sample_rate != 16_000 or stream.channels != 1:
                raise ValueError(f"unexpected LibriSpeech format for {row['sample_id']}")
            duration = float(stream.duration * stream.time_base)
        if abs(duration - row["duration_seconds"]) > 1e-6:
            raise ValueError(f"duration mismatch for {row['sample_id']}")
        row["audio_path"] = audio_path
        row["duration_seconds"] = duration
        media_hashes.add(actual_hash)
        bucket = (
            "<=5s"
            if duration <= 5
            else "<=10s"
            if duration <= 10
            else "<=20s"
            if duration <= 20
            else ">20s"
        )
        duration_buckets[bucket] += 1
    if len(media_hashes) != len(rows):
        raise ValueError("fixed LibriSpeech audio inputs are not byte-unique")
    metadata = {
        "input_count": len(rows),
        "unique_audio_assets": len(media_hashes),
        "manifest_sha256": manifest_hash,
        "ordered_sample_id_sha256": order_hash,
        "duration_buckets": duration_buckets,
        "duration_min_seconds": min(row["duration_seconds"] for row in rows),
        "duration_max_seconds": max(row["duration_seconds"] for row in rows),
        "label_fields_used": False,
    }
    return rows, metadata


def _write_pcm16_wav(source: Path, destination: Path) -> None:
    with av.open(str(source)) as container:
        stream = next(stream for stream in container.streams if stream.type == "audio")
        if stream.sample_rate != 16_000 or stream.channels != 1:
            raise ValueError(f"expected mono 16 kHz source: {source}")
        chunks = [frame.to_ndarray().reshape(-1) for frame in container.decode(stream)]
    samples = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)
    if samples.dtype != np.int16 or samples.ndim != 1 or samples.size == 0:
        raise ValueError(f"expected nonempty signed-16-bit source: {source}")
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())


def _run_mode(
    rows: list[dict[str, Any]],
    *,
    model_dir: Path,
    readout_path: Path,
    device: str,
    precision: str,
    output_path: Path,
    temp_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    load_started = time.perf_counter()
    runtime = SpeechCommandsMultiQueryRuntime(
        model_dir,
        readout_path,
        device=device,
        audio_encoder_precision=precision,
        audio_cache_entries=0,
    )
    load_seconds = time.perf_counter() - load_started
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prediction_rows: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        wav_paths = []
        for slot, row in enumerate(batch):
            wav_path = temp_root / f"{offset}-{slot}.wav"
            _write_pcm16_wav(row["audio_path"], wav_path)
            wav_paths.append(wav_path)
        queries = [
            [
                DecisionQuery(QUESTION, tuple(row["options_by_condition"][condition]))
                for condition in CONDITIONS
            ]
            for row in batch
        ]
        results = runtime.decide_batch(wav_paths, queries)
        for row, condition_results in zip(batch, results, strict=True):
            if len(condition_results) != len(CONDITIONS):
                raise ValueError("runtime did not return both fixed conditions")
            for condition, result in zip(CONDITIONS, condition_results, strict=True):
                prediction_rows[condition].append(
                    {
                        "sample_id": row["sample_id"],
                        "audio_sha256": row["audio_sha256"],
                        "condition": condition,
                        "probabilities": result.probabilities.float().tolist(),
                    }
                )
        for wav_path in wav_paths:
            wav_path.unlink()
    if device == "cuda":
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_allocated = peak_reserved = None
    inference_seconds = time.perf_counter() - started

    hashes = {}
    for condition in CONDITIONS:
        path = output_path.with_name(f"{output_path.stem}-{condition}.jsonl")
        path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in prediction_rows[condition]),
            encoding="utf-8",
        )
        hashes[condition] = _sha256(path)
    del runtime
    return {
        "device": device,
        "precision": precision,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "prediction_files_sha256": hashes,
        "prediction_rows": {condition: len(items) for condition, items in prediction_rows.items()},
        "prediction_data": prediction_rows,
    }


def _compare(cpu: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        cpu_rows = cpu["prediction_data"][condition]
        cuda_rows = cuda["prediction_data"][condition]
        if len(cpu_rows) != len(cuda_rows):
            raise ValueError("CPU/CUDA prediction row counts differ")
        deltas: list[float] = []
        decision_maxima: list[float] = []
        changed = 0
        for left, right in zip(cpu_rows, cuda_rows, strict=True):
            if (
                left["sample_id"] != right["sample_id"]
                or left["audio_sha256"] != right["audio_sha256"]
            ):
                raise ValueError("CPU/CUDA request order or audio identity differs")
            a = np.asarray(left["probabilities"], dtype=np.float64)
            b = np.asarray(right["probabilities"], dtype=np.float64)
            if a.shape != (4,) or b.shape != (4,):
                raise ValueError("expected four option probabilities")
            row_delta = np.abs(a - b)
            deltas.extend(row_delta.tolist())
            decision_maxima.append(float(row_delta.max()))
            changed += int(a.argmax() != b.argmax())
        ordered = sorted(decision_maxima)
        result[condition] = {
            "decision_count": len(decision_maxima),
            "probability_count": len(deltas),
            "max_abs_probability_delta": max(deltas),
            "mean_abs_probability_delta": statistics.fmean(deltas),
            "p50_decision_max_delta": ordered[len(ordered) // 2],
            "p95_decision_max_delta": ordered[int(0.95 * (len(ordered) - 1))],
            "p99_decision_max_delta": ordered[int(0.99 * (len(ordered) - 1))],
            "decisions_over_1e-5": sum(value > PARITY_THRESHOLD for value in decision_maxima),
            "argmax_changed_decisions": changed,
            "strict_probability_parity_passed": max(deltas) <= PARITY_THRESHOLD,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("batch size 8 is fixed for the saved feature reference")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output_dir}")
    rows, request_metadata = _load_requests(args.example_manifest, args.source_root)
    manifest_path = args.bundle_dir / "bundle-manifest.json"
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != BUNDLE_MANIFEST_SHA256:
        raise ValueError(f"fixed runtime bundle hash mismatch: {manifest_hash}")
    bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_bundle(args.bundle_dir, bundle_manifest)
    if args.preflight_only:
        print(
            json.dumps(
                {"request_set": request_metadata, "bundle_manifest_sha256": manifest_hash}, indent=2
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed CPU/CUDA parity comparison")

    args.output_dir.mkdir(parents=True)
    temp_root = args.output_dir / "temporary-wav"
    temp_root.mkdir()
    cpu = _run_mode(
        rows,
        model_dir=args.bundle_dir / "models",
        readout_path=args.bundle_dir / "readout" / "best-readout.pt",
        device="cpu",
        precision="fp32",
        output_path=args.output_dir / "cpu.jsonl",
        temp_root=temp_root,
        batch_size=args.batch_size,
    )
    del cpu["prediction_data"]
    if temp_root.exists():
        remaining = list(temp_root.iterdir())
        if remaining:
            raise RuntimeError(f"temporary WAV cleanup incomplete: {len(remaining)} files")
        temp_root.rmdir()
    temp_root.mkdir()
    cuda = _run_mode(
        rows,
        model_dir=args.bundle_dir / "models",
        readout_path=args.bundle_dir / "readout" / "best-readout.pt",
        device="cuda",
        precision="fp32_no_tf32",
        output_path=args.output_dir / "cuda.jsonl",
        temp_root=temp_root,
        batch_size=args.batch_size,
    )
    del cuda["prediction_data"]
    temp_root.rmdir()
    comparisons = _compare(
        {
            "prediction_data": {
                condition: [
                    json.loads(line)
                    for line in (args.output_dir / f"cpu-{condition}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for condition in CONDITIONS
            }
        },
        {
            "prediction_data": {
                condition: [
                    json.loads(line)
                    for line in (args.output_dir / f"cuda-{condition}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for condition in CONDITIONS
            }
        },
    )
    report = {
        "status": "complete_librispeech_audio_numeric_parity_only",
        "scope": (
            "fixed previously observed dev-clean inputs; no labels, quality metrics, "
            "training, or checkpoint selection"
        ),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "request_set": request_metadata,
        "bundle_manifest_sha256": manifest_hash,
        "audio_weights_sha256": AUDIO_WEIGHTS_SHA256,
        "text_weights_sha256": TEXT_WEIGHTS_SHA256,
        "readout_sha256": READOUT_SHA256,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cpu_fp32": cpu,
        "cuda_fp32_no_tf32": cuda,
        "comparison": comparisons,
        "labels_or_targets_used": False,
        "sealed_audit_accessed": False,
        "training_performed": False,
    }
    report_path = args.output_dir / "precision-parity.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "comparison": comparisons}, indent=2))


if __name__ == "__main__":
    main()
