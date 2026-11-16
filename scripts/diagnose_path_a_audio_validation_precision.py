"""Measure label-free CPU/CUDA probability parity on fixed Audio requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import wave
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_requests(
    reference_path: Path, sample_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    reference_bytes = reference_path.read_bytes()
    for line in reference_bytes.splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        sample_id = str(source["sample_id"])
        # Deliberately select inference inputs and ignore all target/label fields.
        grouped.setdefault(sample_id, []).append(
            {
                "question": str(source["question"]),
                "options": [str(option) for option in source["options"]],
                "question_type": str(source["question_type"]),
            }
        )

    if len(grouped) != 640 or sum(map(len, grouped.values())) != 2560:
        raise ValueError("expected the frozen 640-clip, 2,560-query reference set")
    if any(len(queries) != 4 for queries in grouped.values()):
        raise ValueError("every fixed audio event must have exactly four queries")

    requests = []
    question_types: Counter[str] = Counter()
    duration_counts: Counter[str] = Counter()
    media_hashes: set[str] = set()
    for sample_id, items in grouped.items():
        audio_path = sample_dir / "media" / "validation" / f"{sample_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        media_sha256 = _file_sha256(audio_path)
        media_hashes.add(media_sha256)
        with wave.open(str(audio_path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                raise ValueError(f"unexpected fixed audio format: {audio_path}")
            duration = wav.getnframes() / wav.getframerate()
        duration_counts["<=1s" if duration <= 1 else "<=3s" if duration <= 3 else ">3s"] += 1
        queries = [DecisionQuery(item["question"], tuple(item["options"])) for item in items]
        question_types.update(item["question_type"] for item in items)
        requests.append(
            {
                "sample_id": sample_id,
                "audio_path": audio_path,
                "audio_sha256": media_sha256,
                "duration_seconds": duration,
                "items": items,
                "queries": queries,
            }
        )

    if len(media_hashes) != 640:
        raise ValueError("fixed audio assets are not byte-unique")
    metadata = {
        "reference_path": str(reference_path),
        "reference_sha256": _sha256_bytes(reference_bytes),
        "sample_id_order_sha256": _sha256_bytes("\n".join(grouped).encode()),
        "unique_audio_assets": len(media_hashes),
        "question_type_counts": dict(sorted(question_types.items())),
        "audio_duration_buckets": dict(sorted(duration_counts.items())),
        "audio_duration_min_seconds": min(item["duration_seconds"] for item in requests),
        "audio_duration_max_seconds": max(item["duration_seconds"] for item in requests),
        "labels_used": False,
    }
    return requests, metadata


def _run_mode(
    *,
    requests: list[dict[str, Any]],
    model_dir: Path,
    readout_path: Path,
    device: str,
    precision: str,
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = SpeechCommandsMultiQueryRuntime(
        model_dir,
        readout_path,
        device=device,
        audio_encoder_precision=precision,
        audio_cache_entries=0,
    )
    load_seconds = time.perf_counter() - started
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before_flags = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    inference_started = time.perf_counter()
    rows_written = 0
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for offset in range(0, len(requests), batch_size):
            batch = requests[offset : offset + batch_size]
            outputs = runtime.decide_batch(
                [item["audio_path"] for item in batch],
                [item["queries"] for item in batch],
            )
            for item, decisions in zip(batch, outputs, strict=True):
                for index, (query, decision) in enumerate(
                    zip(item["items"], decisions, strict=True)
                ):
                    row = {
                        "sample_id": item["sample_id"],
                        "audio_sha256": item["audio_sha256"],
                        "question_index": index,
                        "question_type": query["question_type"],
                        "probabilities": decision.probabilities.tolist(),
                    }
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    rows_written += 1
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    after_flags = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if before_flags != after_flags:
        raise RuntimeError("Torch TF32 backend flags were not restored")
    result = {
        "device": device,
        "requested_precision": precision,
        "effective_precision": runtime.audio_encoder_precision,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "rows_written": rows_written,
        "prediction_path": str(output_path),
        "prediction_sha256": _sha256_bytes(output_path.read_bytes()),
        "backend_flags_before": before_flags,
        "backend_flags_after": after_flags,
        "runtime_parameter_count": runtime.parameter_count,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if device == "cuda"
        else None,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())
        if device == "cuda"
        else None,
    }
    del runtime
    return result


def _compare(cpu_path: Path, cuda_path: Path) -> dict[str, Any]:
    differences = []
    argmax_changes = 0
    count = 0
    for cpu_line, cuda_line in zip(
        cpu_path.read_text(encoding="utf-8").splitlines(),
        cuda_path.read_text(encoding="utf-8").splitlines(),
        strict=True,
    ):
        cpu = json.loads(cpu_line)
        cuda = json.loads(cuda_line)
        if (cpu["sample_id"], cpu["question_index"], cpu["audio_sha256"]) != (
            cuda["sample_id"],
            cuda["question_index"],
            cuda["audio_sha256"],
        ):
            raise ValueError("CPU/CUDA request order or audio identity differs")
        left = torch.tensor(cpu["probabilities"], dtype=torch.float64)
        right = torch.tensor(cuda["probabilities"], dtype=torch.float64)
        if left.shape != right.shape:
            raise ValueError("CPU/CUDA option counts differ")
        delta = (left - right).abs()
        differences.extend(delta.tolist())
        argmax_changes += int(left.argmax().item() != right.argmax().item())
        count += 1
    if count != 2560:
        raise ValueError("incomplete CPU/CUDA prediction comparison")
    values = torch.tensor(differences, dtype=torch.float64)
    return {
        "decision_count": count,
        "probability_count": len(differences),
        "max_abs_probability_delta": float(values.max()),
        "mean_abs_probability_delta": float(values.mean()),
        "p50_abs_probability_delta": float(torch.quantile(values, 0.50)),
        "p95_abs_probability_delta": float(torch.quantile(values, 0.95)),
        "p99_abs_probability_delta": float(torch.quantile(values, 0.99)),
        "argmax_changed_decisions": argmax_changes,
        "threshold": 1e-5,
        "strict_probability_parity_passed": bool(values.max() <= 1e-5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("fixed parity protocol requires batch size 8")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    model_dir = args.bundle_dir / "models"
    readout_path = args.bundle_dir / "readout" / "best-readout.pt"
    requests, request_metadata = _load_requests(args.reference_jsonl, args.sample_dir)
    if _file_sha256(model_dir / "audio-whisper-tiny" / "model.safetensors") != AUDIO_WEIGHTS_SHA256:
        raise ValueError("pinned Whisper weight hash mismatch")
    if _file_sha256(model_dir / "text-minilm-en" / "model.safetensors") != TEXT_WEIGHTS_SHA256:
        raise ValueError("pinned MiniLM weight hash mismatch")
    if _file_sha256(readout_path) != READOUT_SHA256:
        raise ValueError("pinned readout weight hash mismatch")
    if args.preflight_only:
        print(json.dumps(request_metadata, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this cross-device diagnostic")

    args.output_dir.mkdir(parents=True)
    cpu = _run_mode(
        requests=requests,
        model_dir=model_dir,
        readout_path=readout_path,
        device="cpu",
        precision="fp32",
        output_path=args.output_dir / "cpu-fp32-predictions.jsonl",
        batch_size=args.batch_size,
    )
    cuda = _run_mode(
        requests=requests,
        model_dir=model_dir,
        readout_path=readout_path,
        device="cuda",
        precision="fp32_no_tf32",
        output_path=args.output_dir / "cuda-fp32-no-tf32-predictions.jsonl",
        batch_size=args.batch_size,
    )
    result = {
        "schema_version": 1,
        "status": "complete_audio_runtime_numeric_parity_diagnostic",
        "scope": (
            "previously observed development inputs; numeric runtime parity only; "
            "no quality metrics or checkpoint selection"
        ),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "audio_weights_sha256": AUDIO_WEIGHTS_SHA256,
        "text_weights_sha256": TEXT_WEIGHTS_SHA256,
        "readout_sha256": READOUT_SHA256,
        "requests": request_metadata,
        "cpu_fp32": cpu,
        "cuda_fp32_no_tf32": cuda,
        "comparison": _compare(
            args.output_dir / "cpu-fp32-predictions.jsonl",
            args.output_dir / "cuda-fp32-no-tf32-predictions.jsonl",
        ),
        "training_performed": False,
        "target_fields_used_or_copied": False,
        "sealed_audit_accessed": False,
    }
    result_path = args.output_dir / "parity-diagnostic.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), **result["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
