"""Compare fixed Path A Audio intermediate tensors across precision/device modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

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
    pair_candidate_features,
)
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher, TypedDecisionSpec

FIXED_AUDIO_SHA256 = "77ad54371520bd5381b7758694bc6d98193e8f8864750ea56ea7f8085f2daaa1"


def _sha256_array(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _capture_mode(
    *,
    mode: str,
    device: str,
    precision: str,
    model_dir: Path,
    readout: Path,
    media: Path,
    queries: list[DecisionQuery],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    backend_settings = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    runtime = SpeechCommandsMultiQueryRuntime(
        model_dir,
        readout,
        device=device,
        audio_encoder_precision=precision,
        audio_cache_entries=0,
    )
    tensors: dict[str, np.ndarray] = {}
    processor = runtime.audio_processor

    class CapturingProcessor:
        n_samples = processor.n_samples

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            encoded = processor(*args, **kwargs)
            tensors["audio_input_features"] = encoded.input_features.detach().cpu().numpy()
            return encoded

    runtime.audio_processor = CapturingProcessor()

    def capture_audio_hidden(_module: Any, _inputs: Any, output: Any) -> None:
        tensors["audio_encoder_hidden"] = output.last_hidden_state.detach().float().cpu().numpy()

    audio_hook = runtime.audio_encoder.register_forward_hook(capture_audio_hidden)
    encoder_backend_settings: dict[str, Any] = {}

    def capture_audio_backend(_module: Any, _inputs: Any) -> None:
        encoder_backend_settings.update(
            {
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            }
        )

    audio_backend_hook = runtime.audio_encoder.register_forward_pre_hook(capture_audio_backend)

    tokenizer = runtime.text_tokenizer

    def capture_tokenizer(*args: Any, **kwargs: Any) -> Any:
        encoded = tokenizer(*args, **kwargs)
        tensors["text_input_ids"] = encoded["input_ids"].detach().cpu().numpy()
        tensors["text_attention_mask"] = encoded["attention_mask"].detach().cpu().numpy()
        return encoded

    runtime.text_tokenizer = capture_tokenizer

    def capture_text_hidden(_module: Any, _inputs: Any, output: Any) -> None:
        tensors["text_encoder_hidden"] = output.last_hidden_state.detach().float().cpu().numpy()

    text_hook = runtime.text_model.register_forward_hook(capture_text_hidden)
    started = time.perf_counter()
    try:
        audio = runtime._encode_audio(media)
        all_texts = runtime._texts_for_queries([queries])
        text_vectors = runtime._encode_texts(all_texts)
        tensors["audio_pooled"] = audio.numpy()
        tensors["text_pooled"] = np.stack([text_vectors[text].numpy() for text in all_texts])
        logits_rows = []
        probability_rows = []
        pair_rows = []
        for query in queries:
            question = text_vectors[f"Question: {query.question}"]
            candidates = torch.stack(
                [text_vectors[f"Candidate answer: {option}."] for option in query.options]
            )
            pairs = pair_candidate_features(audio, question, candidates).to(runtime.device)
            with torch.inference_mode():
                logits = runtime.scorer(pairs).squeeze(-1).float().cpu()
                probabilities = torch.softmax(logits, dim=-1)
            pair_rows.append(pairs.detach().float().cpu().numpy())
            logits_rows.append(logits.numpy())
            probability_rows.append(probabilities.numpy())
        tensors["pair_features"] = np.concatenate(pair_rows, axis=0)
        tensors["option_logits"] = np.concatenate(logits_rows)
        tensors["probabilities"] = np.concatenate(probability_rows)
    finally:
        audio_hook.remove()
        audio_backend_hook.remove()
        text_hook.remove()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    summary = {
        "mode": mode,
        "device": device,
        "requested_precision": precision,
        "effective_precision": runtime.audio_encoder_precision,
        "backend_settings": backend_settings,
        "encoder_backend_settings": encoder_backend_settings,
        "seconds_after_model_load": seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device == "cuda" else None
        ),
        "stages": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _sha256_array(value),
                "finite": bool(np.isfinite(value).all()),
            }
            for key, value in tensors.items()
        },
    }
    del runtime
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary, tensors


def _compare(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "shape_match": True,
        "max_abs_delta": float(delta.max(initial=0.0)),
        "mean_abs_delta": float(delta.mean()) if delta.size else 0.0,
        "exact_match": bool(np.array_equal(left, right)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--dispatcher-assembly", type=Path, required=True)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    assembly = args.assembly.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite precision diagnostic: {output_dir}")
    requests = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    request = next(row for row in requests if row.get("task") == "audio")
    payload = request["payload"]
    if any(key in payload for key in ("target", "label", "target_index")):
        raise ValueError("fixed request payload must not contain targets or labels")
    media = Path(payload["audio_path"]).resolve(strict=True)
    expected_media_hash = FIXED_AUDIO_SHA256
    if _file_sha256(media) != expected_media_hash:
        raise ValueError("fixed audio media hash mismatch")
    queries = [
        DecisionQuery(str(row["question"]), tuple(str(x) for x in row["options"]))
        for row in payload["queries"]
    ]
    model_dir = assembly / "components" / "audio" / "models"
    readout = assembly / "components" / "audio" / "readout" / "best-readout.pt"
    if _file_sha256(model_dir / "audio-whisper-tiny" / "model.safetensors") != AUDIO_WEIGHTS_SHA256:
        raise ValueError("Whisper checkpoint hash mismatch")
    if _file_sha256(model_dir / "text-minilm-en" / "model.safetensors") != TEXT_WEIGHTS_SHA256:
        raise ValueError("MiniLM checkpoint hash mismatch")
    if _file_sha256(readout) != READOUT_SHA256:
        raise ValueError("Audio readout hash mismatch")

    output_dir.mkdir(parents=True)
    modes = [("cpu_fp32", "cpu", "fp32")]
    if torch.cuda.is_available():
        modes.extend(
            [
                ("cuda_fp32", "cuda", "fp32"),
                ("cuda_fp32_no_tf32", "cuda", "fp32_no_tf32"),
            ]
        )
    summaries: dict[str, dict[str, Any]] = {}
    captures: dict[str, dict[str, np.ndarray]] = {}
    for mode, device, precision in modes:
        summary, tensors = _capture_mode(
            mode=mode,
            device=device,
            precision=precision,
            model_dir=model_dir,
            readout=readout,
            media=media,
            queries=queries,
        )
        archive = output_dir / f"{mode}-intermediates.npz"
        np.savez_compressed(archive, **tensors)
        summary["intermediate_archive"] = archive.name
        summary["intermediate_archive_sha256"] = _file_sha256(archive)
        summaries[mode] = summary
        captures[mode] = tensors

    dispatcher_result = None
    if torch.cuda.is_available():
        dispatcher = PathARuntimeDispatcher(
            args.dispatcher_assembly.resolve(strict=True),
            videoprism_python=args.videoprism_python.resolve(strict=True),
            videoprism_source=args.videoprism_source.resolve(strict=True),
            worker_log_dir=output_dir / "dispatcher-worker-logs",
            device="cuda",
            audio_encoder_precision="fp32_no_tf32",
        )
        dispatch_started = time.perf_counter()
        try:
            specs = [
                TypedDecisionSpec(
                    request_id=f"precision-query-{index}",
                    decision_type="multiple_choice",
                    option_ids=tuple(
                        f"option-{option_index}" for option_index in range(len(query.options))
                    ),
                    option_texts=query.options,
                )
                for index, query in enumerate(queries)
            ]
            response = dispatcher.dispatch_typed(
                "audio",
                {
                    "audio_path": str(media),
                    "queries": [
                        {"question": query.question, "options": list(query.options)}
                        for query in queries
                    ],
                    "expected_media_sha256": expected_media_hash,
                },
                decision_specs=specs,
            )
            torch.cuda.synchronize()
            worker = dispatcher.worker_metadata["torch"]
            probabilities = [
                probability
                for decision in response["decisions"]
                for probability in decision["probabilities"]
            ]
            cpu_probabilities = captures["cpu_fp32"]["probabilities"].astype(np.float64)
            probability_delta = np.abs(np.asarray(probabilities) - cpu_probabilities)
            typed_json = response["decision_json"]
            dispatcher_result = {
                "effective_audio_encoder_precision": worker.get("audio_encoder_precision"),
                "dispatch_wall_seconds_including_worker_startup": time.perf_counter()
                - dispatch_started,
                "worker_startup_wall_seconds": response["worker_startup_wall_seconds"],
                "probability_count": len(probabilities),
                "max_abs_probability_delta_vs_cpu_fp32": float(probability_delta.max(initial=0.0)),
                "parity_threshold": 1e-5,
                "parity_passed_for_fixed_request": bool(probability_delta.max(initial=0.0) <= 1e-5),
                "typed_decision_count": len(typed_json),
                "typed_json_passed": len(typed_json) == len(queries),
                "worker_metadata": worker,
            }
        finally:
            dispatcher.close()

    baseline = captures["cpu_fp32"]
    comparisons = {}
    for mode in summaries:
        if mode == "cpu_fp32":
            continue
        comparisons[mode] = {key: _compare(baseline[key], captures[mode][key]) for key in baseline}
    report = {
        "schema_version": 1,
        "status": "complete_intermediate_precision_diagnostic",
        "scope": (
            "one fixed, previously observed development request; no labels, metrics, tuning, "
            "or checkpoint selection"
        ),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "media_sha256": expected_media_hash,
        "readout_sha256": READOUT_SHA256,
        "query_count": len(queries),
        "option_counts": [len(query.options) for query in queries],
        "target_or_label_accessed": False,
        "sealed_audit_accessed": False,
        "training_performed": False,
        "modes": summaries,
        "comparisons_vs_cpu_fp32": comparisons,
        "dispatcher_fp32_no_tf32": dispatcher_result,
    }
    report_path = output_dir / "precision-diagnostic.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(report_path),
                "modes": list(summaries),
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
