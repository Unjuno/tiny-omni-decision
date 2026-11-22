"""Persistent Torch/CUDA worker for Path A Audio, Text and Video scoring."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from tiny_omni_decision.audio_runtime import (
    AUDIO_MODEL_REVISION,
    AUDIO_WEIGHTS_SHA256,
    PAIR_FEATURE_SIZE,
    READOUT_HIDDEN_SIZE,
    TEXT_MODEL_REVISION,
    TEXT_WEIGHTS_SHA256,
    DecisionQuery,
    SpeechCommandsMultiQueryRuntime,
)
from tiny_omni_decision.audio_runtime import (
    READOUT_SHA256 as AUDIO_READOUT_SHA256,
)
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.text_runtime import (
    MODEL_REVISION as TEXT_PROBE_MODEL_REVISION,
)
from tiny_omni_decision.text_runtime import (
    MODEL_WEIGHTS_SHA256 as TEXT_PROBE_MODEL_SHA256,
)
from tiny_omni_decision.text_runtime import (
    READOUT_SHA256 as TEXT_READOUT_SHA256,
)
from tiny_omni_decision.text_runtime import (
    TypedTextDecisionRuntime,
)
from tiny_omni_decision.video_runtime import (
    READOUT_SHA256 as VIDEO_READOUT_SHA256,
)
from tiny_omni_decision.video_runtime import (
    ClevrerBinaryVideoFeatureRuntime,
)


def _serialize_decision(result: Any) -> dict[str, Any]:
    return {
        "logits": result.option_logits.tolist(),
        "probabilities": result.probabilities.tolist(),
        "prediction_index": result.predicted_index,
        "confidence": result.confidence,
    }


def _unique_parameters(modules: list[torch.nn.Module]) -> int:
    values: dict[int, int] = {}
    for module in modules:
        for parameter in module.parameters():
            values[id(parameter)] = parameter.numel()
    return sum(values.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--audio-encoder-precision",
        choices=("auto", "bf16", "fp32", "fp32_no_tf32"),
        default="auto",
    )
    parser.add_argument("--audio-cache-entries", type=int, default=64)
    args = parser.parse_args()
    assembly = args.assembly.resolve(strict=True)
    components = assembly / "components"
    audio_root = components / "audio"
    text_root = components / "text"
    video_root = components / "video"
    model_root = audio_root / "models"
    start = time.perf_counter()
    process = psutil.Process()
    rss_before_load = process.memory_info().rss
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA worker requested but CUDA is unavailable")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with contextlib.redirect_stdout(sys.stderr):
        audio = SpeechCommandsMultiQueryRuntime(
            model_root,
            audio_root / "readout/best-readout.pt",
            device=args.device,
            audio_encoder_precision=args.audio_encoder_precision,
            audio_cache_entries=args.audio_cache_entries,
        )
        text = TypedTextDecisionRuntime(
            model_root / "text-minilm-en",
            text_root / "readout/best-readout.pt",
            device=args.device,
        )
        video = ClevrerBinaryVideoFeatureRuntime(
            model_root / "text-minilm-en",
            video_root / "readout/best-readout.pt",
            device=args.device,
        )
        libri_task_root = components / "librispeech"
        libri_scorer = None
        libri_readout_sha256 = None
        task_manifest_path = libri_task_root / "task-manifest.json"
        if task_manifest_path.is_file():
            task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
            if task_manifest.get("task_id") != "librispeech-transcript-choice-v1":
                raise ValueError("unsupported LibriSpeech task identity")
            readout_entry = next(
                (
                    item
                    for item in task_manifest.get("parameters", [])
                    if item.get("name") == "librispeech-readout"
                ),
                None,
            )
            if readout_entry is None or int(readout_entry.get("parameters", -1)) != 344_321:
                raise ValueError("LibriSpeech readout metadata is missing or unexpected")
            libri_readout_sha256 = str(readout_entry["identity"])
            readout_path = libri_task_root / "readout/best-readout.pt"
            if hashlib.sha256(readout_path.read_bytes()).hexdigest() != libri_readout_sha256:
                raise ValueError("LibriSpeech readout hash differs from task manifest")
            libri_scorer = FrozenFeatureCandidateScorer(
                feature_size=PAIR_FEATURE_SIZE, hidden_size=READOUT_HIDDEN_SIZE
            ).to(args.device)
            state = torch.load(readout_path, map_location="cpu", weights_only=True)
            libri_scorer.load_state_dict(state, strict=True)
            libri_scorer.eval()
            for parameter in libri_scorer.parameters():
                parameter.requires_grad_(False)
    startup_peak_allocated = (
        torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    )
    startup_peak_reserved = torch.cuda.max_memory_reserved() if args.device == "cuda" else None

    # These task runtimes use the identical pinned MiniLM files. Share one
    # resident module after their own hash-checked initializations.
    shared_encoder = text.encoder
    shared_tokenizer = text.tokenizer
    audio.text_model = shared_encoder
    audio.text_tokenizer = shared_tokenizer
    video.encoder = shared_encoder
    video.tokenizer = shared_tokenizer
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    parameter_count = _unique_parameters(
        [
            audio.audio_model,
            audio.text_model,
            audio.scorer,
            text.encoder,
            text.scorer,
            video.scorer,
            *([libri_scorer] if libri_scorer is not None else []),
        ]
    )
    ready = {
        "ready": True,
        "profile": "torch_audio_text_video_scorer",
        "process_id": os.getpid(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "resident_unique_parameters": parameter_count,
        "audio_cache_max_entries": args.audio_cache_entries,
        "audio_encoder_precision": audio.audio_encoder_precision,
        "shared_minilm_instances_after_consolidation": 1,
        "audio_encoder_revision": AUDIO_MODEL_REVISION,
        "audio_encoder_weights_sha256": AUDIO_WEIGHTS_SHA256,
        "audio_readout_sha256": AUDIO_READOUT_SHA256,
        "librispeech_readout_sha256": libri_readout_sha256,
        "librispeech_readout_loaded": libri_scorer is not None,
        "text_encoder_revision": TEXT_MODEL_REVISION,
        "text_encoder_weights_sha256": TEXT_WEIGHTS_SHA256,
        "text_probe_model_revision": TEXT_PROBE_MODEL_REVISION,
        "text_probe_model_weights_sha256": TEXT_PROBE_MODEL_SHA256,
        "text_readout_sha256": TEXT_READOUT_SHA256,
        "video_readout_sha256": VIDEO_READOUT_SHA256,
        "rss_before_load_bytes": rss_before_load,
        "rss_after_load_bytes": process.memory_info().rss,
        "peak_working_set_after_load_bytes": getattr(
            process.memory_info(), "peak_wset", None
        ),
        "startup_peak_cuda_allocated_bytes": startup_peak_allocated,
        "startup_peak_cuda_reserved_bytes": startup_peak_reserved,
        "cuda_allocated_bytes_after_load": (
            torch.cuda.memory_allocated() if args.device == "cuda" else None
        ),
        "cuda_reserved_bytes_after_load": (
            torch.cuda.memory_reserved() if args.device == "cuda" else None
        ),
        "model_load_seconds": time.perf_counter() - start,
    }
    protocol = sys.stdout
    protocol.write(json.dumps(ready, ensure_ascii=False) + "\n")
    protocol.flush()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                protocol.write('{"stopped":true}\n')
                protocol.flush()
                break
            task = str(request["task"])
            payload = request["payload"]
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                if task == "text":
                    decision = text.decide(
                        str(payload["state"]),
                        str(payload["question"]),
                        [str(item) for item in payload["options"]],
                    )
                    result: dict[str, Any] = _serialize_decision(decision)
                elif task == "audio":
                    queries = [
                        DecisionQuery(
                            question=str(item["question"]),
                            options=tuple(str(option) for option in item["options"]),
                        )
                        for item in payload["queries"]
                    ]
                    decisions = audio.decide_many(
                        Path(payload["audio_path"]), queries,
                        expected_media_sha256=payload.get("expected_media_sha256"),
                    )
                    result = {
                        "decisions": [_serialize_decision(item) for item in decisions],
                        "encoder_metrics": dict(audio.audio_cache_metrics),
                    }
                elif task == "librispeech":
                    if libri_scorer is None:
                        raise ValueError("this assembly has no LibriSpeech task readout")
                    queries = [
                        DecisionQuery(
                            question=str(item["question"]),
                            options=tuple(str(option) for option in item["options"]),
                        )
                        for item in payload["queries"]
                    ]
                    decisions = audio.decide_many_with_readout(
                        Path(payload["audio_path"]),
                        queries,
                        libri_scorer,
                        expected_media_sha256=payload.get("expected_media_sha256"),
                    )
                    result = {
                        "decisions": [_serialize_decision(item) for item in decisions],
                        "encoder_metrics": dict(audio.audio_cache_metrics),
                        "readout_sha256": libri_readout_sha256,
                    }
                elif task == "video_score":
                    feature = torch.tensor(payload["video_feature"], dtype=torch.float32)
                    decision = video.decide(
                        feature,
                        str(payload["question"]),
                        str(payload["candidate_statement"]),
                    )
                    result = _serialize_decision(decision)
                else:
                    raise ValueError(f"unsupported Torch task: {task}")
            if args.device == "cuda":
                torch.cuda.synchronize()
            result.update(
                {
                    "request_id": request.get("request_id"),
                    "task": task,
                    "worker_inference_seconds": time.perf_counter() - started,
                    "cuda_allocated_bytes_after_request": (
                        torch.cuda.memory_allocated() if args.device == "cuda" else None
                    ),
                    "cuda_peak_allocated_bytes_since_consolidation": (
                        torch.cuda.max_memory_allocated() if args.device == "cuda" else None
                    ),
                    "rss_after_request_bytes": process.memory_info().rss,
                    "peak_working_set_bytes": getattr(
                        process.memory_info(), "peak_wset", None
                    ),
                }
            )
            protocol.write(json.dumps(result, ensure_ascii=False) + "\n")
            protocol.flush()
        except Exception as error:  # return request errors without killing a resident worker
            protocol.write(
                json.dumps(
                    {
                        "request_id": request.get("request_id") if "request" in locals() else None,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            protocol.flush()


if __name__ == "__main__":
    main()
