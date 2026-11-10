"""Persistent Torch/CUDA worker for Path A Audio, Text and Video scoring."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.audio_runtime import (
    AUDIO_MODEL_REVISION,
    AUDIO_WEIGHTS_SHA256,
    TEXT_MODEL_REVISION,
    TEXT_WEIGHTS_SHA256,
    DecisionQuery,
    SpeechCommandsMultiQueryRuntime,
)
from tiny_omni_decision.audio_runtime import (
    READOUT_SHA256 as AUDIO_READOUT_SHA256,
)
from tiny_omni_decision.speech_commands_m1_runtime import SpeechCommandsM1Runtime
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


def _build_task_runtimes(
    assembly: Path, *, device: str, audio_cache_entries: int
) -> tuple[
    SpeechCommandsMultiQueryRuntime, TypedTextDecisionRuntime, ClevrerBinaryVideoFeatureRuntime
]:
    """Load MiniLM once and inject it into the Torch task runtimes."""
    components = assembly / "components"
    audio_root = components / "audio"
    text_root = components / "text"
    video_root = components / "video"
    model_root = audio_root / "models"
    text_model_dir = model_root / "text-minilm-en"
    shared_tokenizer = AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)
    shared_encoder = AutoModel.from_pretrained(
        text_model_dir, local_files_only=True, dtype=torch.float32
    ).eval().to(device)
    for parameter in shared_encoder.parameters():
        parameter.requires_grad_(False)

    audio = SpeechCommandsMultiQueryRuntime(
        model_root,
        audio_root / "readout/best-readout.pt",
        device=device,
        audio_cache_entries=audio_cache_entries,
        shared_text_encoder=shared_encoder,
        shared_text_tokenizer=shared_tokenizer,
    )
    text = TypedTextDecisionRuntime(
        model_root / "text-minilm-en",
        text_root / "readout/best-readout.pt",
        device=device,
        shared_encoder=shared_encoder,
        shared_tokenizer=shared_tokenizer,
    )
    video = ClevrerBinaryVideoFeatureRuntime(
        model_root / "text-minilm-en",
        video_root / "readout/best-readout.pt",
        device=device,
        shared_encoder=shared_encoder,
        shared_tokenizer=shared_tokenizer,
    )
    return audio, text, video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--audio-cache-entries", type=int, default=64)
    parser.add_argument("--speech-commands-m1-readout", type=Path)
    args = parser.parse_args()
    assembly = args.assembly.resolve(strict=True)
    start = time.perf_counter()
    process = psutil.Process()
    rss_before_load = process.memory_info().rss
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA worker requested but CUDA is unavailable")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with contextlib.redirect_stdout(sys.stderr):
        audio, text, video = _build_task_runtimes(
            assembly, device=args.device, audio_cache_entries=args.audio_cache_entries
        )
        keyword = (
            SpeechCommandsM1Runtime(audio, args.speech_commands_m1_readout)
            if args.speech_commands_m1_readout is not None
            else None
        )
    startup_peak_allocated = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    startup_peak_reserved = torch.cuda.max_memory_reserved() if args.device == "cuda" else None

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    parameter_count = _unique_parameters(
        [
            audio.audio_model,
            audio.text_model,
            audio.scorer,
            text.encoder,
            text.scorer,
            video.scorer,
            *([keyword.readout] if keyword is not None else []),
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
        "shared_minilm_load_count": 1,
        "shared_minilm_instances": 1,
        "audio_encoder_revision": AUDIO_MODEL_REVISION,
        "audio_encoder_weights_sha256": AUDIO_WEIGHTS_SHA256,
        "audio_readout_sha256": AUDIO_READOUT_SHA256,
        "speech_commands_m1_readout_sha256": (
            keyword.readout_sha256 if keyword is not None else None
        ),
        "speech_commands_m1_readout_parameters": (
            keyword.parameter_count if keyword is not None else None
        ),
        "text_encoder_revision": TEXT_MODEL_REVISION,
        "text_encoder_weights_sha256": TEXT_WEIGHTS_SHA256,
        "text_probe_model_revision": TEXT_PROBE_MODEL_REVISION,
        "text_probe_model_weights_sha256": TEXT_PROBE_MODEL_SHA256,
        "text_readout_sha256": TEXT_READOUT_SHA256,
        "video_readout_sha256": VIDEO_READOUT_SHA256,
        "rss_before_load_bytes": rss_before_load,
        "rss_after_load_bytes": process.memory_info().rss,
        "peak_working_set_after_load_bytes": getattr(process.memory_info(), "peak_wset", None),
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
                        Path(payload["audio_path"]),
                        queries,
                        expected_media_sha256=payload.get("expected_media_sha256"),
                    )
                    result = {
                        "decisions": [_serialize_decision(item) for item in decisions],
                        "encoder_metrics": dict(audio.audio_cache_metrics),
                    }
                elif task == "audio_keyword":
                    if keyword is None:
                        raise ValueError("worker was not started with --speech-commands-m1-readout")
                    decision, keyword_metrics = keyword.decide(
                        Path(payload["audio_path"]),
                        expected_media_sha256=str(payload["expected_media_sha256"]),
                    )
                    result = {
                        **_serialize_decision(decision),
                        "keyword_metrics": keyword_metrics,
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
                    "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
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
