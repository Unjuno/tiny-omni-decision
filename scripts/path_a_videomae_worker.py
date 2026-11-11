"""Persistent JSONL worker for the pinned VideoMAE Clevr-4 image scorer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from tiny_omni_decision.perception_video_runtime import PerceptionVideoTextRuntime
from tiny_omni_decision.videomae_runtime import VideoMAEClevr4ImageRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--perception-video-readout", type=Path)
    parser.add_argument("--text-model-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA worker requested but CUDA is unavailable")
    start = time.perf_counter()
    process = psutil.Process()
    if args.perception_video_readout is not None and args.text_model_dir is None:
        raise ValueError("--text-model-dir is required with --perception-video-readout")
    runtime: VideoMAEClevr4ImageRuntime | None = None
    shared_encoder: torch.nn.Module | None = None
    if args.readout is not None:
        runtime = VideoMAEClevr4ImageRuntime(
            args.model_root,
            args.readout,
            args.dependency_root,
            device=args.device,
        )
        shared_encoder = runtime.encoder
    perception_runtime: PerceptionVideoTextRuntime | None = None
    ready = {
        "ready": True,
        "profile": "path_a_videomaev2_shared_encoder_worker",
        "process_id": __import__("os").getpid(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_load_seconds": time.perf_counter() - start,
        **(runtime.metadata if runtime is not None else {"device": args.device}),
    }
    protocol = sys.stdout
    protocol.write(json.dumps(ready, ensure_ascii=False) + "\n")
    protocol.flush()

    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                if perception_runtime is not None:
                    perception_runtime.close()
                protocol.write('{"stopped":true}\n')
                protocol.flush()
                break
            task = request.get("task")
            if task not in {"image", "video_text"}:
                raise ValueError("VideoMAE worker only accepts image or video_text requests")
            payload = request["payload"]
            started = time.perf_counter()
            if task == "image":
                if runtime is None:
                    raise ValueError("image task requires --readout with the pinned image scorer")
                result = runtime.decide(
                    Path(payload["image_path"]),
                    str(payload["taxonomy"]),
                    [str(value) for value in payload["options"]],
                    expected_media_sha256=str(payload["expected_media_sha256"]),
                )
            else:
                if args.perception_video_readout is None or args.text_model_dir is None:
                    raise ValueError(
                        "video_text requires --perception-video-readout and --text-model-dir"
                    )
                if perception_runtime is None:
                    perception_runtime = PerceptionVideoTextRuntime(
                        model_root=args.model_root,
                        dependency_root=args.dependency_root,
                        text_model_dir=args.text_model_dir,
                        readout_path=args.perception_video_readout,
                        device=args.device,
                        video_encoder=shared_encoder,
                    )
                    shared_encoder = perception_runtime.video_encoder
                result = perception_runtime.decide(
                    video_path=Path(payload["video_path"]),
                    question=str(payload["question"]),
                    options=[str(value) for value in payload["options"]],
                    expected_media_sha256=str(payload["expected_media_sha256"]),
                )
                result["video_text_encoder"] = perception_runtime.metadata
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
                    "cuda_peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated() if args.device == "cuda" else None
                    ),
                    "worker_rss_bytes": process.memory_info().rss,
                    "worker_peak_working_set_bytes": getattr(
                        process.memory_info(), "peak_wset", None
                    ),
                }
            )
            protocol.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            protocol.flush()
        except Exception as error:  # keep the resident worker alive for later requests
            protocol.write(
                json.dumps(
                    {
                        "request_id": request.get("request_id"),
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            protocol.flush()


if __name__ == "__main__":
    main()
