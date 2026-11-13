"""Persistent JSONL worker for the pinned VideoMAE Clevr-4 image scorer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from tiny_omni_decision.videomae_runtime import VideoMAEClevr4ImageRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA worker requested but CUDA is unavailable")
    start = time.perf_counter()
    runtime = VideoMAEClevr4ImageRuntime(
        args.model_root,
        args.readout,
        args.dependency_root,
        device=args.device,
    )
    ready = {
        "ready": True,
        "profile": "path_a_videomaev2_clevr4_image",
        "process_id": __import__("os").getpid(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_load_seconds": time.perf_counter() - start,
        **runtime.metadata,
    }
    protocol = sys.stdout
    protocol.write(json.dumps(ready, ensure_ascii=False) + "\n")
    protocol.flush()

    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                protocol.write('{"stopped":true}\n')
                protocol.flush()
                break
            if request.get("task") != "image":
                raise ValueError("VideoMAE worker only accepts image requests")
            payload = request["payload"]
            started = time.perf_counter()
            result = runtime.decide(
                Path(payload["image_path"]),
                str(payload["taxonomy"]),
                [str(value) for value in payload["options"]],
                expected_media_sha256=str(payload["expected_media_sha256"]),
            )
            if args.device == "cuda":
                torch.cuda.synchronize()
            result.update(
                {
                    "request_id": request.get("request_id"),
                    "task": "image",
                    "worker_inference_seconds": time.perf_counter() - started,
                    "cuda_allocated_bytes_after_request": (
                        torch.cuda.memory_allocated() if args.device == "cuda" else None
                    ),
                    "cuda_peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated() if args.device == "cuda" else None
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
