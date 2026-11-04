"""Persistent JSONL worker for raw-input, text-free Speech Commands decisions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from tiny_omni_decision.textfree_audio_runtime import TextFreeAudioRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cache-entries", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(4)
    runtime = TextFreeAudioRuntime(
        args.encoder_dir,
        args.readout,
        device=args.device,
        cache_entries=args.cache_entries,
    )
    protocol = sys.stdout
    protocol.write(
        json.dumps(
            {
                "ready": True,
                "profile": "path_a_textfree_audio",
                "device": str(runtime.device),
                "cuda_initialized": torch.cuda.is_initialized(),
                "encoder_revision": "openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "encoder_parameters": 8_208_384,
                "readout_parameters": 3_850,
                "resident_parameters": runtime.parameter_count,
                "decoder_loaded": False,
                "text_encoder_loaded": False,
                "cache_max_entries": args.cache_entries,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    protocol.flush()

    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                protocol.write('{"stopped":true}\n')
                protocol.flush()
                break
            payload = request["payload"]
            started = time.perf_counter()
            decision = runtime.decide(
                Path(payload["audio_path"]),
                payload["options"],
                expected_media_sha256=payload.get("expected_media_sha256"),
            )
            probabilities = [row["probability"] for row in decision["options"]]
            prediction_index = next(
                index for index, row in enumerate(decision["options"])
                if row["option_id"] == decision["selected_option_id"]
            )
            response = {
                "request_id": request.get("request_id"),
                "probabilities": probabilities,
                "prediction_index": prediction_index,
                "confidence": probabilities[prediction_index],
                "decision": decision,
                "encoder_metrics": decision["encoder_metrics"],
                "worker_inference_seconds": time.perf_counter() - started,
            }
            protocol.write(json.dumps(response, ensure_ascii=False) + "\n")
            protocol.flush()
        except Exception as error:  # return request errors without killing the resident worker
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

    runtime.close()


if __name__ == "__main__":
    main()
