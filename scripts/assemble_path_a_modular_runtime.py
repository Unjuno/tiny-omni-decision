"""Assemble immutable task-scoped Path A runtimes in a new directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tiny_omni_decision.modular_bundle import assemble_modular_bundle
from tiny_omni_decision.text_runtime import (
    MODEL_PARAMETERS as TEXT_MODEL_PARAMETERS,
)
from tiny_omni_decision.text_runtime import (
    MODEL_REVISION as TEXT_MODEL_REVISION,
)
from tiny_omni_decision.text_runtime import (
    MODEL_WEIGHTS_SHA256 as TEXT_MODEL_WEIGHTS_SHA256,
)
from tiny_omni_decision.text_runtime import (
    READOUT_PARAMETERS as TEXT_READOUT_PARAMETERS,
)
from tiny_omni_decision.text_runtime import (
    READOUT_SHA256 as TEXT_READOUT_SHA256,
)
from tiny_omni_decision.video_runtime import (
    READOUT_PARAMETERS as VIDEO_READOUT_PARAMETERS,
)
from tiny_omni_decision.video_runtime import (
    READOUT_SHA256 as VIDEO_READOUT_SHA256,
)
from tiny_omni_decision.video_runtime import (
    VIDEO_MODEL_PARAMETERS,
    VIDEO_MODEL_SHA256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-bundle", type=Path, required=True)
    parser.add_argument("--image-bundle", type=Path, required=True)
    parser.add_argument("--text-readout", type=Path)
    parser.add_argument("--text-report", type=Path)
    parser.add_argument("--video-readout", type=Path)
    parser.add_argument("--video-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.text_readout is None) != (args.text_report is None):
        parser.error("--text-readout and --text-report must be provided together")
    if (args.video_readout is None) != (args.video_report is None):
        parser.error("--video-readout and --video-report must be provided together")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).stdout.strip()
    shared_tasks = {}
    if args.text_readout is not None and args.text_report is not None:
        text_readout = args.text_readout.resolve(strict=True)
        text_report_path = args.text_report.resolve(strict=True)
        text_report = json.loads(text_report_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(text_readout.read_bytes()).hexdigest()
        if digest != TEXT_READOUT_SHA256:
            raise ValueError("text readout hash differs from the pinned runtime")
        if (
            text_report.get("status") != "complete_text_only_synthetic_probe"
            or text_report.get("model_revision") != TEXT_MODEL_REVISION
            or text_report.get("model_weights_sha256") != TEXT_MODEL_WEIGHTS_SHA256
        ):
            raise ValueError("text report model revision or source hash mismatch")
        source_root = Path(__file__).resolve().parents[1]
        shared_tasks["text"] = {
            "task_id": "typed-decisions-synth-text-v0",
            "scope": (
                "synthetic Typed Decisions Synth only; not broad real-world text decision quality"
            ),
            "source_report_sha256": hashlib.sha256(text_report_path.read_bytes()).hexdigest(),
            "runtime_environment": {
                "python": "3.11.9",
                "torch": "2.6.0+cu124",
                "transformers": "5.6.2",
                "device_profile": "CUDA capable; can use shared MiniLM weights",
            },
            "shared_assets": [
                {
                    "name": "minilm-en",
                    "path": "components/audio/models/text-minilm-en",
                    "weights_sha256": TEXT_MODEL_WEIGHTS_SHA256,
                    "parameters": TEXT_MODEL_PARAMETERS,
                    "model_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
                    "revision": TEXT_MODEL_REVISION,
                }
            ],
            "parameters": [
                {
                    "name": "text-readout",
                    "identity": TEXT_READOUT_SHA256,
                    "parameters": TEXT_READOUT_PARAMETERS,
                    "path": "readout/best-readout.pt",
                }
            ],
            "files": {
                "readout/best-readout.pt": text_readout,
                "runtime/tiny_omni_decision/text_runtime.py": source_root
                / "src/tiny_omni_decision/text_runtime.py",
                "runtime/tiny_omni_decision/decision.py": source_root
                / "src/tiny_omni_decision/decision.py",
            },
        }
    if args.video_readout is not None and args.video_report is not None:
        video_readout = args.video_readout.resolve(strict=True)
        video_report_path = args.video_report.resolve(strict=True)
        video_report = json.loads(video_report_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(video_readout.read_bytes()).hexdigest()
        if digest != VIDEO_READOUT_SHA256:
            raise ValueError("video readout hash differs from the pinned runtime")
        if (
            video_report.get("experiment_id")
            != "path-a-videoprism-clevrer-binary-reasoning-v0"
            or video_report.get("artifact_hashes", {}).get("best-readout.pt") != digest
            or video_report.get("effective_config", {}).get("video_encoder_frozen") is not True
        ):
            raise ValueError("video report/readout provenance mismatch")
        source_root = Path(__file__).resolve().parents[1]
        shared_tasks["video"] = {
            "task_id": "clevrer-videoprism-binary-feature-decision-v0",
            "scope": (
                "CLEVRER explanatory/predictive/counterfactual binary decisions from a "
                "precomputed pooled 8-frame VideoPrism feature; encoder execution is a separate "
                "runtime profile and raw-video end-to-end packaging is not provided"
            ),
            "source_report_sha256": hashlib.sha256(video_report_path.read_bytes()).hexdigest(),
            "runtime_environment": {
                "python": "3.11.9",
                "torch": "2.6.0+cu124",
                "transformers": "5.6.2",
                "device_profile": (
                    "CUDA scorer; VideoPrism feature extraction uses a separate JAX/TF profile"
                ),
            },
            "shared_assets": [
                {
                    "name": "videoprism-base-f16r288",
                    "path": "components/image/videoprism-base-f16r288.npz",
                    "weights_sha256": VIDEO_MODEL_SHA256,
                    "parameters": VIDEO_MODEL_PARAMETERS,
                },
                {
                    "name": "minilm-en",
                    "path": "components/audio/models/text-minilm-en",
                    "weights_sha256": TEXT_MODEL_WEIGHTS_SHA256,
                    "parameters": TEXT_MODEL_PARAMETERS,
                    "model_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
                    "revision": TEXT_MODEL_REVISION,
                },
            ],
            "parameters": [
                {
                    "name": "video-readout",
                    "identity": VIDEO_READOUT_SHA256,
                    "parameters": VIDEO_READOUT_PARAMETERS,
                    "path": "readout/best-readout.pt",
                }
            ],
            "files": {
                "readout/best-readout.pt": video_readout,
                "runtime/tiny_omni_decision/video_runtime.py": source_root
                / "src/tiny_omni_decision/video_runtime.py",
                "runtime/tiny_omni_decision/decision.py": source_root
                / "src/tiny_omni_decision/decision.py",
            },
        }
    manifest = assemble_modular_bundle(
        {"audio": args.audio_bundle, "image": args.image_bundle},
        args.output_dir,
        source_commit=source_commit,
        shared_tasks=shared_tasks,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
