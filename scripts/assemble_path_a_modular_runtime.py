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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-bundle", type=Path, required=True)
    parser.add_argument("--image-bundle", type=Path, required=True)
    parser.add_argument("--text-readout", type=Path)
    parser.add_argument("--text-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.text_readout is None) != (args.text_report is None):
        parser.error("--text-readout and --text-report must be provided together")
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
    manifest = assemble_modular_bundle(
        {"audio": args.audio_bundle, "image": args.image_bundle},
        args.output_dir,
        source_commit=source_commit,
        shared_tasks=shared_tasks,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
