"""Build an immutable runtime bundle for the selected LibriSpeech choice readout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from tiny_omni_decision.audio_runtime import (
    AUDIO_MODEL_ID,
    AUDIO_MODEL_REVISION,
    AUDIO_WEIGHTS_SHA256,
    TEXT_MODEL_ID,
    TEXT_MODEL_REVISION,
    TEXT_WEIGHTS_SHA256,
)
from tiny_omni_decision.bundle_validation import verify_bundle


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> tuple[list[dict[str, Any]], int]:
    entries = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": sha256_file(path)})
        total += size
    return entries, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_bundle.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime bundle: {output}")

    source_manifest_path = source / "bundle-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    verify_bundle(source, source_manifest)
    if (
        source_manifest.get("models", {}).get("audio", {}).get("weights_sha256")
        != AUDIO_WEIGHTS_SHA256
    ):
        raise ValueError("source bundle Whisper hash does not match the pinned runtime")
    if (
        source_manifest.get("models", {}).get("text", {}).get("weights_sha256")
        != TEXT_WEIGHTS_SHA256
    ):
        raise ValueError("source bundle MiniLM hash does not match the pinned runtime")

    run_report_path = run_dir / "run-report.json"
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    if (
        run_report.get("status")
        != "complete_librispeech_train_clean_100_frozen_readout_development_experiment"
    ):
        raise ValueError("source run is not the completed LibriSpeech long-curve experiment")
    if run_report.get("sealed_audit_loaded") or run_report.get("test_split_loaded"):
        raise ValueError("source run unexpectedly loaded sealed audit or test data")
    training = run_report.get("training", {})
    selected_epoch = int(training.get("selected_epoch", 0))
    if selected_epoch < 1 or int(training.get("epochs_run", 0)) != 100:
        raise ValueError("source run does not contain the expected completed 100-epoch selection")
    checkpoint = run_dir / "best-readout.pt"
    expected_readout_hash = run_report["artifacts"]["best-readout.pt"]["sha256"]
    if sha256_file(checkpoint) != expected_readout_hash:
        raise ValueError("selected readout does not match the run report")
    construction = run_report["split_and_construction"]
    split = construction["split"]
    if any(
        split.get(key) != 0
        for key in (
            "train_validation_speaker_overlap",
            "train_validation_audio_asset_overlap",
            "train_validation_normalized_text_overlap",
        )
    ):
        raise ValueError("source split has an integrity overlap")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    shutil.copytree(
        source / "models" / "audio-whisper-tiny", output / "models" / "audio-whisper-tiny"
    )
    shutil.copytree(source / "models" / "text-minilm-en", output / "models" / "text-minilm-en")
    (output / "readout").mkdir()
    shutil.copy2(checkpoint, output / "readout" / "best-readout.pt")
    runtime_root = output / "runtime" / "tiny_omni_decision"
    runtime_root.mkdir(parents=True)
    package_root = Path(__file__).resolve().parents[1] / "src" / "tiny_omni_decision"
    for filename in (
        "__init__.py",
        "decision.py",
        "decision_json.py",
        "audio_runtime.py",
        "feature_cache.py",
    ):
        shutil.copy2(package_root / filename, runtime_root / filename)

    readme = [
        "# LibriSpeech Path A transcript-choice runtime",
        "",
        "This bundle contains frozen Whisper Tiny and MiniLM encoders plus a readout.",
        "It is limited to four-way supplied-transcript choice for spoken audio.",
        "It is not general QA, ASR, multimodal fusion, or a product release.",
        "",
        "Input is mono 16-bit PCM WAV at 16 kHz. Output probabilities follow caller option order.",
        "Metrics use development validation for checkpoint selection, not a blind audit.",
        "",
        "`bundle-manifest.json` records revisions, hashes, parameters, package bytes, and limits.",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")
    files, bundle_bytes = tree_manifest(output)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "safetensors": importlib.metadata.version("safetensors"),
        "cuda_available_at_build": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    manifest = {
        "schema_version": 1,
        "bundle_id": "path-a-audio-librispeech-transcript-choice-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip(),
        "scope": (
            "development runtime for LibriSpeech train-clean-100 supplied-transcript choice only"
        ),
        "models": {
            "audio": {
                "id": AUDIO_MODEL_ID,
                "revision": AUDIO_MODEL_REVISION,
                "license": "Apache-2.0",
                "weights_sha256": AUDIO_WEIGHTS_SHA256,
                "parameters": 37_760_640,
                "directory": "models/audio-whisper-tiny",
            },
            "text": {
                "id": TEXT_MODEL_ID,
                "revision": TEXT_MODEL_REVISION,
                "license": "Apache-2.0",
                "weights_sha256": TEXT_WEIGHTS_SHA256,
                "parameters": 17_389_824,
                "directory": "models/text-minilm-en",
            },
            "readout": {
                "path": "readout/best-readout.pt",
                "sha256": expected_readout_hash,
                "parameters": int(run_report["models"]["readout_parameters_trainable"]),
                "selection": (
                    f"minimum equal-condition held-speaker validation NLL, epoch {selected_epoch}"
                ),
                "validation_macro": run_report["validation_metrics"][
                    "selected_librispeech_readout"
                ]["macro_condition"],
            },
        },
        "total_parameters": int(run_report["models"]["full_component_parameters"]),
        "source_run_report_sha256": sha256_file(run_report_path),
        "source_run_config_sha256": run_report["config_sha256"],
        "source_split_manifest_sha256": run_report["split_manifest_sha256"],
        "source_validation_examples_sha256": run_report["artifacts"]["validation-examples.jsonl"][
            "sha256"
        ],
        "environment_at_build": environment,
        "bundle_bytes_excluding_manifest": bundle_bytes,
        "files": files,
        "limitations": [
            "only one English audiobook transcript-choice task was measured",
            "development validation was used for checkpoint selection; no blind audit claim",
            "audio must be 16 kHz mono PCM WAV; FLAC decoding is outside this runtime API",
            "Python dependencies are external and not included in bundle bytes",
            "quantized package size, memory, latency, and target-device performance are unmeasured",
        ],
    }
    (output / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    verify_bundle(output, manifest)
    print(json.dumps({"bundle": str(output), "bundle_bytes": bundle_bytes, "files": len(files)}))


if __name__ == "__main__":
    main()
