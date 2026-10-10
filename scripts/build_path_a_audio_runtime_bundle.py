"""Build an immutable local bundle for the saved Speech Commands Path A runtime."""

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

import torch
import transformers

from tiny_omni_decision.audio_runtime import (
    AUDIO_MODEL_ID,
    AUDIO_MODEL_REVISION,
    AUDIO_WEIGHTS_SHA256,
    READOUT_SHA256,
    TEXT_MODEL_ID,
    TEXT_MODEL_REVISION,
    TEXT_WEIGHTS_SHA256,
)

DEFAULT_ROOT = Path(r"C:\CodexArtifacts\pretrained-reuse-r2")
DEFAULT_MODELS = DEFAULT_ROOT / "models"
DEFAULT_READOUT = DEFAULT_ROOT / "audio-multiquery-v0" / "best-readout.pt"
DEFAULT_RUN_REPORT = DEFAULT_ROOT / "audio-multiquery-v0" / "run-report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> tuple[list[dict[str, object]], int]:
    files = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": _sha256(path)})
        total_bytes += size
    return files, total_bytes


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--readout", type=Path, default=DEFAULT_READOUT)
    parser.add_argument("--run-report", type=Path, default=DEFAULT_RUN_REPORT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "path-a-audio-multiquery-runtime-bundle-v0",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    models_dir = args.models_dir.resolve()
    readout_path = args.readout.resolve()
    run_report_path = args.run_report.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime bundle: {output}")
    for path in (
        models_dir / "audio-whisper-tiny" / "model.safetensors",
        models_dir / "text-minilm-en" / "model.safetensors",
        readout_path,
        run_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(models_dir / "audio-whisper-tiny" / "model.safetensors") != AUDIO_WEIGHTS_SHA256:
        raise ValueError("pinned Whisper Tiny weight hash mismatch")
    if _sha256(models_dir / "text-minilm-en" / "model.safetensors") != TEXT_WEIGHTS_SHA256:
        raise ValueError("pinned MiniLM weight hash mismatch")
    if _sha256(readout_path) != READOUT_SHA256:
        raise ValueError("validation-selected Path A readout hash mismatch")
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    if run_report.get("model", {}).get("readout_state_dict_sha256") != READOUT_SHA256:
        raise ValueError("readout is not the checkpoint recorded by the source run")
    if run_report.get("sealed_audit_loaded") or run_report.get("test_split_loaded"):
        raise ValueError("source run unexpectedly accessed a held-out split")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    shutil.copytree(models_dir / "audio-whisper-tiny", output / "models" / "audio-whisper-tiny",
                    ignore=shutil.ignore_patterns(".cache"))
    shutil.copytree(models_dir / "text-minilm-en", output / "models" / "text-minilm-en",
                    ignore=shutil.ignore_patterns(".cache"))
    (output / "readout").mkdir()
    shutil.copy2(readout_path, output / "readout" / "best-readout.pt")
    runtime_root = output / "runtime" / "tiny_omni_decision"
    runtime_root.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1] / "src" / "tiny_omni_decision"
    for filename in ("__init__.py", "decision.py", "audio_runtime.py"):
        shutil.copy2(source_root / filename, runtime_root / filename)

    (output / "README.md").write_text(
        "# Speech Commands Path A runtime bundle\n\n"
        "This is a local development bundle for the frozen Speech Commands "
        "multi-query experiment. It contains pinned Whisper Tiny and MiniLM "
        "snapshots, the validation-selected readout, and the runtime source. "
        "It is not a general audio QA model or a product release.\n\n"
        "Input must be an uncompressed mono 16-bit PCM WAV at 16 kHz. The API "
        "accepts a question and two or more candidate strings and returns a "
        "probability distribution in candidate order. Quality is measured only "
        "for the four templates listed in the source experiment report.\n\n"
        "See `bundle-manifest.json` for revisions, hashes, parameter counts, "
        "package bytes, and limits. Python dependencies are external and must "
        "match `environment_at_build`.\n",
        encoding="utf-8",
    )
    files, bundle_bytes = _tree_manifest(output)
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
        "bundle_id": "path-a-audio-multiquery-runtime-v0",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip(),
        "scope": (
            "development runtime for the four Speech Commands query templates; "
            "not general speech QA, joint AV reasoning, or a product release"
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
                "sha256": READOUT_SHA256,
                "parameters": 344_321,
                "selection": "minimum macro validation NLL, epoch 4",
            },
        },
        "total_parameters": 55_494_785,
        "source_run_report_sha256": _sha256(run_report_path),
        "source_validation_manifest_sha256": run_report["data_hashes"]["validation_manifest"],
        "environment_at_build": environment,
        "bundle_bytes_excluding_manifest": bundle_bytes,
        "files": files,
        "limitations": [
            "only Speech Commands v0.02 English keywords and four fixed query types were measured",
            "the validation set is development data, not a blind or sealed audit",
            "question wording and options outside the four trained templates are unvalidated",
            "software dependencies are not vendored; use the pinned runtime environment",
        ],
    }
    (output / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bundle": str(output), "bundle_bytes": bundle_bytes, "files": len(files)}))


if __name__ == "__main__":
    main()
