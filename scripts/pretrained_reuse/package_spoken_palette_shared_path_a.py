"""Extend an immutable Path A assembly with the shared-Whisper AV task."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tiny_omni_decision.modular_bundle import extend_modular_bundle, verify_modular_bundle

ROOT = Path(__file__).resolve().parents[2]
WHISPER_ID = "openai/whisper-tiny"
WHISPER_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
WHISPER_FULL_SHA256 = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
WHISPER_ENCODER_SHA256 = "d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8"
WHISPER_PARAMETERS = 37_760_640
DINO_PARAMETERS = 22_056_576
SCORER_PARAMETERS = 98_561


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_source_file(root: Path, entry: dict[str, Any]) -> Path:
    relative = Path(str(entry["path"]))
    source = (root / relative).resolve(strict=True)
    if not source.is_relative_to(root.resolve()):
        raise ValueError("source bundle file escapes its root")
    if source.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"source bundle file size mismatch: {relative}")
    if sha256_file(source) != entry["sha256"]:
        raise ValueError(f"source bundle file hash mismatch: {relative}")
    return source


def build_shared_task_stage(
    source_bundle: Path, stage: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    source = source_bundle.resolve(strict=True)
    receipt = json.loads((source / "bundle-receipt.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "runtime-bundle.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("device") != "cpu"
        or sha256_file(source / "runtime-bundle.json")
        != receipt.get("runtime_bundle_sha256")
    ):
        raise ValueError("spoken-palette source runtime receipt failed")
    if stage.exists():
        raise FileExistsError(f"refusing to overwrite task staging directory: {stage}")
    stage.mkdir(parents=True)

    file_entries = metadata.get("files")
    if not isinstance(file_entries, dict):
        raise ValueError("spoken-palette source bundle has no file inventory")
    excluded = {"whisper_encoder", "whisper_encoder.safetensors"}
    retained = {key: entry for key, entry in file_entries.items() if key not in excluded}
    if not excluded <= set(file_entries):
        raise ValueError("expected standalone Whisper encoder entries are missing")

    staged_files: dict[str, Path] = {}
    for entry in retained.values():
        source_path = _verified_source_file(source, entry)
        relative = Path(str(entry["path"]))
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        if sha256_file(destination) != entry["sha256"]:
            raise ValueError(f"staged copy hash mismatch: {relative}")
        staged_files[relative.as_posix()] = destination

    model_entries = metadata.get("source_revisions", {})
    if model_entries.get("whisper") != f"{WHISPER_ID}@{WHISPER_REVISION}":
        raise ValueError("spoken-palette Whisper revision is not the pinned Path A revision")
    metadata["files"] = retained
    metadata["shared_audio_encoder"] = {
        "model_id": WHISPER_ID,
        "revision": WHISPER_REVISION,
        "weights_sha256": WHISPER_FULL_SHA256,
        "shared_encoder_sha256": WHISPER_ENCODER_SHA256,
        "shared_encoder_parameter_count": 8_208_384,
        "identity_check": (
            "all 67 WhisperEncoder tensors match the corresponding tensors under "
            "model.encoder.* in the Path A full Whisper-tiny checkpoint"
        ),
        "processor_policy": (
            "retain the spoken-palette WhisperFeatureExtractor configuration; its "
            "mel_filters differ from the Path A Speech Commands processor"
        ),
    }
    metadata["integration_scope"] = (
        "Path A task-scoped dispatcher plus the narrow spoken-color/generated-screen "
        "Decision task; not a general Omni quality claim"
    )
    bundle_path = stage / "runtime-bundle.json"
    bundle_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    staged_files["runtime-bundle.json"] = bundle_path
    new_receipt = {
        "runtime_bundle_sha256": sha256_file(bundle_path),
        "payload_files": len(staged_files),
        "shared_audio_encoder": True,
        "sealed_audit_opened": False,
    }
    receipt_path = stage / "bundle-receipt.json"
    receipt_path.write_text(json.dumps(new_receipt, indent=2) + "\n", encoding="utf-8")
    staged_files["bundle-receipt.json"] = receipt_path
    return metadata, staged_files


def build(base_assembly: Path, source_bundle: Path, output_dir: Path) -> dict[str, Any]:
    base = base_assembly.resolve(strict=True)
    source = source_bundle.resolve(strict=True)
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite integrated assembly: {output}")
    base_manifest = verify_modular_bundle(base)
    audio_component = next(
        (item for item in base_manifest["components"] if item.get("name") == "audio"), None
    )
    if audio_component is None:
        raise ValueError("base Path A assembly has no Audio component")
    audio_model = next(
        (
            item
            for item in audio_component["parameters"]
            if item.get("name") == "model:audio"
        ),
        None,
    )
    if (
        audio_model is None
        or audio_model.get("identity") != WHISPER_FULL_SHA256
        or int(audio_model.get("parameters", -1)) != WHISPER_PARAMETERS
    ):
        raise ValueError("base assembly Whisper identity or parameter count differs")
    shared_asset_file = (
        base
        / "components/audio/models/audio-whisper-tiny/model.safetensors"
    )
    if sha256_file(shared_asset_file) != WHISPER_FULL_SHA256:
        raise ValueError("base assembly Whisper weight hash differs from the pinned source")

    stage = output.with_name(output.name + ".spoken-palette-task-staging")
    metadata, stage_files = build_shared_task_stage(source, stage)
    parameters = metadata["logical_prequantization_parameters"]
    if (
        int(parameters.get("dinov2_small", -1)) != DINO_PARAMETERS
        or int(parameters.get("decision_scorer", -1)) != SCORER_PARAMETERS
    ):
        raise ValueError("spoken-palette logical parameter accounting differs")
    source_report = (
        ROOT / "docs/pretrained_reuse/R2M_SPOKEN_PALETTE_AUDIO_IMAGE_DECISION_20261122.md"
    )
    source_report_sha = sha256_file(source_report)
    source_runtime = json.loads((source / "runtime-bundle.json").read_text(encoding="utf-8"))
    source_code_files = {
        "package_builder": Path(__file__).resolve(),
        "spoken_palette_runtime": ROOT / "src/tiny_omni_decision/spoken_palette_runtime.py",
        "runtime_dispatch": ROOT / "src/tiny_omni_decision/runtime_dispatch.py",
        "torch_worker": ROOT / "scripts/path_a_torch_worker.py",
    }
    source_env = {
        "python": "3.11.9",
        "torch": "2.6.0+cu124",
        "transformers": "5.6.2",
        "device": "cpu only for this task runtime",
        "source_code_sha256": {
            name: sha256_file(path) for name, path in source_code_files.items()
        },
    }
    shared_tasks = {
        "spoken_palette_av": {
            "task_id": "spoken-color-generated-screen-supplied-option-v1",
            "scope": metadata["task_scope"],
            "source_report_sha256": source_report_sha,
            "runtime_environment": source_env,
            "files": stage_files,
            "parameters": [
                {
                    "name": "dinov2-small-dynamic-int8",
                    "path": "dinov2/dynamic-int8-state.pt",
                    "identity": source_runtime["files"]["dinov2_dynamic_int8_state"]["sha256"],
                    "parameters": DINO_PARAMETERS,
                },
                {
                    "name": "spoken-palette-decision-scorer",
                    "path": "decision/scorer.safetensors",
                    "identity": source_runtime["files"]["scorer"]["sha256"],
                    "parameters": SCORER_PARAMETERS,
                },
            ],
            "shared_assets": [
                {
                    "path": "components/audio",
                    "file": "models/audio-whisper-tiny/model.safetensors",
                    "weights_sha256": WHISPER_FULL_SHA256,
                    "parameters": WHISPER_PARAMETERS,
                    "revision": f"{WHISPER_ID}@{WHISPER_REVISION}",
                }
            ],
        }
    }
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = extend_modular_bundle(
        base,
        shared_tasks,
        output,
        source_commit=source_commit,
    )
    return {
        "status": "PASS_SHARED_AUDIO_AV_TASK_PACKAGED",
        "source_commit": source_commit,
        "base_manifest_sha256": sha256_file(base / "package-manifest.json"),
        "output_dir": str(output),
        "package_manifest_sha256": sha256_file(output / "package-manifest.json"),
        "unique_runtime_parameters": manifest["unique_runtime_parameters"],
        "package_file_bytes_excluding_manifest": manifest[
            "package_file_bytes_excluding_manifest"
        ],
        "package_target_met": manifest["package_target_met"],
        "shared_audio_encoder": True,
        "task_staging_dir": str(stage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-assembly", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base_assembly, args.source_bundle, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
