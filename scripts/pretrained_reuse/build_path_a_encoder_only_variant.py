"""Build a new Path A assembly with the unused Whisper decoder removed."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safetensors import safe_open

from tiny_omni_decision.audio_runtime import (
    AUDIO_ENCODER_PARAMETERS,
    AUDIO_ENCODER_STATE_SHA256,
    AUDIO_WEIGHTS_SHA256,
)
from tiny_omni_decision.modular_bundle import sha256_file, verify_modular_bundle


def _verify_encoder_provenance(
    full_checkpoint: Path,
    encoder_state: Path,
    audit_receipt: Path,
) -> dict[str, Any]:
    receipt = json.loads(audit_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete_exact_encoder_only_export"
        or receipt.get("source_weights_sha256") != AUDIO_WEIGHTS_SHA256
        or int(receipt.get("inference_encoder_parameters", -1)) != AUDIO_ENCODER_PARAMETERS
        or receipt.get("max_encoder_weight_delta") != 0
        or receipt.get("encoder_smoke_max_output_delta") != 0
    ):
        raise ValueError("encoder audit receipt does not prove exact source parity")
    if sha256_file(full_checkpoint) != AUDIO_WEIGHTS_SHA256:
        raise ValueError("Path A source Whisper checkpoint hash mismatch")
    if sha256_file(encoder_state) != AUDIO_ENCODER_STATE_SHA256:
        raise ValueError("encoder state is not the pinned standalone Whisper encoder export")

    source_tensors: dict[str, Any] = {}
    with safe_open(full_checkpoint, framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith("model.encoder."):
                source_tensors[name.removeprefix("model.encoder.")] = source.get_tensor(name)
    with safe_open(encoder_state, framework="pt", device="cpu") as exported:
        if set(source_tensors) != set(exported.keys()):
            raise ValueError("source and exported encoder tensor names differ")
        for name, tensor in source_tensors.items():
            if not tensor.equal(exported.get_tensor(name)):
                raise ValueError(f"encoder tensor differs from the source checkpoint: {name}")
    if sum(tensor.numel() for tensor in source_tensors.values()) != AUDIO_ENCODER_PARAMETERS:
        raise ValueError("source encoder tensor count differs from the pinned parameter count")
    return {
        "source_checkpoint_sha256": AUDIO_WEIGHTS_SHA256,
        "encoder_state_sha256": AUDIO_ENCODER_STATE_SHA256,
        "encoder_tensor_count": len(source_tensors),
        "encoder_parameter_count": AUDIO_ENCODER_PARAMETERS,
        "encoder_tensor_values_exact": True,
        "audit_receipt_sha256": sha256_file(audit_receipt),
        "audit_state_sha256": str(receipt["encoder_state_sha256"]),
    }


def _refresh_package_inventory(root: Path, manifest: dict[str, Any]) -> None:
    files = []
    content_sizes: dict[str, int] = {}
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        if path.name == "package-manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": digest})
        content_sizes.setdefault(digest, size)
    manifest["files"] = files
    manifest["package_file_bytes_excluding_manifest"] = sum(item["bytes"] for item in files)
    manifest["unique_content_file_bytes"] = sum(content_sizes.values())
    manifest["package_target_met"] = manifest["package_file_bytes_excluding_manifest"] <= int(
        manifest["package_target_bytes"]
    )

    unique_parameters: dict[str, int] = {}
    parameter_artifacts = []
    component_parameter_sum = 0
    for component in manifest["components"]:
        for item in component.get("parameters", []):
            identity = str(item["identity"])
            count = int(item["parameters"])
            previous = unique_parameters.setdefault(identity, count)
            if previous != count:
                raise ValueError("parameter identity has conflicting counts")
            parameter_artifacts.append({**item, "component": component["name"]})
            component_parameter_sum += count
        for item in component.get("shared_parameter_refs", []):
            component_parameter_sum += int(item["parameters"])
    manifest["parameter_artifacts"] = parameter_artifacts
    manifest["unique_runtime_parameters"] = sum(unique_parameters.values())
    manifest["component_parameter_sum"] = component_parameter_sum


def build_variant(
    base_assembly: Path,
    encoder_state: Path,
    audit_receipt: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base = base_assembly.resolve(strict=True)
    encoder = encoder_state.resolve(strict=True)
    receipt_path = audit_receipt.resolve(strict=True)
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite encoder-only assembly: {output}")
    base_manifest_sha = sha256_file(base / "package-manifest.json")
    base_manifest = verify_modular_bundle(base)
    audio_path = base / "components/audio/models/audio-whisper-tiny/model.safetensors"
    provenance = _verify_encoder_provenance(audio_path, encoder, receipt_path)

    shutil.copytree(base, output)
    try:
        manifest_path = output / "package-manifest.json"
        package = json.loads(manifest_path.read_text(encoding="utf-8"))
        audio_component = next(item for item in package["components"] if item["name"] == "audio")
        palette_component = next(
            item for item in package["components"] if item["name"] == "spoken_palette_av"
        )

        audio_root = output / "components/audio"
        audio_model_dir = audio_root / "models/audio-whisper-tiny"
        compact_path = audio_model_dir / "encoder.safetensors"
        shutil.copyfile(encoder, compact_path)
        (audio_model_dir / "model.safetensors").unlink()

        audio_manifest_path = audio_root / "bundle-manifest.json"
        audio_manifest = json.loads(audio_manifest_path.read_text(encoding="utf-8"))
        audio_model = audio_manifest["models"]["audio"]
        audio_model.update(
            {
                "weights_sha256": AUDIO_ENCODER_STATE_SHA256,
                "parameters": AUDIO_ENCODER_PARAMETERS,
                "source_checkpoint_sha256": AUDIO_WEIGHTS_SHA256,
                "storage_format": "encoder_only_safetensors",
                "file": "encoder.safetensors",
            }
        )
        old_weight_path = "models/audio-whisper-tiny/model.safetensors"
        audio_manifest["files"] = [
            entry for entry in audio_manifest["files"] if entry["path"] != old_weight_path
        ]
        audio_manifest["files"].append(
            {
                "path": "models/audio-whisper-tiny/encoder.safetensors",
                "bytes": compact_path.stat().st_size,
                "sha256": AUDIO_ENCODER_STATE_SHA256,
            }
        )
        audio_manifest["total_parameters"] = (
            AUDIO_ENCODER_PARAMETERS
            + int(audio_manifest["models"]["text"]["parameters"])
            + int(audio_manifest["models"]["readout"]["parameters"])
        )
        audio_manifest["bundle_bytes_excluding_manifest"] = sum(
            int(entry["bytes"]) for entry in audio_manifest["files"]
        )
        audio_manifest["storage_variant"] = "encoder_only"
        audio_manifest["storage_variant_provenance"] = provenance
        audio_manifest_path.write_text(
            json.dumps(audio_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        audio_component["source_bundle_manifest_sha256"] = sha256_file(audio_manifest_path)
        for item in audio_component["parameters"]:
            if item.get("name") == "model:audio":
                item["identity"] = AUDIO_ENCODER_STATE_SHA256
                item["parameters"] = AUDIO_ENCODER_PARAMETERS

        task_manifest_path = output / "components/spoken_palette_av/task-manifest.json"
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        shared = next(
            item for item in task_manifest["shared_assets"] if item["path"] == "components/audio"
        )
        shared.update(
            {
                "file": "models/audio-whisper-tiny/encoder.safetensors",
                "weights_sha256": AUDIO_ENCODER_STATE_SHA256,
                "parameters": AUDIO_ENCODER_PARAMETERS,
                "source_checkpoint_sha256": AUDIO_WEIGHTS_SHA256,
            }
        )
        task_manifest["shared_encoder_provenance"] = provenance
        task_manifest_path.write_text(
            json.dumps(task_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        palette_component["task_manifest_sha256"] = sha256_file(task_manifest_path)
        palette_component["shared_parameter_refs"] = [
            {
                **item,
                "identity": AUDIO_ENCODER_STATE_SHA256,
                "parameters": AUDIO_ENCODER_PARAMETERS,
            }
            if item.get("identity") == AUDIO_WEIGHTS_SHA256
            else item
            for item in palette_component["shared_parameter_refs"]
        ]

        package.update(
            {
                "package_id": "path-a-modular-runtime-assembly-v4-whisper-encoder-only-v1",
                "created_utc": datetime.now(UTC).isoformat(),
                "source_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[2],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "storage_variant": "whisper_encoder_only",
                "source_package_id": base_manifest["package_id"],
                "source_package_manifest_sha256": base_manifest_sha,
                "encoder_only_provenance": provenance,
                "builder_sha256": sha256_file(Path(__file__).resolve()),
                "runtime_code_sha256": {
                    "audio_runtime.py": sha256_file(
                        Path(__file__).resolve().parents[2]
                        / "src/tiny_omni_decision/audio_runtime.py"
                    ),
                    "path_a_torch_worker.py": sha256_file(
                        Path(__file__).resolve().parents[2] / "scripts/path_a_torch_worker.py"
                    ),
                },
            }
        )
        readme = output / "README.md"
        readme.write_text(
            "# Path A modular runtime assembly — Whisper encoder-only variant\n\n"
            "This isolated storage variant removes Whisper decoder weights unused by the current "
            "Audio and spoken-palette inference routes. It preserves task scopes and readouts. "
            "It is not promoted and does not claim a package-size target or new quality result. "
            "The source assembly remains immutable. See package-manifest.json for hashes and "
            "exact byte and parameter accounting.\n",
            encoding="utf-8",
        )
        _refresh_package_inventory(output, package)
        manifest_path.write_text(
            json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        verified = verify_modular_bundle(output)
        if sha256_file(base / "package-manifest.json") != base_manifest_sha:
            raise ValueError("source package changed during encoder-only assembly")
        return {
            "status": "PASS_ENCODER_ONLY_ASSEMBLY_CREATED",
            "output_dir": str(output),
            "source_package_manifest_sha256": base_manifest_sha,
            "package_manifest_sha256": sha256_file(manifest_path),
            "source_package_bytes": base_manifest["package_file_bytes_excluding_manifest"],
            "output_package_bytes": verified["package_file_bytes_excluding_manifest"],
            "bytes_saved": base_manifest["package_file_bytes_excluding_manifest"]
            - verified["package_file_bytes_excluding_manifest"],
            "source_unique_runtime_parameters": base_manifest["unique_runtime_parameters"],
            "output_unique_runtime_parameters": verified["unique_runtime_parameters"],
            "package_target_met": verified["package_target_met"],
            "encoder_provenance": provenance,
        }
    except Exception:
        shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-assembly", type=Path, required=True)
    parser.add_argument("--encoder-state", type=Path, required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_variant(
                args.base_assembly, args.encoder_state, args.audit_receipt, args.output_dir
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
