from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a frozen ternary student + Recovery LoRA bundle"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    config_path = (
        args.config.resolve() if args.config.is_absolute() else (repo / args.config).resolve()
    )
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite export bundle: {output}")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    student = config["student"]
    data = config["data"]
    run_dir = Path(config["training"]["output_dir"]).resolve()
    run_metadata_path = run_dir / "run-metadata.json"
    run_metadata_bytes = run_metadata_path.read_bytes()
    run_metadata = json.loads(run_metadata_bytes)
    if run_metadata.get("state") != "complete":
        raise ValueError("Recovery run is not complete")
    if run_metadata.get("sealed_audit_loaded") or run_metadata.get("product_teacher_promotion"):
        raise ValueError("run metadata violates the Recovery export boundary")

    base_source = Path(student["model_path"]).resolve()
    model_manifest_path = (repo / student["model_manifest"]).resolve()
    model_manifest = yaml.safe_load(model_manifest_path.read_text(encoding="utf-8"))
    overlay_source = Path(student["ternary_overlay"]).resolve()
    adapter_source = Path(run_metadata["best_adapter_dir"]).resolve()
    overlay_manifest_path = overlay_source / "manifest.json"
    overlay_weights_path = overlay_source / "weights.safetensors"
    adapter_weights_path = adapter_source / "adapter_model.safetensors"
    adapter_config_path = adapter_source / "adapter_config.json"

    if sha256(model_manifest_path) != run_metadata["base_manifest_sha256"]:
        raise ValueError("student base manifest hash differs from completed run")
    if sha256(config_path) != run_metadata["config_sha256"]:
        raise ValueError("Recovery config hash differs from completed run")
    if (
        run_metadata["source_qat_run_metadata_sha256"]
        != student["source_qat_run_metadata_sha256"]
    ):
        raise ValueError("source QAT run metadata provenance mismatch")
    if sha256(overlay_manifest_path) != run_metadata["ternary_overlay_manifest_sha256"]:
        raise ValueError("selected ternary overlay manifest hash mismatch")
    if sha256(overlay_weights_path) != run_metadata["ternary_overlay_tensor_sha256"]:
        raise ValueError("selected ternary overlay tensor hash mismatch")
    if sha256(adapter_weights_path) != run_metadata["best_adapter_sha256"]:
        raise ValueError("selected Recovery adapter hash mismatch")
    if run_metadata["source_qat_best_shadow_sha256"] != student["source_best_shadow_sha256"]:
        raise ValueError("selected QAT shadow provenance mismatch")
    target_names = run_metadata["recovery_target_modules"]
    if len(target_names) != 168 or any(
        not name.startswith("language_model.layers.") for name in target_names
    ):
        raise ValueError("Recovery adapter target inventory escapes the decoder-only policy")

    files: list[dict[str, Any]] = []
    for entry in model_manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model manifest contains an unsafe relative path")
        source = base_source / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if "sha256" in entry and sha256(source) != entry["sha256"]:
            raise ValueError(f"pinned base file hash mismatch: {relative}")
        if "size_bytes" in entry and source.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"pinned base file size mismatch: {relative}")
        if "git_blob_sha1" in entry and git_blob_sha1(source) != entry["git_blob_sha1"]:
            raise ValueError(f"pinned base Git blob hash mismatch: {relative}")
        destination = output / "base" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": str(Path("base") / relative).replace("\\", "/"),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    overlay_destination = output / "ternary-overlay"
    overlay_destination.mkdir(parents=True)
    for source in (overlay_manifest_path, overlay_weights_path):
        destination = overlay_destination / source.name
        shutil.copy2(source, destination)
        files.append(
            {
                "path": str(destination.relative_to(output)).replace("\\", "/"),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    adapter_destination = output / "recovery-adapter"
    adapter_destination.mkdir(parents=True)
    for source in (adapter_config_path, adapter_weights_path):
        destination = adapter_destination / source.name
        shutil.copy2(source, destination)
        files.append(
            {
                "path": str(destination.relative_to(output)).replace("\\", "/"),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    shutil.copy2(config_path, output / "recovery-config.yaml")
    shutil.copy2(run_metadata_path, output / "recovery-run-metadata.json")
    files.extend(
        {
            "path": name,
            "bytes": (output / name).stat().st_size,
            "sha256": sha256(output / name),
        }
        for name in ("recovery-config.yaml", "recovery-run-metadata.json")
    )
    overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "format": "tiny-omni-embeddinggemma2-ternary-recovery-bundle-v1",
        "status": "exported_validation_candidate_not_product_promoted",
        "base_model_id": model_manifest["repo_id"],
        "base_revision": model_manifest["revision"],
        "base_license": model_manifest["license"],
        "base_manifest_sha256": sha256(model_manifest_path),
        "base_checkpoint_is_included": True,
        "ternary_overlay_manifest_sha256": sha256(overlay_manifest_path),
        "ternary_overlay_tensor_sha256": sha256(overlay_weights_path),
        "ternary_target_parameter_count": overlay_manifest["target_parameter_count"],
        "ternary_packed_code_bytes": overlay_manifest["packed_code_bytes"],
        "ternary_scale_bytes": overlay_manifest["scale_bytes"],
        "higher_precision_parameter_bytes": overlay_manifest["higher_precision_parameter_bytes"],
        "runtime_buffer_bytes": overlay_manifest["runtime_buffer_bytes"],
        "selected_qat_step": run_metadata["source_qat_best_step"],
        "selected_recovery_step": run_metadata["best_validation_step"],
        "recovery_adapter_sha256": sha256(adapter_weights_path),
        "recovery_adapter_bytes": adapter_weights_path.stat().st_size,
        "recovery_adapter_rank": run_metadata["recovery_rank"],
        "recovery_trainable_parameter_count": run_metadata["trainable_parameter_count"],
        "recovery_target_modules": adapter_config.get("target_modules"),
        "score_temperature": student["score_temperature"],
        "validation_snapshot_sha256": data["validation_snapshot_sha256"],
        "validation_ordered_ids_sha256": run_metadata["validation_ids_ordered_sha256"],
        "run_metadata_sha256": sha256(run_metadata_path),
        "config_sha256": sha256(config_path),
        "files": [],
        "total_file_bytes": 0,
        "license_notice": (
            "EmbeddingGemma 2 base is Apache-2.0 per pinned model metadata. Preserve "
            "Google attribution and upstream model card/license terms when redistributing."
        ),
        "upstream_model_card": "https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2",
    }
    write_json(output / "bundle-manifest.json", manifest)
    license_text = (
        "# Bundle license and attribution\n\n"
        "The included EmbeddingGemma 2 checkpoint and processor are identified by "
        f"`{model_manifest['repo_id']}@{model_manifest['revision']}`. The pinned "
        f"manifest reports `{model_manifest['license']}`. Preserve Google DeepMind "
        "attribution and follow the upstream model card and license terms: "
        "https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2.\n\n"
        "The packed ternary overlay and Recovery LoRA are project-derived artifacts. "
        "Their exact provenance and hashes are recorded in `bundle-manifest.json`.\n"
    )
    license_path = output / "LICENSE-NOTICE.md"
    license_path.write_text(license_text, encoding="utf-8")
    files.append(
        {
            "path": "LICENSE-NOTICE.md",
            "bytes": license_path.stat().st_size,
            "sha256": sha256(license_path),
        }
    )
    manifest["license_notice_file_sha256"] = sha256(license_path)
    manifest["files"] = sorted(files, key=lambda item: item["path"])
    manifest["total_file_bytes"] = sum(file["bytes"] for file in files)
    write_json(output / "bundle-manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "total_file_bytes": manifest["total_file_bytes"],
                "bundle_manifest_sha256": sha256(output / "bundle-manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
