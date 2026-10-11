"""Build and verify a content-addressed assembly of independent runtimes."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tiny_omni_decision.bundle_validation import verify_bundle


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_parameters(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for name, model in manifest.get("models", {}).items():
        identity = (
            model.get("sha256") or model.get("export_sha256")
            if name == "readout"
            else model.get("weights_sha256")
        )
        artifacts.append(
            {
                "name": f"model:{name}",
                "identity": identity,
                "parameters": int(model["parameters"]),
            }
        )
    encoder = manifest.get("encoder")
    if encoder:
        artifacts.append(
            {
                "name": "encoder",
                "identity": encoder.get("checkpoint_sha256"),
                "parameters": int(encoder["parameters"]),
            }
        )
    for name, readout in manifest.get("readouts", {}).items():
        artifacts.append(
            {
                "name": f"readout:{name}",
                "identity": readout.get("sha256"),
                "parameters": int(readout["parameters"]),
            }
        )
    readout = manifest.get("readout")
    if readout:
        artifacts.append(
            {
                "name": "readout",
                "identity": readout.get("sha256") or readout.get("export_sha256"),
                "parameters": int(readout["parameters"]),
            }
        )
    return artifacts


def assemble_modular_bundle(
    components: dict[str, Path], output_dir: Path, *, source_commit: str
) -> dict[str, Any]:
    """Copy verified component bundles to a new root and write an assembly manifest.

    Existing components are only read. The destination must not exist, so a
    previous assembly can never be silently overwritten.
    """
    if not components or any(
        not name.strip() or "/" in name or "\\" in name for name in components
    ):
        raise ValueError("component names must be nonempty single path segments")
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite modular bundle: {output}")

    verified: list[tuple[str, Path, dict[str, Any]]] = []
    for name, component_root in components.items():
        source = component_root.resolve(strict=True)
        manifest_path = source / "bundle-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_bundle(source, manifest)
        verified.append((name, source, manifest))

    output.mkdir(parents=True)
    try:
        component_rows = []
        for name, source, manifest in verified:
            destination = output / "components" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
            copied_manifest = json.loads(
                (destination / "bundle-manifest.json").read_text(encoding="utf-8")
            )
            verify_bundle(destination, copied_manifest)
            component_rows.append(
                {
                    "name": name,
                    "bundle_id": manifest["bundle_id"],
                    "source_bundle_manifest_sha256": sha256_file(
                        source / "bundle-manifest.json"
                    ),
                    "path": f"components/{name}",
                    "scope": manifest.get("scope", ""),
                    "parameters": _runtime_parameters(manifest),
                    "runtime_environment": manifest.get("runtime_environment")
                    or manifest.get("environment_at_build", {}),
                }
            )

        (output / "README.md").write_text(
            "# Path A modular runtime assembly v0\n\n"
            "This directory combines independently verified, task-scoped Audio and Image runtime "
            "bundles. It is a packaging and relocation artifact, not a jointly trained or fused "
            "model. Audio supports the four fixed Speech Commands query templates; Image supports "
            "the four fixed CLEVR-4 taxonomies. The MiniLM text tower in the Audio component only "
            "supports that component's query and option encoding. This package has no standalone "
            "text-decision task, temporal video decision runtime, or joint audio-video task.\n\n"
            "The two runtime components use separate Python environments: the Audio runtime uses "
            "PyTorch/Transformers, while the VideoPrism Image runtime uses JAX/TensorFlow. This "
            "assembly does not vendor Python environments. See package-manifest.json for component "
            "hashes, exact sizes, parameters, and environment records. The measured package "
            "exceeds "
            "the provisional 200 MB engineering target.\n",
            encoding="utf-8",
        )

        all_files = []
        content_sizes: dict[str, int] = {}
        for path in sorted(item for item in output.rglob("*") if item.is_file()):
            relative = path.relative_to(output).as_posix()
            digest = sha256_file(path)
            size = path.stat().st_size
            all_files.append({"path": relative, "bytes": size, "sha256": digest})
            content_sizes.setdefault(digest, size)

        unique_parameter_artifacts: dict[str, int] = {}
        parameter_items = []
        for component in component_rows:
            for artifact in component["parameters"]:
                identity = artifact["identity"]
                if not identity:
                    raise ValueError(
                        f"component {component['name']} has an unhashed parameter artifact"
                    )
                count = artifact["parameters"]
                previous = unique_parameter_artifacts.setdefault(identity, count)
                if previous != count:
                    raise ValueError("one parameter artifact hash has conflicting parameter counts")
                parameter_items.append({**artifact, "component": component["name"]})

        manifest = {
            "schema_version": 1,
            "package_id": "path-a-modular-runtime-assembly-v0",
            "created_utc": datetime.now(UTC).isoformat(),
            "source_commit": source_commit,
            "status": "modular_task_scoped_assembly",
            "supports_joint_audio_video": False,
            "has_standalone_text_task": False,
            "has_temporal_video_task": False,
            "package_target_bytes": 200_000_000,
            "component_count": len(component_rows),
            "components": component_rows,
            "parameter_artifacts": parameter_items,
            "unique_runtime_parameters": sum(unique_parameter_artifacts.values()),
            "unique_content_file_bytes": sum(content_sizes.values()),
            "package_file_bytes_excluding_manifest": sum(
                row["bytes"] for row in all_files
            ),
            "package_target_met": sum(row["bytes"] for row in all_files) <= 200_000_000,
            "files": all_files,
        }
        (output / "package-manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return manifest
    except Exception:
        shutil.rmtree(output)
        raise


def verify_modular_bundle(package_dir: Path) -> dict[str, Any]:
    """Verify package inventory, file hashes, component manifests and accounting."""
    root = package_dir.resolve(strict=True)
    manifest_path = root / "package-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported modular package schema")

    expected = {}
    for entry in manifest.get("files", []):
        relative = Path(str(entry["path"]))
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("package manifest contains a path outside the package")
        if not resolved.is_file() or resolved.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"package file missing or wrong-sized: {relative}")
        if sha256_file(resolved) != entry["sha256"]:
            raise ValueError(f"package file hash mismatch: {relative}")
        expected[relative.as_posix()] = entry
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "package-manifest.json"
    }
    if actual != set(expected):
        raise ValueError("package file set differs from its manifest")

    for component in manifest.get("components", []):
        component_root = root / str(component["path"])
        component_manifest_path = component_root / "bundle-manifest.json"
        if sha256_file(component_manifest_path) != component["source_bundle_manifest_sha256"]:
            raise ValueError(f"component manifest hash mismatch: {component['name']}")
        verify_bundle(
            component_root,
            json.loads(component_manifest_path.read_text(encoding="utf-8")),
        )
    size = sum(int(entry["bytes"]) for entry in manifest["files"])
    if size != int(manifest["package_file_bytes_excluding_manifest"]):
        raise ValueError("package byte accounting mismatch")
    if bool(manifest["package_target_met"]) != (size <= int(manifest["package_target_bytes"])):
        raise ValueError("package target status disagrees with measured bytes")
    return manifest
