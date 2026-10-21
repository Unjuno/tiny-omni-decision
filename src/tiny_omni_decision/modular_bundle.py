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
    components: dict[str, Path],
    output_dir: Path,
    *,
    source_commit: str,
    shared_tasks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Copy verified component bundles to a new root and write an assembly manifest.

    Existing components are only read. The destination must not exist, so a
    previous assembly can never be silently overwritten.
    """
    shared_tasks = shared_tasks or {}
    names = [*components, *shared_tasks]
    if not names or len(names) != len(set(names)) or any(
        not name.strip() or "/" in name or "\\" in name for name in names
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

        for name, spec in shared_tasks.items():
            task_root = output / "components" / name
            task_root.mkdir(parents=True, exist_ok=True)
            local_files = []
            for relative_name, source_path in spec["files"].items():
                relative = Path(str(relative_name))
                destination = (task_root / relative).resolve()
                if not destination.is_relative_to(task_root.resolve()):
                    raise ValueError("shared task file path escapes its component")
                source = Path(source_path).resolve(strict=True)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                local_files.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                    }
                )

            local_file_hashes = {item["path"]: item["sha256"] for item in local_files}
            for artifact in spec["parameters"]:
                if artifact.get("path") not in local_file_hashes or local_file_hashes[
                    artifact["path"]
                ] != artifact["identity"]:
                    raise ValueError("shared task parameter identity does not match its file")

            shared_assets = []
            for asset in spec["shared_assets"]:
                relative = Path(str(asset["path"]))
                shared_path = (output / relative).resolve()
                if not shared_path.is_relative_to(output) or not shared_path.exists():
                    raise ValueError("shared task dependency is missing or outside the package")
                if shared_path.is_dir():
                    asset_file = str(asset.get("file", "model.safetensors"))
                    weight_path = (shared_path / asset_file).resolve()
                    if not weight_path.is_relative_to(shared_path):
                        raise ValueError("shared asset file path escapes its directory")
                else:
                    weight_path = shared_path
                if not weight_path.is_file() or sha256_file(weight_path) != asset["weights_sha256"]:
                    raise ValueError("shared task dependency weight hash mismatch")
                shared_assets.append({**asset, "path": relative.as_posix()})

            task_manifest = {
                "schema_version": 1,
                "task_id": spec["task_id"],
                "scope": spec["scope"],
                "source_report_sha256": spec["source_report_sha256"],
                "runtime_environment": spec.get("runtime_environment", {}),
                "shared_assets": shared_assets,
                "parameters": spec["parameters"],
                "files": local_files,
            }
            task_manifest_path = task_root / "task-manifest.json"
            task_manifest_path.write_text(
                json.dumps(task_manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            component_rows.append(
                {
                    "kind": "shared_task",
                    "name": name,
                    "task_id": task_manifest["task_id"],
                    "path": f"components/{name}",
                    "task_manifest_sha256": sha256_file(task_manifest_path),
                    "scope": spec["scope"],
                    "runtime_environment": spec.get("runtime_environment", {}),
                    "parameters": spec["parameters"],
                    "shared_parameter_refs": [
                        {
                            "identity": str(asset["weights_sha256"]),
                            "parameters": int(asset["parameters"]),
                        }
                        for asset in shared_assets
                    ],
                }
            )

        (output / "README.md").write_text(
            "# Path A modular runtime assembly v4\n\n"
            "This directory combines independently verified, task-scoped runtime components. It "
            "is a packaging artifact, not a jointly trained or fused model. Each task supports "
            "only "
            "the scope stated in its component manifest. Shared model files are referenced by path "
            "instead of copied again. There is no joint audio-video task.\n\n"
            "The components use separate Python environments where required and do not vendor "
            "those environments. See package-manifest.json for component hashes, exact sizes, "
            "parameters, shared assets, and environment records. The measured package exceeds the "
            "provisional 200 MB engineering target.\n",
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

        component_parameter_sum = sum(
            int(item["parameters"])
            for component in component_rows
            for item in component["parameters"]
        ) + sum(
            int(item["parameters"])
            for component in component_rows
            for item in component.get("shared_parameter_refs", [])
        )

        manifest = {
            "schema_version": 1,
            "package_id": "path-a-modular-runtime-assembly-v4",
            "created_utc": datetime.now(UTC).isoformat(),
            "source_commit": source_commit,
            "status": "modular_task_scoped_assembly",
            "supports_joint_audio_video": False,
            "has_standalone_text_task": "text" in shared_tasks,
            "has_temporal_video_task": "video" in shared_tasks,
            "package_target_bytes": 200_000_000,
            "component_count": len(component_rows),
            "components": component_rows,
            "parameter_artifacts": parameter_items,
            "unique_runtime_parameters": sum(unique_parameter_artifacts.values()),
            "component_parameter_sum": component_parameter_sum,
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

    expected: dict[str, dict[str, Any]] = {}
    content_sizes: dict[str, int] = {}
    for entry in manifest.get("files", []):
        relative = Path(str(entry["path"]))
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("package manifest contains a path outside the package")
        if relative.as_posix() in expected:
            raise ValueError(f"duplicate package file entry: {relative}")
        if not resolved.is_file() or resolved.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"package file missing or wrong-sized: {relative}")
        if sha256_file(resolved) != entry["sha256"]:
            raise ValueError(f"package file hash mismatch: {relative}")
        expected[relative.as_posix()] = entry
        content_sizes.setdefault(str(entry["sha256"]), int(entry["bytes"]))
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "package-manifest.json"
    }
    if actual != set(expected):
        raise ValueError("package file set differs from its manifest")

    components = manifest.get("components", [])
    if len(components) != int(manifest.get("component_count", -1)):
        raise ValueError("component count differs from package manifest")
    for component in components:
        component_root = (root / str(component["path"])).resolve()
        if not component_root.is_relative_to(root):
            raise ValueError("component manifest contains a path outside the package")
        if component.get("kind", "bundle") == "shared_task":
            task_manifest_path = component_root / "task-manifest.json"
            if sha256_file(task_manifest_path) != component["task_manifest_sha256"]:
                raise ValueError(f"task manifest hash mismatch: {component['name']}")
            task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
            if task_manifest.get("task_id") != component.get("task_id"):
                raise ValueError("task identity differs from component manifest")
            if task_manifest.get("parameters") != component.get("parameters"):
                raise ValueError("task parameters differ from component manifest")
            if task_manifest.get("scope") != component.get("scope"):
                raise ValueError("task scope differs from component manifest")
            task_files = {}
            for entry in task_manifest.get("files", []):
                relative = Path(str(entry["path"]))
                path = (component_root / relative).resolve()
                if not path.is_relative_to(component_root):
                    raise ValueError("task manifest contains a path outside its component")
                if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
                    raise ValueError(f"task file missing or wrong-sized: {relative}")
                if sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"task file hash mismatch: {relative}")
                task_files[relative.as_posix()] = entry
            actual_task_files = {
                path.relative_to(component_root).as_posix()
                for path in component_root.rglob("*")
                if path.is_file() and path.name != "task-manifest.json"
            }
            if actual_task_files != set(task_files):
                raise ValueError("task files differ from task manifest")
            for asset in task_manifest.get("shared_assets", []):
                shared_path = (root / str(asset["path"])).resolve()
                if not shared_path.is_relative_to(root):
                    raise ValueError("task dependency points outside the package")
                if shared_path.is_dir():
                    asset_file = str(asset.get("file", "model.safetensors"))
                    weight_path = (shared_path / asset_file).resolve()
                    if not weight_path.is_relative_to(shared_path):
                        raise ValueError("shared asset file path escapes its directory")
                else:
                    weight_path = shared_path
                if not weight_path.is_file() or sha256_file(weight_path) != asset["weights_sha256"]:
                    raise ValueError("shared task dependency weight hash mismatch")
        else:
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
    if sum(content_sizes.values()) != int(manifest["unique_content_file_bytes"]):
        raise ValueError("unique content byte accounting mismatch")
    parameter_counts: dict[str, int] = {}
    for artifact in manifest.get("parameter_artifacts", []):
        identity = str(artifact.get("identity", ""))
        if not identity:
            raise ValueError("parameter artifact has no identity hash")
        count = int(artifact["parameters"])
        previous = parameter_counts.setdefault(identity, count)
        if previous != count:
            raise ValueError("one parameter artifact hash has conflicting parameter counts")
    if sum(parameter_counts.values()) != int(manifest["unique_runtime_parameters"]):
        raise ValueError("parameter accounting mismatch")
    component_parameter_sum = sum(
        int(item["parameters"])
        for component in manifest.get("components", [])
        for item in component.get("parameters", [])
    ) + sum(
        int(item["parameters"])
        for component in manifest.get("components", [])
        for item in component.get("shared_parameter_refs", [])
    )
    if component_parameter_sum != int(manifest["component_parameter_sum"]):
        raise ValueError("component parameter accounting mismatch")
    if bool(manifest["package_target_met"]) != (size <= int(manifest["package_target_bytes"])):
        raise ValueError("package target status disagrees with measured bytes")
    return manifest
