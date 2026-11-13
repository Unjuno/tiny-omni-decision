"""Copy only the hash-pinned files listed by an immutable package manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from tiny_omni_decision.modular_bundle import sha256_file, verify_modular_bundle


def rebuild_manifest_listed_package(source: Path, output: Path) -> dict[str, Any]:
    root = source.resolve(strict=True)
    destination = output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite package copy: {destination}")
    manifest_path = root / "package-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("package manifest must contain a nonempty file inventory")

    verified: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for entry in entries:
        relative_text = str(entry["path"])
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in seen:
            raise ValueError(f"unsafe or duplicate manifest path: {relative_text}")
        seen.add(relative_text)
        source_path = (root / Path(*relative.parts)).resolve(strict=True)
        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise ValueError(f"manifest path escapes package or is not a file: {relative_text}")
        if source_path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"manifest size mismatch: {relative_text}")
        if sha256_file(source_path) != str(entry["sha256"]):
            raise ValueError(f"manifest hash mismatch: {relative_text}")
        verified.append((source_path, destination.joinpath(*relative.parts)))

    destination.mkdir(parents=True)
    for source_path, destination_path in verified:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    shutil.copy2(manifest_path, destination / manifest_path.name)
    verification = verify_modular_bundle(destination)
    return {
        "status": "PASS_MANIFEST_LISTED_PACKAGE_COPY",
        "source": str(root),
        "output": str(destination),
        "manifest_listed_files": len(verified),
        "package_file_bytes_excluding_root_manifest": verification[
            "package_file_bytes_excluding_manifest"
        ],
        "unique_runtime_parameters": verification["unique_runtime_parameters"],
        "package_target_met": verification["package_target_met"],
        "strict_verifier_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rebuild_manifest_listed_package(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
