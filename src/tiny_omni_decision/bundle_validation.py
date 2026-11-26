"""Validation helpers for immutable runtime bundles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(bundle: Path, manifest: dict[str, Any]) -> None:
    """Verify each declared file and reject undeclared files and path traversal."""
    bundle = bundle.resolve()
    expected_paths = set()
    for entry in manifest.get("files", []):
        relative = Path(str(entry["path"]))
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError("bundle manifest contains a path outside the bundle")
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"bundle file missing or wrong-sized: {relative}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"bundle file hash mismatch: {relative}")
        expected_paths.add(relative.as_posix())
    actual_paths = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    if actual_paths != expected_paths:
        raise ValueError("bundle file set differs from its manifest")
