from __future__ import annotations

import pytest

from tiny_omni_decision.bundle_validation import verify_bundle


def test_runtime_bundle_verifier_checks_hashes_sizes_and_exact_file_set(tmp_path) -> None:
    package = tmp_path / "bundle"
    package.mkdir()
    payload = package / "runtime.py"
    payload.write_bytes(b"verified runtime")
    import hashlib

    manifest = {
        "files": [
            {
                "path": "runtime.py",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ]
    }

    verify_bundle(package, manifest)

    (package / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        verify_bundle(package, manifest)


def test_runtime_bundle_verifier_rejects_tampered_payload(tmp_path) -> None:
    package = tmp_path / "bundle"
    package.mkdir()
    payload = package / "weight.bin"
    payload.write_bytes(b"before")
    import hashlib

    manifest = {
        "files": [
            {
                "path": "weight.bin",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ]
    }
    payload.write_bytes(b"after!")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(package, manifest)


def test_runtime_bundle_verifier_rejects_path_traversal(tmp_path) -> None:
    package = tmp_path / "bundle"
    package.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    import hashlib

    manifest = {
        "files": [
            {
                "path": "../outside.bin",
                "bytes": outside.stat().st_size,
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        ]
    }

    with pytest.raises(ValueError, match="outside the bundle"):
        verify_bundle(package, manifest)

