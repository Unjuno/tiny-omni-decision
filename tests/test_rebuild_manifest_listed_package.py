from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.rebuild_manifest_listed_package import rebuild_manifest_listed_package


def _write_component(root: Path) -> None:
    payload = b"pinned-component"
    (root / "weights.bin").write_bytes(payload)
    readout = b"pinned-readout"
    (root / "readout.bin").write_bytes(readout)
    (root / "bundle-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_id": "test-component",
                "scope": "test-only",
                "models": {
                    "encoder": {
                        "weights_sha256": hashlib.sha256(payload).hexdigest(),
                        "parameters": 7,
                    }
                },
                "readout": {
                    "sha256": hashlib.sha256(readout).hexdigest(),
                    "parameters": 3,
                },
                "files": [
                    {
                        "path": "weights.bin",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    },
                    {
                        "path": "readout.bin",
                        "bytes": len(readout),
                        "sha256": hashlib.sha256(readout).hexdigest(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_rebuild_copies_only_manifest_listed_files_and_preserves_source(tmp_path: Path):
    from tiny_omni_decision.modular_bundle import assemble_modular_bundle

    source_component = tmp_path / "source-component"
    source_component.mkdir()
    _write_component(source_component)
    source = tmp_path / "source-package"
    assemble_modular_bundle({"audio": source_component}, source, source_commit="test")
    extra = source / "components/audio/__pycache__/module.pyc"
    extra.parent.mkdir()
    extra.write_bytes(b"unlisted-cache")

    output = tmp_path / "verified-copy"
    result = rebuild_manifest_listed_package(source, output)

    assert result["strict_verifier_passed"] is True
    assert (source / "components/audio/__pycache__/module.pyc").is_file()
    assert not (output / "components/audio/__pycache__").exists()


def test_rebuild_refuses_tampered_payload_and_existing_output(tmp_path: Path):
    from tiny_omni_decision.modular_bundle import assemble_modular_bundle

    source_component = tmp_path / "source-component"
    source_component.mkdir()
    _write_component(source_component)
    source = tmp_path / "source-package"
    assemble_modular_bundle({"audio": source_component}, source, source_commit="test")
    listed = source / "components/audio/weights.bin"
    listed.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="manifest size mismatch|manifest hash mismatch"):
        rebuild_manifest_listed_package(source, tmp_path / "rejected")
    with pytest.raises(FileExistsError):
        rebuild_manifest_listed_package(source, source)
