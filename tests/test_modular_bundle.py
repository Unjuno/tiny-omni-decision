from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tiny_omni_decision.modular_bundle import assemble_modular_bundle, verify_modular_bundle


def _write_component(root: Path, bundle_id: str, file_name: str, contents: bytes) -> None:
    path = root / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    manifest = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "scope": "test component",
        "models": {
            "encoder": {
                "weights_sha256": hashlib.sha256(contents).hexdigest(),
                "parameters": 7,
            }
        },
        "readout": {
            "sha256": hashlib.sha256(contents + b"-readout").hexdigest(),
            "parameters": 3,
        },
        "files": [
            {
                "path": file_name,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        ],
    }
    (root / "bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_assembly_copies_and_verifies_immutable_components(tmp_path: Path) -> None:
    audio = tmp_path / "audio-source"
    image = tmp_path / "image-source"
    _write_component(audio, "audio-v0", "weights.bin", b"same")
    _write_component(image, "image-v0", "weights.bin", b"other")
    output = tmp_path / "assembled"

    manifest = assemble_modular_bundle(
        {"audio": audio, "image": image}, output, source_commit="abc123"
    )

    assert audio.joinpath("weights.bin").read_bytes() == b"same"
    assert image.joinpath("weights.bin").read_bytes() == b"other"
    assert manifest["unique_runtime_parameters"] == 20
    assert manifest["component_count"] == 2
    assert manifest["supports_joint_audio_video"] is False
    assert manifest["has_standalone_text_task"] is False
    assert "README.md" in {entry["path"] for entry in manifest["files"]}
    assert verify_modular_bundle(output)["package_target_met"] is True


def test_assembly_refuses_existing_output_and_bad_component(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_component(source, "audio-v0", "weights.bin", b"payload")
    output = tmp_path / "assembled"
    output.mkdir()
    with pytest.raises(FileExistsError):
        assemble_modular_bundle({"audio": source}, output, source_commit="abc123")

    (source / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="wrong-sized|hash mismatch"):
        assemble_modular_bundle(
            {"audio": source}, tmp_path / "rejected", source_commit="abc123"
        )


def test_verifier_rejects_tampering_and_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_component(source, "audio-v0", "weights.bin", b"payload")
    output = tmp_path / "assembled"
    assemble_modular_bundle({"audio": source}, output, source_commit="abc123")
    (output / "components/audio/weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="wrong-sized|hash mismatch"):
        verify_modular_bundle(output)

    manifest_path = output / "package-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../../escape"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the package|missing or wrong-sized"):
        verify_modular_bundle(output)


def test_verifier_rejects_accounting_and_component_path_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_component(source, "audio-v0", "weights.bin", b"payload")
    output = tmp_path / "assembled"
    assemble_modular_bundle({"audio": source}, output, source_commit="abc123")
    manifest_path = output / "package-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["unique_content_file_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unique content byte accounting"):
        verify_modular_bundle(output)

    manifest["unique_content_file_bytes"] -= 1
    manifest["components"][0]["path"] = "../../outside"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="component manifest contains a path outside"):
        verify_modular_bundle(output)
