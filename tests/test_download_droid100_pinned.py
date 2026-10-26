import base64
import hashlib
import json
from pathlib import Path

import pytest

from scripts.download_droid100_pinned import (
    safe_object_path,
    validate_manifest,
    verify_file,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/pretrained_reuse/droid100-pinned-objects-20261026.json"


def test_pinned_manifest_has_exact_100_episode_shard_inventory():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(data)
    assert data["expected_episode_count"] == 100
    assert sum("tfrecord-" in row["name"] for row in data["objects"]) == 31


def test_manifest_rejects_missing_or_untrusted_shard():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["objects"].pop()
    with pytest.raises(ValueError, match="two metadata objects and 31 TFRecord shards"):
        validate_manifest(data)


def test_object_path_rejects_traversal(tmp_path):
    with pytest.raises(ValueError, match="safe relative path"):
        safe_object_path(tmp_path, "robotics/../outside.tfrecord")


def test_download_verifier_checks_expected_size_and_gcs_md5(tmp_path):
    payload = b"pinned tfrecord test"
    file_path = tmp_path / "example"
    file_path.write_bytes(payload)
    item = {
        "name": "example",
        "size": len(payload),
        "md5Hash": base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode(),
    }
    assert verify_file(file_path, item)["sha256"] == hashlib.sha256(payload).hexdigest()
    file_path.write_bytes(payload + b"!")
    with pytest.raises(ValueError, match="size/MD5 mismatch"):
        verify_file(file_path, item)
