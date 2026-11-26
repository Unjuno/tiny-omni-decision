import hashlib
import json

import pytest


def test_worker_physical_dependencies_are_checked_before_replay(tmp_path):
    from scripts.replay_native_av_correspondence import verified_primary

    dependency = tmp_path / "dependency.py"
    dependency.write_bytes(b"fixed")
    parent_dependency = tmp_path / "producer.py"
    parent_dependency.write_bytes(b"parent")
    child = tmp_path / "video-worker"
    child.mkdir()
    receipt = child / "result.json"
    receipt.write_text(
        json.dumps(
            dict(
                status="PASS_NATIVE_VIDEO_WORKER",
                validated_hashes={str(dependency): hashlib.sha256(b"fixed").hexdigest()},
            )
        ),
        encoding="utf-8",
    )
    primary = tmp_path / "result.json"
    primary.write_text(
        json.dumps(
            dict(
                status="PASS_NATIVE_AV_CORRESPONDENCE_RUNTIME",
                profile_sha256="a" * 64,
                validated_hashes={str(parent_dependency): hashlib.sha256(b"parent").hexdigest()},
                artifact_hashes={
                    "video-worker/result.json": hashlib.sha256(receipt.read_bytes()).hexdigest()
                },
                video_worker_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            )
        ),
        encoding="utf-8",
    )
    expected = hashlib.sha256(primary.read_bytes()).hexdigest()
    verified_primary(tmp_path, expected)
    dependency.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verified_primary(tmp_path, expected)


def test_same_numeric_outputs_cannot_override_changed_profile():
    from scripts.replay_native_av_correspondence import require_profile

    require_profile("a" * 64, "a" * 64)
    with pytest.raises(ValueError):
        require_profile("b" * 64, "a" * 64)
