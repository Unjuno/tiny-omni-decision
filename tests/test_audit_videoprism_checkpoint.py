from __future__ import annotations

import hashlib

import pytest

from scripts.audit_videoprism_checkpoint import validate_checkpoint


def test_validate_checkpoint_accepts_matching_sha256(tmp_path):
    checkpoint = tmp_path / "weights.npz"
    checkpoint.write_bytes(b"pinned-model-weights")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    assert validate_checkpoint(checkpoint, expected) == expected


def test_validate_checkpoint_rejects_wrong_sha256(tmp_path):
    checkpoint = tmp_path / "weights.npz"
    checkpoint.write_bytes(b"different-checkpoint")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_checkpoint(checkpoint, "0" * 64)
