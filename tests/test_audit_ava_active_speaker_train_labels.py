from __future__ import annotations

import io
import tarfile

import pytest

from scripts.audit_ava_active_speaker_train_labels import audit_train_archive


def _make_tar(path, member_name: str, payload: bytes) -> None:
    with tarfile.open(path, mode="w:bz2") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def test_train_label_audit_counts_rows_without_rewriting_label_variants(tmp_path):
    archive = tmp_path / "ava_activespeaker_train_v1.0.tar.bz2"
    _make_tar(
        archive,
        "ava_activespeaker_train_v1.0/video_01-activespeaker.csv",
        (
            b"video_01,10,0.1,0.2,0.5,0.8,SPEAKING_AUDIBLE,track_1\n"
            b"video_01,10.04,0.1,0.2,0.5,0.8,SPEAKING_NOT_AUDIBLE,track_1\n"
        ),
    )

    result = audit_train_archive(archive)

    assert result["train_video_count"] == 1
    assert result["train_annotation_rows"] == 2
    assert result["label_counts"] == {
        "SPEAKING_AUDIBLE": 1,
        "SPEAKING_NOT_AUDIBLE": 1,
    }
    assert result["validation_archive_opened"] is False
    assert result["challenge_test_archive_opened"] is False
    assert result["invalid_row_counts"] == {}


def test_audit_rejects_non_train_archive_before_opening_it(tmp_path):
    archive = tmp_path / "ava_activespeaker_val_v1.0.tar.bz2"
    archive.write_bytes(b"not opened")

    with pytest.raises(ValueError, match="TRAIN archive"):
        audit_train_archive(archive)


def test_audit_rejects_unexpected_member_prefix(tmp_path):
    archive = tmp_path / "ava_activespeaker_train_v1.0.tar.bz2"
    _make_tar(archive, "ava_activespeaker_test_v1.0/video_01.csv", b"")

    with pytest.raises(ValueError, match="unexpected TRAIN member path"):
        audit_train_archive(archive)
