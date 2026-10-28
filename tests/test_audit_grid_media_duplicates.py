import hashlib
import io
import tarfile
import zipfile

import pytest

from scripts.audit_grid_media_duplicates import (
    archive_member_hashes,
    digest_stream,
    find_duplicates,
)


def test_digest_stream_hashes_complete_payload():
    assert digest_stream(io.BytesIO(b"payload")) == hashlib.sha256(b"payload").hexdigest()


def test_archive_member_hashes_reads_audio_and_video_payloads(tmp_path):
    audio_path = tmp_path / "audio.tar"
    with tarfile.open(audio_path, "w") as archive:
        payload = b"wav bytes"
        info = tarfile.TarInfo("audio/s2/bbaa1a.wav")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    video_path = tmp_path / "video.zip"
    with zipfile.ZipFile(video_path, "w") as archive:
        archive.writestr("s2/bbaa1a.mpg", b"mpg bytes")

    assert archive_member_hashes(audio_path, ".wav") == {
        "bbaa1a": hashlib.sha256(b"wav bytes").hexdigest()
    }
    assert archive_member_hashes(video_path, ".mpg") == {
        "bbaa1a": hashlib.sha256(b"mpg bytes").hexdigest()
    }


def test_archive_member_hashes_reject_duplicate_media_identity(tmp_path):
    path = tmp_path / "audio.tar"
    with tarfile.open(path, "w") as archive:
        for name in ("a/bbaa1a.wav", "b/bbaa1a.wav"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="duplicate .wav payload"):
        archive_member_hashes(path, ".wav")


def test_find_duplicates_flags_cross_split_but_not_cross_modality_match():
    shared = hashlib.sha256(b"same audio").hexdigest()
    records = [
        {"modality": "audio", "sha256": shared, "split": "train", "code": "a"},
        {"modality": "audio", "sha256": shared, "split": "validation", "code": "b"},
        {"modality": "video", "sha256": shared, "split": "validation", "code": "c"},
    ]

    result = find_duplicates(records)

    assert len(result["cross_split_exact_payload_duplicates"]) == 1
    assert result["cross_split_exact_payload_duplicates"][0]["modality"] == "audio"
    assert result["within_split_exact_payload_duplicates"] == []
