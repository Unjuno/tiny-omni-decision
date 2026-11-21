import hashlib
import zipfile
from io import StringIO

import pytest

from scripts.extract_soundactions_frozen_features import (
    AUDIO_WINDOW_SAMPLES,
    audio_window_starts,
)
from scripts.prepare_soundactions_media import (
    _extract_verified_member,
    _publisher_hashes,
    selected_csv_rows,
)


def test_selected_csv_rows_skip_nonselected_labels_before_parsing():
    class GuardedCsvStream:
        def __init__(self, value):
            self.source = StringIO(value)

        def read(self, size=-1):
            return self.source.read(size)

        def readline(self, size=-1):
            line = self.source.readline(size)
            assert "SECRET" not in line
            return line

    source = GuardedCsvStream(
        "number,perception_type,title\n"
        '1,iterative,"train title, safe"\n'
        '2,SECRET,"sealed value must not enter selected rows"\n'
    )
    assert selected_csv_rows(source, {1}) == {
        1: {"number": "1", "perception_type": "iterative", "title": "train title, safe"}
    }


def test_selected_csv_rows_fail_closed_on_malformed_selected_record():
    source = StringIO("number,label\n1\n")
    with pytest.raises(ValueError, match="delimiter"):
        selected_csv_rows(source, {1})


def test_publisher_hashes_select_only_allowed_audio_and_video(tmp_path):
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(
        "a" * 64 + "  JPG/SoundAction 001 - still.jpg\n"
        + "b" * 64
        + "  MP4/SoundAction 001 - video.mp4\n"
        + "c" * 64
        + "  WAV/SoundAction 001 - audio.wav\n",
        encoding="utf-8",
    )

    assert _publisher_hashes(checksums, {1}) == {("MP4", 1): "b" * 64, ("WAV", 1): "c" * 64}


def test_publisher_hashes_reject_missing_allowed_modality(tmp_path):
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text("b" * 64 + "  MP4/SoundAction 001 - video.mp4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        _publisher_hashes(checksums, {1})


def test_extract_verified_member_writes_then_publishes_complete_bytes(tmp_path):
    payload = b"verified training media" * 4096
    archive_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train/audio.wav", payload)
    destination = tmp_path / "output" / "audio.wav"
    destination.parent.mkdir()

    with zipfile.ZipFile(archive_path) as archive:
        _extract_verified_member(
            archive,
            "train/audio.wav",
            destination,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        )

    assert destination.read_bytes() == payload
    assert list(destination.parent.glob("*.partial")) == []


def test_extract_verified_member_preserves_existing_mismatched_file(tmp_path):
    archive_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train/audio.wav", b"source")
    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"user data")

    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(FileExistsError, match="publisher hash"),
    ):
        _extract_verified_member(
            archive,
            "train/audio.wav",
            destination,
            hashlib.sha256(b"source").hexdigest(),
            len(b"source"),
        )

    assert destination.read_bytes() == b"user data"


def test_extract_verified_member_does_not_publish_hash_mismatch(tmp_path):
    archive_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train/audio.wav", b"source")
    destination = tmp_path / "audio.wav"

    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="checksum"):
        _extract_verified_member(
            archive,
            "train/audio.wav",
            destination,
            hashlib.sha256(b"wrong").hexdigest(),
            len(b"source"),
        )

    assert not destination.exists()
    assert len(list(tmp_path.glob("audio.wav.*.partial"))) == 1


def test_audio_windows_cover_short_exact_and_long_recordings():
    assert audio_window_starts(100) == [0]
    assert audio_window_starts(AUDIO_WINDOW_SAMPLES) == [0]
    starts = audio_window_starts(AUDIO_WINDOW_SAMPLES + 1)
    assert starts == [0, 1]
    longer = audio_window_starts(AUDIO_WINDOW_SAMPLES + 2_500_000)
    assert longer[0] == 0
    assert longer[-1] == 2_500_000
    assert all(right - left <= 32_000 for left, right in zip(longer[:-1], longer[1:], strict=True))
