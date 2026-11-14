import hashlib
import struct
import zlib
from types import SimpleNamespace

import pytest

from scripts import acquire_fsd50k_annotation_manifest as acquisition


def test_free_space_gate_uses_windows_compatible_disk_usage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        acquisition.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100, used=20, free=80),
    )

    acquisition.require_free_space(tmp_path, 80)


def test_free_space_gate_fails_closed_when_media_and_reserve_do_not_fit(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        acquisition.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100, used=20, free=79),
    )

    with pytest.raises(ValueError, match="insufficient disk space"):
        acquisition.require_free_space(tmp_path, 80)


def test_member_budget_accepts_observed_long_clip_and_rejects_over_limit():
    observed = SimpleNamespace(
        is_dir=lambda: False,
        flag_bits=0,
        file_size=48_532_044,
        compress_size=26_869_071,
        filename="FSD50K.dev_audio/observed.wav",
        volume=5,
        header_offset=0,
        CRC=1,
    )
    assert acquisition.planned_budget(
        {"train": [{"sample_id": "observed"}]},
        {"FSD50K.dev_audio/observed.wav": observed},
        central_directory_bytes=0,
    )[0] == 48_532_044

    oversized = SimpleNamespace(
        is_dir=lambda: False,
        flag_bits=0,
        file_size=acquisition.MAX_MEMBER_BYTES + 1,
        compress_size=1,
        filename="FSD50K.dev_audio/oversized.wav",
        volume=5,
        header_offset=0,
        CRC=1,
    )
    with pytest.raises(ValueError, match="unsupported or oversized"):
        acquisition.planned_budget(
            {"train": [{"sample_id": "oversized"}]},
            {"FSD50K.dev_audio/oversized.wav": oversized},
            central_directory_bytes=0,
        )


def test_extract_member_uses_zip_name_length_before_extra_length(monkeypatch, tmp_path):
    member_name = "FSD50K.dev_audio/example.wav"
    payload = b"small wav payload"
    extra = b"zip-extra"
    checksum = zlib.crc32(payload)
    header = struct.pack(
        "<4s5H3I2H",
        b"PK\x03\x04",
        20,
        0,
        0,
        0,
        0,
        checksum,
        len(payload),
        len(payload),
        len(member_name),
        len(extra),
    )
    raw = header + member_name.encode() + extra + payload
    monkeypatch.setattr(
        acquisition,
        "_read_parts",
        lambda readers, disk, offset, count: header if count == 30 else raw,
    )
    info = SimpleNamespace(
        volume=0,
        header_offset=0,
        compress_type=0,
        compress_size=len(payload),
        file_size=len(payload),
        CRC=checksum,
    )
    destination = tmp_path / "example.wav"

    size, digest = acquisition._extract_member([], info, member_name, destination)

    assert destination.read_bytes() == payload
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
