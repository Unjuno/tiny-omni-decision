from __future__ import annotations

import io
import zipfile

import pytest

from tiny_omni_decision.robomnist_media import (
    compressed_member_range,
    decode_pinned_member,
)


def _member(data: bytes, compression: int = zipfile.ZIP_DEFLATED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        archive.writestr("folder/member.bin", data)
        info = archive.getinfo("folder/member.bin")
    blob = stream.getvalue()
    header = blob[info.header_offset : info.header_offset + 30]
    offset, length = compressed_member_range(
        header,
        {
            "name": info.filename,
            "offset": info.header_offset,
            "compressed_size": info.compress_size,
            "compression": info.compress_type,
        },
    )
    member = {
        "name": info.filename,
        "offset": info.header_offset,
        "compressed_size": info.compress_size,
        "compression": info.compress_type,
        "size": info.file_size,
        "crc32": f"{info.CRC:08x}",
    }
    return header, blob[offset : offset + length], member


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_ranged_zip_member_decodes_and_validates(compression: int) -> None:
    payload = b"video-or-audio-member\x00" * 200
    header, tail, member = _member(payload, compression)

    assert decode_pinned_member(header, tail, member) == payload


def test_ranged_member_fails_closed_on_filename_crc_or_truncation() -> None:
    payload = b"pinned payload" * 20
    header, tail, member = _member(payload)

    wrong_name = {**member, "name": "other.bin"}
    with pytest.raises(ValueError, match="filename"):
        decode_pinned_member(header, tail, wrong_name)

    wrong_crc = {**member, "crc32": "00000000"}
    with pytest.raises(ValueError, match="CRC"):
        decode_pinned_member(header, tail, wrong_crc)

    with pytest.raises(ValueError, match="integrity"):
        decode_pinned_member(header, tail[:-1], member)


def test_member_rejects_encrypted_or_wrong_compression_header() -> None:
    header, _, member = _member(b"plain")
    encrypted = bytearray(header)
    encrypted[6] |= 1
    with pytest.raises(ValueError, match="pinned member"):
        compressed_member_range(bytes(encrypted), member)

    wrong_method = bytearray(header)
    wrong_method[8:10] = (zipfile.ZIP_STORED).to_bytes(2, "little")
    with pytest.raises(ValueError, match="pinned member"):
        compressed_member_range(bytes(wrong_method), member)


def test_member_rejects_missing_or_out_of_range_metadata() -> None:
    with pytest.raises(ValueError, match="complete 30-byte"):
        compressed_member_range(b"short", {})

    header, _, member = _member(b"plain")
    with pytest.raises(ValueError, match="pinned member"):
        compressed_member_range(header, {**member, "offset": -1})
