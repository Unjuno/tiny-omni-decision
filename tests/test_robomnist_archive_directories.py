import io
import struct
import zipfile
from pathlib import Path

import pytest


def test_robomnist_archive_directory_auditor_is_available():
    assert Path("scripts/audit_robomnist_archive_directories.py").is_file()


def fake_zip_bytes():
    stem = "00_Ar1_Vhigh_Ac0/Rp01_Ac0_Vhigh_Ar1_UNC062_"
    names = [
        stem + f"Rx{i}_{kind}.{ext}"
        for i in [1, 2, 3]
        for kind, ext in [("cam", "mp4"), ("mic", "wav")]
    ] + [stem + "Rxall_csi.json", stem + "Rxall_pos.json"]
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in names:
            archive.writestr(name, b"x" * 262144)
    return target.getvalue()


class FakeResponse:
    def __init__(self, status, headers, body):
        self.status_code = status
        self.headers = headers
        self.body = body

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        pass


class FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, headers, stream, timeout):
        assert stream is True
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        body = self.payload[start : end + 1]
        return FakeResponse(
            206,
            {
                "Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
                "Content-Length": str(len(body)),
            },
            body,
        )


def test_reads_only_bounded_zip_directory_and_groups_complete_repetitions():
    from scripts.audit_robomnist_archive_directories import read_zip_directory

    payload = fake_zip_bytes()
    budget = {"total_remaining": 1024 * 1024, "transferred": 0}
    record = {
        "name": "00_Ar1_Vhigh_Ac0.zip",
        "size": len(payload),
        "download_url": "https://example.test/archive.zip",
    }
    result = read_zip_directory(FakeSession(payload), record, budget)
    assert result["repetition_groups"] == 1
    assert result["real_members"] == 8
    assert len(result["groups"][0]["input_members"]) == 7
    assert result["groups"][0]["supervision_only_member"].endswith("pos.json")
    assert all(member["local_header_offset"] >= 0 for member in result["member_inventory"])
    assert len({member["local_header_offset"] for member in result["member_inventory"]}) == 8
    assert result["range_bytes"] < len(payload)
    assert budget["transferred"] == result["range_bytes"]


@pytest.mark.parametrize("status", [200, 416])
def test_refuses_non_partial_http_response_without_reading_body(status):
    from scripts.audit_robomnist_archive_directories import read_zip_directory

    payload = fake_zip_bytes()

    class BadSession:
        def get(self, url, headers, stream, timeout):
            return FakeResponse(status, {}, payload)

    budget = {"total_remaining": 1024 * 1024, "transferred": 0}
    record = {
        "name": "00_Ar1_Vhigh_Ac0.zip",
        "size": len(payload),
        "download_url": "https://example.test/archive.zip",
    }
    with pytest.raises(ValueError, match="expected 206"):
        read_zip_directory(BadSession(), record, budget)
    assert budget["transferred"] == 0


def test_rejects_multidisk_or_zip64_eocd():
    from scripts.audit_robomnist_archive_directories import CENTRAL, EOCD, parse_eocd

    assert EOCD == b"PK\x05\x06"
    assert CENTRAL == b"PK\x01\x02"
    eocd = struct.pack("<4s4H2IH", EOCD, 1, 0, 1, 1, 46, 0, 0)
    with pytest.raises(ValueError, match="multi-disk"):
        parse_eocd(eocd, 0, len(eocd))
    zip64 = struct.pack("<4s4H2IH", EOCD, 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    with pytest.raises(ValueError, match="ZIP64"):
        parse_eocd(zip64, 0, len(zip64))


def test_repetition_payload_reader_extracts_one_crc_checked_member_by_range():
    from scripts.audit_robomnist_repetition_payload import extract_member

    payload = io.BytesIO()
    content = b"bounded test member" * 128
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("safe/member.bin", content)
    raw = payload.getvalue()
    info = zipfile.ZipFile(io.BytesIO(raw)).getinfo("safe/member.bin")
    member = {
        "name": info.filename,
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "local_header_offset": info.header_offset,
        "crc32": f"{info.CRC:08x}",
    }
    remaining = [1024 * 1024]
    expanded = [0]
    result = extract_member(
        FakeSession(raw), "https://example.test/archive.zip", len(raw), member, remaining, expanded
    )
    assert result == content
    assert expanded == [len(content)]
    assert remaining[0] < 1024 * 1024


def test_repetition_payload_reader_rejects_crc_mismatch():
    from scripts.audit_robomnist_repetition_payload import extract_member

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("safe/member.bin", b"content")
    raw = payload.getvalue()
    info = zipfile.ZipFile(io.BytesIO(raw)).getinfo("safe/member.bin")
    member = {
        "name": info.filename,
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "local_header_offset": info.header_offset,
        "crc32": "00000000",
    }
    with pytest.raises(ValueError, match="CRC-32"):
        extract_member(
            FakeSession(raw),
            "https://example.test/archive.zip",
            len(raw),
            member,
            [1024 * 1024],
            [0],
        )


def test_repetition_payload_timestamp_summary_checks_monotonicity_and_rate():
    from scripts.audit_robomnist_repetition_payload import summarize_timestamps

    result = summarize_timestamps(
        [{"time_stamp_ns": [0, 33_000_000, 66_000_000, 100_000_000]}]
    )
    assert len(result) == 1
    assert result[0]["count"] == 4
    assert result[0]["strictly_increasing"] is True
    assert result[0]["median_interval_ns"] == 33_000_000
    assert result[0]["duration_seconds"] == 0.1
    assert summarize_timestamps([{"time_stamp_ns": [0, 1, 1]}])[0][
        "strictly_increasing"
    ] is False
