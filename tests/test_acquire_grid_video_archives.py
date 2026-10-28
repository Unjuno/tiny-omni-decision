import argparse
import hashlib
import json

import pytest

from scripts.acquire_grid_video_archives import (
    download_one,
    load_inventory,
    parse_existing_archive,
    validate_existing_archive,
)


def _summary(tmp_path):
    receipts = {}
    for index in range(32):
        speaker = f"s{index + 1}"
        receipts[speaker] = {
            "video_url": f"https://zenodo.org/api/records/3625687/files/{speaker}.zip/content",
            "video_archive_bytes": 4,
            "video_publisher_md5": "md5:098f6bcd4621d373cade4e832627b4f6",
        }
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps({"status": "PASS_DIRECT_SOURCE_SPLIT_BUILT", "source_receipts": receipts}),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_load_inventory_requires_pinned_complete_official_inventory(tmp_path):
    path, digest = _summary(tmp_path)

    actual, rows = load_inventory(path, digest)

    assert actual == digest
    assert len(rows) == 32
    assert rows[0]["speaker"] == "s1"
    assert rows[-1]["speaker"] == "s9"
    assert rows[0]["publisher_md5"] == "098f6bcd4621d373cade4e832627b4f6"


def test_load_inventory_rejects_hash_and_url_mismatches(tmp_path):
    path, digest = _summary(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_inventory(path, "0" * 64)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_receipts"]["s1"]["video_url"] = "https://example.com/s1.zip"
    path.write_text(json.dumps(payload), encoding="utf-8")
    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unexpected official video URL"):
        load_inventory(path, changed_digest)


def test_existing_archive_requires_exact_size_and_publisher_md5(tmp_path):
    path, digest = _summary(tmp_path)
    _, rows = load_inventory(path, digest)
    archive = tmp_path / "s1.zip"
    archive.write_bytes(b"test")

    receipt = validate_existing_archive(rows[0], archive)
    assert receipt["sha256"] == hashlib.sha256(b"test").hexdigest()
    assert receipt["verified_by"] == "existing_publisher_md5_and_sha256"

    archive.write_bytes(b"bad!")
    with pytest.raises(ValueError, match="publisher MD5 mismatch"):
        validate_existing_archive(rows[0], archive)


def test_download_one_checks_range_stream_hash_and_publisher_md5(tmp_path, monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __init__(self):
            self.payload = b"test"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            payload, self.payload = self.payload, b""
            return payload

    monkeypatch.setattr(
        "scripts.acquire_grid_video_archives.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    item = {
        "speaker": "s1",
        "url": "https://zenodo.org/api/records/3625687/files/s1.zip/content",
        "expected_bytes": 4,
        "publisher_md5": "098f6bcd4621d373cade4e832627b4f6",
    }

    result = download_one(item, tmp_path, tmp_path / "progress.jsonl")

    assert result["sha256"] == hashlib.sha256(b"test").hexdigest()
    assert (tmp_path / "s1.zip").read_bytes() == b"test"
    assert not (tmp_path / "s1.zip.part").exists()


def test_existing_archive_argument_requires_speaker_equals_path(tmp_path):
    assert parse_existing_archive(f"s30={tmp_path / 's30.zip'}") == (
        "s30",
        (tmp_path / "s30.zip").resolve(),
    )
    with pytest.raises(argparse.ArgumentTypeError):
        parse_existing_archive("missing-separator")
