import hashlib

import pytest

from scripts.download_adl_archives import (
    archive_jobs,
    download_one,
    hash_archive,
    verify_receipt_downloads,
)


def source_rows():
    return [
        {
            "key": f"{index:02d}_activity.zip",
            "size": index + 1,
            "checksum": "md5:" + "0" * 32,
            "links": {"self": f"https://example.invalid/{index:02d}.zip"},
        }
        for index in range(25)
    ]


def test_archive_jobs_pins_all_25_indices_and_sorts_by_size():
    rows = source_rows()
    rows[0]["size"], rows[1]["size"] = rows[1]["size"], rows[0]["size"]
    jobs = archive_jobs(rows)
    assert len(jobs) == 25
    assert jobs[0]["key"] == "01_activity.zip"
    assert jobs[-1]["key"] == "24_activity.zip"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.__setitem__(0, {**rows[0], "key": "25_activity.zip"}),
        lambda rows: rows[0].__setitem__("checksum", "md5:not-a-digest"),
        lambda rows: rows[0]["links"].__setitem__("self", "http://example.invalid/a.zip"),
    ],
)
def test_archive_jobs_reject_incomplete_or_untrusted_metadata(mutate):
    rows = source_rows()
    mutate(rows)
    with pytest.raises(ValueError):
        archive_jobs(rows)


def test_hash_archive_checks_exact_size_and_publisher_md5(tmp_path):
    payload = b"pinned archive payload"
    path = tmp_path / "archive.zip"
    path.write_bytes(payload)
    digest = hashlib.md5(payload).hexdigest()
    result = hash_archive(path, len(payload), f"md5:{digest}")
    assert result["bytes"] == len(payload)
    assert result["publisher_md5"] == digest
    assert len(result["sha256"]) == 64
    with pytest.raises(ValueError, match="size mismatch"):
        hash_archive(path, len(payload) + 1, f"md5:{digest}")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        hash_archive(path, len(payload), "md5:" + "0" * 32)


class FakeResponse:
    def __init__(self, status_code, headers, body):
        self.status_code = status_code
        self.headers = headers
        self.body = body

    def raise_for_status(self):
        return None

    def iter_content(self, _chunk_size):
        yield self.body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.call = None

    def get(self, url, **kwargs):
        self.call = (url, kwargs)
        return self.response


def test_download_one_resumes_only_exact_range_and_rehashes_full_archive(tmp_path):
    payload = b"an archive split across two requests"
    offset = 13
    part = tmp_path / "00_activity.zip.part"
    part.write_bytes(payload[:offset])
    md5 = hashlib.md5(payload).hexdigest()
    session = FakeSession(
        FakeResponse(
            206,
            {
                "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(payload) - offset),
            },
            payload[offset:],
        )
    )
    row = {
        "key": "00_activity.zip",
        "size": len(payload),
        "checksum": f"md5:{md5}",
        "links": {"self": "https://example.invalid/00.zip"},
    }

    result = download_one(session, row, tmp_path)

    target = tmp_path / row["key"]
    assert target.read_bytes() == payload
    assert not part.exists()
    assert result["bytes"] == len(payload)
    assert result["resumed_bytes"] == offset
    assert result["publisher_md5"] == md5
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert session.call[1]["headers"] == {"Range": f"bytes={offset}-"}


def test_download_one_preserves_partial_when_server_ignores_range(tmp_path):
    payload = b"an archive split across two requests"
    offset = 13
    part = tmp_path / "00_activity.zip.part"
    part.write_bytes(payload[:offset])
    row = {
        "key": part.name.removesuffix(".part"),
        "size": len(payload),
        "checksum": f"md5:{hashlib.md5(payload).hexdigest()}",
        "links": {"self": "https://example.invalid/00.zip"},
    }
    session = FakeSession(FakeResponse(200, {"Content-Length": str(len(payload))}, payload))

    with pytest.raises(ValueError, match="did not honor exact resume range"):
        download_one(session, row, tmp_path)

    assert part.read_bytes() == payload[:offset]


def test_verify_receipt_downloads_rehashes_completed_archives(tmp_path):
    payload = b"completed archive"
    key = "00_activity.zip"
    path = tmp_path / key
    path.write_bytes(payload)
    digest = hashlib.md5(payload).hexdigest()
    row = {"key": key, "size": len(payload), "checksum": f"md5:{digest}"}
    item = {
        "key": key,
        "bytes": len(payload),
        "publisher_md5": digest,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    assert verify_receipt_downloads(tmp_path, [item], {key: row}) == {key}
    path.write_bytes(b"completed archivE")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_receipt_downloads(tmp_path, [item], {key: row})
