import hashlib
import json
import struct
import zlib
from types import SimpleNamespace

import pytest

from scripts import acquire_fsd50k_annotation_manifest as acquisition
from scripts.build_fsd50k_annotation_decision_manifest import (
    DATASET_REVISION,
    TARGET_LABELS,
    canonical_sha256,
    summarize_split,
)


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


def test_range_reader_retries_short_http_body_without_advancing_position(monkeypatch):
    class Response:
        status = 206

        def __init__(self, payload, content_range):
            self.payload = payload
            self.headers = {"Content-Range": content_range}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return self.payload

    payload = b"abcdefgh"
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 90
        assert request.get_header("Accept-encoding") == "identity"
        assert request.get_header("Range") == "bytes=0-7"
        content_range = "bytes 0-7/16"
        return Response(payload[:3] if calls == 1 else payload, content_range)

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _seconds: None)
    budget = acquisition.SharedBudget(100)
    reader = acquisition.BudgetedRangeReader("https://example.invalid/archive", 16, budget)

    assert reader.read(8) == payload
    assert calls == 2
    assert reader.position == 8
    assert reader.transferred == 11
    assert budget.requested == 16


def test_range_reader_splits_persistently_short_large_range(monkeypatch):
    class Response:
        status = 206

        def __init__(self, payload, content_range):
            self.payload = payload
            self.headers = {"Content-Range": content_range}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return self.payload

    payload = bytes(index % 251 for index in range(300_000))
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 90
        header = request.get_header("Range")
        start, end = map(int, header.removeprefix("bytes=").split("-"))
        expected = end - start + 1
        content_range = f"bytes {start}-{end}/300000"
        if expected > acquisition.SHORT_RESPONSE_FALLBACK_CHUNK_BYTES:
            return Response(payload[start : start + expected - 50], content_range)
        return Response(payload[start : end + 1], content_range)

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _seconds: None)
    budget = acquisition.SharedBudget(10_000_000)
    reader = acquisition.BudgetedRangeReader(
        "https://example.invalid/archive", len(payload), budget
    )

    assert reader.read(len(payload)) == payload
    assert calls == 6 + 3
    assert reader.position == len(payload)
    assert reader.transferred == 6 * (len(payload) - 50) + len(payload)
    assert budget.requested == 6 * len(payload) + len(payload)


def test_failed_partial_is_verified_before_fresh_attempt_resume(tmp_path):
    preflight = tmp_path / "preflight"
    partial = tmp_path / "attempt-03"
    preflight.mkdir()
    partial.mkdir()
    splits = {}
    for split, offset in (("train", 0), ("calibration", 2), ("validation", 4)):
        rows = []
        for sample_id in range(offset, offset + 2):
            rows.append(
                {
                    "sample_id": f"{sample_id:06d}",
                    "source_id": f"fsd:{sample_id}",
                    "group_id": f"uploader-{sample_id}",
                    "split": split,
                    "license": "http://creativecommons.org/licenses/by/3.0/",
                    "targets": {
                        label: bool((sample_id + i) % 2)
                        for i, label in enumerate(TARGET_LABELS)
                    },
                }
            )
        splits[split] = rows
        (preflight / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    summaries = {split: summarize_split(rows) for split, rows in splits.items()}
    source_sha = {"fixture": "a" * 64}
    manifest_sha = canonical_sha256({"splits": summaries, "source_sha256": source_sha})
    (preflight / "preflight.json").write_text(
        json.dumps(
            {
                "status": "PREFLIGHT_PASS",
                "dataset_revision": DATASET_REVISION,
                "manifest_sha256": manifest_sha,
                "source_sha256": source_sha,
                "splits": summaries,
                "official_eval_labels_read": False,
            }
        ),
        encoding="utf-8",
    )

    row = splits["train"][0]
    payload = b"verified wav payload"
    media = partial / "train" / f"{row['sample_id']}.wav"
    media.parent.mkdir()
    media.write_bytes(payload)
    receipt = {
        **{key: row[key] for key in ("sample_id", "source_id", "group_id", "split", "license")},
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "zip_crc32": zlib.crc32(payload),
    }
    receipt_path = partial / "files.jsonl"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    (partial / "result.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "error": "ValueError('range length mismatch')",
                "runner_sha256": (
                    "784687424d0720626e89a284a4a9546e585b75e977d5321664d5c022dbe7a71d"
                ),
                "dataset_revision": DATASET_REVISION,
                "preflight_manifest_sha256": manifest_sha,
                "archive_directory_sha256": (
                    "a52c8e77acf8c01e5108a3872461a65f2a583b2ddc3e0bb6554e51dc7ae12c5f"
                ),
                "official_eval_labels_read": False,
                "official_eval_media_acquired": False,
                "expected_files": 6,
                "downloaded_files": 1,
                "output_bytes": len(payload),
                "files_jsonl_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "range_bytes": 100,
            }
        ),
        encoding="utf-8",
    )

    prior, records = acquisition.validate_failed_partial(preflight, partial)
    assert prior["downloaded_files"] == 1
    assert records == [receipt]

    (partial / "progress.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "runner_sha256": acquisition.INTERRUPTED_ATTEMPT_RUNNER_SHA256,
                "dataset_revision": DATASET_REVISION,
                "preflight_manifest_sha256": manifest_sha,
                "archive_directory_sha256": (
                    "a52c8e77acf8c01e5108a3872461a65f2a583b2ddc3e0bb6554e51dc7ae12c5f"
                ),
                "official_eval_labels_read": False,
                "official_eval_media_acquired": False,
                "expected_files": 6,
                "downloaded_files": 1,
                "resumed_files": 0,
                "resumed_range_bytes": 0,
                "range_bytes": 100,
                "range_transfer_budget_bytes": 1000,
                "output_bytes": len(payload),
            }
        ),
        encoding="utf-8",
    )
    attempt_04_result = json.loads((partial / "result.json").read_text(encoding="utf-8"))
    attempt_04_result.update(
        runner_sha256=(
            "8b7f5b08aecb109db6072f5bcc035e42defff57defa875239c1d62568ef0ad4b"
        ),
        error=(
            'ValueError("range request failed after 6 attempts at bytes=10-19: '
            "ValueError('short range response: got=5, expected=10, "
            'start=10, end=19, attempt=6\')")'
        ),
    )
    (partial / "result.json").write_text(json.dumps(attempt_04_result), encoding="utf-8")
    prior, records = acquisition.validate_failed_partial(preflight, partial)
    assert prior["downloaded_files"] == 1
    assert records == [receipt]

    orphan = partial / "train" / "000001.wav"
    orphan.write_bytes(b"valid-looking but uncommitted")
    interrupted, committed_records = acquisition.validate_interrupted_partial(
        preflight, partial
    )
    assert interrupted["status"] == "INTERRUPTED_PREFIX_VERIFIED"
    assert interrupted["downloaded_files"] == 1
    assert interrupted["range_bytes"] == 100
    assert committed_records == [receipt]
    assert orphan.exists()

    media.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="failed integrity check"):
        acquisition.validate_failed_partial(preflight, partial)
    with pytest.raises(ValueError, match="failed integrity check"):
        acquisition.validate_interrupted_partial(preflight, partial)
