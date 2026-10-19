from __future__ import annotations

import hashlib
import json
import subprocess
import sys


def test_partial_download_audit_reports_verified_identity_without_rehashing_media(
    tmp_path,
) -> None:
    selection = {
        "selection": {
            "all_selected_video_bytes": 4,
            "train": {
                "videos": [
                    {
                        "session_id": "session-a",
                        "offset_data": 123,
                        "size_bytes": 4,
                    }
                ]
            },
            "validation": {"videos": []},
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    media_root = tmp_path / "media"
    media_path = media_root / "train" / "session-a.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"data")
    record = {
        "archive_etag": '"pinned"',
        "archive_offset": 123,
        "path": str(media_path),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "session_id": "session-a",
        "sha256": "0" * 64,
        "size_bytes": 4,
        "split": "train",
    }
    journal = media_root / "download-progress.jsonl"
    journal.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.audit_holoassist_download_progress",
            "--selection",
            str(selection_path),
            "--media-root",
            str(media_root),
            "--journal",
            str(journal),
            "--archive-etag",
            '"pinned"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PASS_PARTIAL_HOLOASSIST_DOWNLOAD_AUDIT"
    assert report["completed_files"] == 1
    assert report["unlogged_files"] == 0
    assert report["completed_bytes"] == 4
    assert report["media_hashes_recomputed"] is False


def test_partial_download_audit_rejects_wrong_size_for_completed_media(tmp_path) -> None:
    selection = {
        "archive_etag": '"pinned"',
        "selection": {
            "all_selected_video_bytes": 4,
            "train": {
                "videos": [
                    {"session_id": "session-a", "offset_data": 123, "size_bytes": 4}
                ]
            },
            "validation": {"videos": []},
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    media_root = tmp_path / "media"
    media_path = media_root / "train" / "session-a.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"bad")
    record = {
        "archive_etag": '"pinned"',
        "archive_offset": 123,
        "path": str(media_path),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "session_id": "session-a",
        "sha256": "0" * 64,
        "size_bytes": 4,
        "split": "train",
    }
    journal = media_root / "download-progress.jsonl"
    journal.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.audit_holoassist_download_progress",
            "--selection",
            str(selection_path),
            "--media-root",
            str(media_root),
            "--journal",
            str(journal),
            "--archive-etag",
            '"pinned"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "wrong size" in result.stderr


def test_partial_download_audit_ignores_only_incomplete_tail_and_counts_partial_bytes(
    tmp_path,
) -> None:
    selection = {
        "archive_etag": '"pinned"',
        "selection": {
            "all_selected_video_bytes": 4,
            "train": {
                "videos": [
                    {"session_id": "session-a", "offset_data": 123, "size_bytes": 4}
                ]
            },
            "validation": {"videos": []},
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    media_root = tmp_path / "media"
    partial_path = media_root / "train" / "session-a.mp4.part"
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(b"ab")
    journal = media_root / "download-progress.jsonl"
    journal.write_bytes(b'{"partial":')

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.audit_holoassist_download_progress",
            "--selection",
            str(selection_path),
            "--media-root",
            str(media_root),
            "--journal",
            str(journal),
            "--archive-etag",
            '"pinned"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["incomplete_trailing_journal_line_ignored"] is True
    assert report["completed_files"] == 0
    assert report["partial_files"] == 1
    assert report["partial_bytes"] == 2
