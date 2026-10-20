import hashlib
import json

import pytest

from scripts.acquire_perception_av_event_scale import (
    SPLIT_COUNTS,
    RetryingRemoteZip,
    reconcile_existing_media,
    validate_plan,
)
from scripts.plan_perception_av_event_media import TRAIN_ARCHIVE_PIN


def _make_plan(path):
    rows = []
    summary = {}
    for split, count in SPLIT_COUNTS.items():
        summary[split] = {"video_count": count, "candidate_events": count}
        for index in range(count):
            video_id = f"{split}_{index}"
            rows.append(
                {
                    "video_id": video_id,
                    "model_split": split,
                    "metadata": {"split": "train", "video_id": video_id},
                    "member": f"videos/{video_id}.mp4",
                    "candidate_events": 1,
                    "bytes": 100,
                }
            )
    plan = {
        "status": "PASS_FRESH_VIDEO_SCALE_PLAN",
        "training": False,
        "test_opened": False,
        "archive_pin": TRAIN_ARCHIVE_PIN,
        "rows": rows,
        "split_summary": summary,
        "candidate_events": len(rows),
        "total_uncompressed_bytes": 100 * len(rows),
    }
    path.write_text(json.dumps(plan), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest(), plan


def test_acquisition_plan_checks_exact_video_and_event_counts(tmp_path):
    path = tmp_path / "plan.json"
    digest, plan = _make_plan(path)
    assert validate_plan(path, digest)["candidate_events"] == len(plan["rows"])


def test_acquisition_plan_rejects_split_or_duplicate_id_drift(tmp_path):
    path = tmp_path / "plan.json"
    _, plan = _make_plan(path)
    plan["rows"][0]["model_split"] = "selection"
    path.write_text(json.dumps(plan), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="split sizes differ"):
        validate_plan(path, digest)


def test_resume_reconciles_unflushed_files_and_preserves_partials(tmp_path):
    complete_bytes = b"complete media"
    complete = tmp_path / "video_complete.mp4"
    complete.write_bytes(complete_bytes)
    partial = tmp_path / "video_partial.mp4"
    partial.write_bytes(b"short")
    rows = [
        {
            "video_id": "video_complete",
            "model_split": "fit",
            "bytes": len(complete_bytes),
            "crc32": 1,
            "member": "videos/video_complete.mp4",
        },
        {
            "video_id": "video_partial",
            "model_split": "fit",
            "bytes": 100,
            "crc32": 2,
            "member": "videos/video_partial.mp4",
        },
    ]
    clips, partials = reconcile_existing_media(tmp_path, rows, [], set())
    assert [row["video_id"] for row in clips] == ["video_complete"]
    assert partials == [{"name": "video_partial.mp4.partial-01", "bytes": 5}]
    assert (tmp_path / "video_partial.mp4.partial-01").read_bytes() == b"short"


def test_remote_zip_retries_transient_tls_failure_without_changing_range(monkeypatch):
    calls = []

    class Response:
        status_code = 206
        content = b"abc"
        headers = {"Content-Range": "bytes 0-2/3"}

        def raise_for_status(self):
            return None

    def get(url, headers, timeout):
        calls.append(headers["Range"])
        if len(calls) == 1:
            import requests

            raise requests.exceptions.SSLError("transient EOF")
        return Response()

    monkeypatch.setattr("scripts.acquire_perception_av_event_scale.requests.get", get)
    monkeypatch.setattr("scripts.acquire_perception_av_event_scale.time.sleep", lambda _: None)
    remote = RetryingRemoteZip("https://example.invalid/media.zip", 3, '"pin"')
    assert remote.read(3) == b"abc"
    assert calls == ["bytes=0-2", "bytes=0-2"]
    assert remote.tell() == 3
    assert remote.retries == 1
