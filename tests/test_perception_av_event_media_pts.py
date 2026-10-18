import pytest

from scripts.audit_perception_av_event_media_pts import (
    event_window_from_pts,
    load_pinned_media_manifest,
    overlap_samples,
)


def test_inclusive_event_end_uses_next_video_pts_as_exclusive_boundary():
    start, end = event_window_from_pts(
        [0.0, 0.1, 0.23, 0.34], [0.1, 0.13, 0.11, 0.1], [1, 2]
    )
    assert start == pytest.approx(0.1)
    assert end == pytest.approx(0.34)


def test_event_ending_on_last_decoded_frame_uses_its_duration():
    start, end = event_window_from_pts(
        [0.0, 0.1, 0.23], [0.1, 0.13, 0.11], [1, 2]
    )
    assert start == pytest.approx(0.1)
    assert end == pytest.approx(0.34)


def test_bad_or_missing_frame_duration_fails_closed():
    with pytest.raises(ValueError, match="outside decoded video"):
        event_window_from_pts([0.0], [0.1], [1, 1])
    with pytest.raises(ValueError, match="no duration"):
        event_window_from_pts([0.0, 0.1], [0.1, None], [1, 1])


def test_audio_window_maps_to_clamped_sample_offsets():
    assert overlap_samples(1.0, 2.0, 0.0, 1.95, 100) == (100, 195, 0.95)


def test_no_audio_overlap_is_explicit():
    assert overlap_samples(1.0, 2.0, 0.0, 0.5, 100) == (0, 0, 0.0)


def test_media_manifest_snapshot_returns_hash_of_exact_bytes(tmp_path):
    import hashlib
    import json

    from scripts.audit_perception_av_event_media_pts import PLAN_SHA256

    raw = json.dumps(
        {
            "split": "train",
            "plan_sha256": PLAN_SHA256,
            "status": "RUNNING",
            "clips": [{"video_id": "video_1"}],
        },
        separators=(",", ":"),
    ).encode()
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)

    manifest, digest = load_pinned_media_manifest(path, "train", {"video_1"})

    assert manifest["clips"][0]["video_id"] == "video_1"
    assert digest == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("clips", "planned", "error"),
    [
        ([{"video_id": "video_1"}, {"video_id": "video_1"}], {"video_1"}, "duplicate"),
        ([{"video_id": "video_outside_plan"}], {"video_1"}, "unplanned"),
        ([{}], {"video_1"}, "missing/invalid"),
    ],
)
def test_media_manifest_rejects_invalid_video_inventory(tmp_path, clips, planned, error):
    import json

    from scripts.audit_perception_av_event_media_pts import PLAN_SHA256

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "split": "train",
                "plan_sha256": PLAN_SHA256,
                "status": "RUNNING",
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        load_pinned_media_manifest(path, "train", planned)
