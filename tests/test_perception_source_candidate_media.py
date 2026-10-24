from scripts.preflight_perception_source_candidate_media import (
    audio_interval_coverage,
    select_smoke_records,
)


def record(video_id, sound_id, *, none_visible=False, cut_frame=-1):
    return {
        "status": "candidate",
        "record_id": f"{video_id}:sound:{sound_id}",
        "group_id": video_id,
        "media_reference": {"video_id": video_id},
        "supervision": {"none_visible": none_visible},
        "provenance": {"upstream_cut_frame": cut_frame},
    }


def test_smoke_selection_is_seeded_group_disjoint_and_media_bounded():
    records = [record(f"v{i}", j) for i in range(5) for j in range(2)]
    media_ids = {f"v{i}" for i in range(4)}
    selected_a = select_smoke_records(records, media_ids, limit=3, seed=17)
    selected_b = select_smoke_records(records, media_ids, limit=3, seed=17)
    assert [item["record_id"] for item in selected_a] == [
        item["record_id"] for item in selected_b
    ]
    assert len(selected_a) == 3
    assert len({item["group_id"] for item in selected_a}) == 3
    assert all(item["group_id"] in media_ids for item in selected_a)


def test_audio_interval_coverage_clips_and_unions_decoded_frames():
    coverage = audio_interval_coverage(
        [100_000, 500_000], [(0.0, 0.3), (0.25, 0.6), (0.8, 1.0)]
    )
    assert coverage == 0.4


def test_audio_interval_coverage_reports_missing_samples():
    coverage = audio_interval_coverage([0, 1_000_000], [(0.0, 0.4), (0.6, 1.0)])
    assert coverage == 0.8


def test_smoke_selection_can_require_none_visible_and_cut_frame_groups():
    records = [
        record("visible", 0),
        record("none", 0, none_visible=True),
        record("cut", 0, cut_frame=25),
        record("other", 0),
    ]
    selected = select_smoke_records(
        records,
        {"visible", "none", "cut", "other"},
        limit=3,
        seed=17,
        require_none_visible=1,
        require_cut_frame=1,
    )
    assert any(item["supervision"]["none_visible"] for item in selected)
    assert any(item["provenance"]["upstream_cut_frame"] >= 0 for item in selected)
    assert len({item["group_id"] for item in selected}) == 3
