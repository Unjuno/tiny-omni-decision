from scripts.preflight_perception_source_candidate_media import (
    audio_interval_coverage,
    select_smoke_records,
)


def record(video_id, sound_id):
    return {
        "status": "candidate",
        "record_id": f"{video_id}:sound:{sound_id}",
        "group_id": video_id,
        "media_reference": {"video_id": video_id},
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
