import pytest

from tiny_omni_decision.perception_visibility_transfer import (
    build_inference_manifest,
    merge_media_receipts,
    select_unseen_events,
)


def video(video_id, targets):
    return {
        "metadata": {
            "video_id": video_id,
            "audio_samples": 10_000,
            "audio_sample_rate": 1_000,
            "num_frames": 100,
            "frame_rate": 10,
        },
        "sound_localisation": [
            {
                "id": index,
                "timestamps": [1_000_000 + index * 10_000, 1_100_000 + index * 10_000],
                "is_visible": target,
            }
            for index, target in enumerate(targets)
        ],
    }


def test_selects_deterministic_unseen_train_videos_with_one_row_per_group():
    annotations = {
        "fit-visible": video("fit-visible", [1]),
        "fit-hidden": video("fit-hidden", [0]),
        "new-v1": video("new-v1", [1, 0]),
        "new-v2": video("new-v2", [0, 1]),
        "new-v3": video("new-v3", [1]),
        "new-v4": video("new-v4", [1]),
        "new-v5": video("new-v5", [1]),
        "new-v6": video("new-v6", [0]),
        "new-v7": video("new-v7", [0]),
        "new-v8": video("new-v8", [0]),
    }

    first = select_unseen_events(
        annotations,
        excluded_video_ids={"fit-visible", "fit-hidden"},
        source_split="train",
        seed=17,
        events_per_class=2,
    )
    second = select_unseen_events(
        annotations,
        excluded_video_ids={"fit-visible", "fit-hidden"},
        source_split="train",
        seed=17,
        events_per_class=2,
    )

    assert first == second
    assert len(first) == 4
    assert {row["target"] for row in first} == {0, 1}
    assert len({row["video_id"] for row in first}) == 4
    assert not ({row["video_id"] for row in first} & {"fit-visible", "fit-hidden"})
    assert all(row["split"] == "train_transfer" for row in first)


def test_rejects_nontraining_source_split():
    with pytest.raises(ValueError, match="train split only"):
        select_unseen_events(
            {"v": video("v", [1])},
            excluded_video_ids=set(),
            source_split="valid",
            seed=17,
            events_per_class=1,
        )


def test_fails_closed_when_a_class_has_too_few_unseen_video_groups():
    annotations = {"only-visible": video("only-visible", [1])}
    with pytest.raises(ValueError, match="insufficient unseen groups"):
        select_unseen_events(
            annotations,
            excluded_video_ids=set(),
            source_split="train",
            seed=17,
            events_per_class=1,
        )


def test_rejects_invalid_visibility_labels_instead_of_silently_dropping_them():
    annotations = {"v": video("v", [2])}
    with pytest.raises(ValueError, match="binary visibility target"):
        select_unseen_events(
            annotations,
            excluded_video_ids=set(),
            source_split="train",
            seed=17,
            events_per_class=1,
        )


def test_inference_manifest_preserves_train_only_fixed_windows():
    selected = [
        {
            "split": "train_transfer",
            "video_id": "video_100",
            "event_id": 7,
            "target": 1,
            "start_event_s": 5.0,
            "end_event_s": 6.0,
        }
    ]
    result = build_inference_manifest(selected)
    row = result["train"][0]
    assert set(result) == {"train", "validation"}
    assert result["validation"] == []
    assert row["split"] == "train"
    assert row["target"] == 1
    assert row["video_window_s"] == [4.5, 6.5]
    assert len(row["video_frame_times_s"]) == 8
    assert row["video_frame_times_s"][0] == 4.5
    assert row["video_frame_times_s"][-1] == 6.5
    assert row["options"][0]["audio_window_s"] == [4.5, 6.5]


def test_inference_manifest_rejects_non_train_and_duplicate_groups():
    row = {
        "split": "validation",
        "video_id": "video_100",
        "event_id": 7,
        "target": 1,
        "start_event_s": 5.0,
        "end_event_s": 6.0,
    }
    with pytest.raises(ValueError, match="train-only"):
        build_inference_manifest([row])
    row["split"] = "train_transfer"
    with pytest.raises(ValueError, match="one event per video"):
        build_inference_manifest([row, row | {"event_id": 8}])


def test_inference_manifest_rejects_event_without_full_two_second_context():
    row = {
        "split": "train_transfer",
        "video_id": "video_100",
        "event_id": 7,
        "target": 1,
        "start_event_s": 0.2,
        "end_event_s": 0.8,
    }
    with pytest.raises(ValueError, match="full centered 2-second"):
        build_inference_manifest([row])


def test_merge_media_receipts_prefers_new_verified_rows_and_requires_both_modalities():
    prior = [
        {"video_id": "v1", "member": "videos/v1.mp4", "sha256": "old-v"},
        {"video_id": "v1", "member": "audios/v1.wav", "sha256": "old-a"},
    ]
    new = [{"video_id": "v1", "member": "videos/v1.mp4", "sha256": "new-v"}]
    merged = merge_media_receipts({"v1"}, prior, new)
    assert [row["sha256"] for row in merged] == ["new-v", "old-a"]
    with pytest.raises(ValueError, match="missing verified mp4 receipt"):
        merge_media_receipts({"v2"}, [], new)
