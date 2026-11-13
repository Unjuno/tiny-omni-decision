from tiny_omni_decision.perception_source_candidate import consolidate_source_candidate_records


def candidate(record_id, options, targets, *, interval=(0, 1_000_000), none=False):
    return {
        "record_id": record_id,
        "group_id": "video_1",
        "split": "train",
        "status": "candidate",
        "hold_reasons": [],
        "media_reference": {"video_id": "video_1"},
        "request": {
            "task": "audio_visual_source_candidate_reranking",
            "instruction": "Select every visual candidate that caused the audio event.",
            "audio_interval_us": list(interval),
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "frame_boxes": [{"frame_id": 2, "bbox_xyxy_normalized": box}],
                }
                for candidate_id, box in options
            ],
        },
        "supervision": {"target_candidate_ids": targets, "none_visible": none},
        "provenance": {
            "sound_annotation_id": int(record_id.rsplit(":", 1)[1]),
            "audio_interval_us": list(interval),
            "upstream_cut_frame": -1,
        },
    }


def test_identical_inputs_with_same_target_are_coalesced_without_reweighting():
    rows = [
        candidate("video_1:sound:0", [("option_0", [0.0, 0.0, 0.2, 0.2])], ["option_0"]),
        candidate("video_1:sound:1", [("option_9", [0.0, 0.0, 0.2, 0.2])], ["option_9"]),
    ]

    result = consolidate_source_candidate_records(rows)

    assert [row["status"] for row in result] == ["candidate", "duplicate"]
    assert result[0]["provenance"]["coalesced_sound_annotation_ids"] == [0, 1]
    assert result[1]["provenance"]["coalesced_into_record_id"] == "video_1:sound:0"
    assert "request" not in result[1]


def test_contradictory_identical_inputs_are_held_instead_of_training_on_conflict():
    rows = [
        candidate("video_1:sound:0", [("option_0", [0.0, 0.0, 0.2, 0.2])], ["option_0"]),
        candidate(
            "video_1:sound:1",
            [("option_9", [0.0, 0.0, 0.2, 0.2])],
            [],
            none=True,
        ),
    ]

    result = consolidate_source_candidate_records(rows)

    assert all(row["status"] == "held" for row in result)
    assert all("contradictory_identical_model_input" in row["hold_reasons"] for row in result)
    assert all("request" not in row for row in result)


def test_coincident_unresolved_annotation_holds_candidate_for_same_interval():
    rows = [candidate("video_1:sound:0", [("option_0", [0.0, 0.0, 0.2, 0.2])], ["option_0"])]
    rows.append(
        {
            "record_id": "video_1:sound:1",
            "group_id": "video_1",
            "split": "train",
            "status": "held",
            "hold_reasons": ["unknown_or_empty_source_annotation"],
            "provenance": {"audio_interval_us": [0, 1_000_000], "sound_annotation_id": 1},
        }
    )

    result = consolidate_source_candidate_records(rows)

    assert result[0]["status"] == "held"
    assert "coincident_unresolved_sound_annotation" in result[0]["hold_reasons"]


def test_partially_targeted_identical_candidate_crops_are_held():
    box = [0.1, 0.1, 0.4, 0.4]
    row = candidate(
        "video_1:sound:0",
        [("option_0", box), ("option_1", box)],
        ["option_0"],
    )

    result = consolidate_source_candidate_records([row])

    assert result[0]["status"] == "held"
    assert "indistinguishable_positive_candidate_crops" in result[0]["hold_reasons"]
