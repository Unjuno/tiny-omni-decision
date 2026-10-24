from scripts.audit_perception_source_candidate_identity import audit_records


def record(record_id, options, targets, *, interval=(0, 1_000_000), none=False):
    return {
        "record_id": record_id,
        "status": "candidate",
        "media_reference": {"video_id": "video_1"},
        "request": {
            "task": "audio_visual_source_candidate_reranking",
            "instruction": "Select every visual candidate that caused the audio event.",
            "audio_interval_us": list(interval),
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "frame_boxes": [
                        {"frame_id": 5, "bbox_xyxy_normalized": box},
                    ],
                }
                for candidate_id, box in options
            ],
        },
        "supervision": {
            "target_candidate_ids": targets,
            "none_visible": none,
        },
    }


def test_identity_audit_maps_permuted_option_ids_to_the_same_visual_inputs():
    rows = [
        record(
            "a",
            [("option_0", [0.0, 0.0, 0.2, 0.2]), ("option_1", [0.8, 0.8, 1.0, 1.0])],
            ["option_0"],
        ),
        record(
            "b",
            [("option_0", [0.8, 0.8, 1.0, 1.0]), ("option_1", [0.0, 0.0, 0.2, 0.2])],
            ["option_0"],
        ),
    ]

    report = audit_records(rows)

    assert report["unique_model_inputs"] == 1
    assert report["contradictory_input_groups"] == 1
    assert report["contradictory_records"] == 2


def test_identity_audit_does_not_flag_identical_targets_or_different_inputs():
    first = record("a", [("option_0", [0.0, 0.0, 0.2, 0.2])], ["option_0"])
    same = record("b", [("option_1", [0.0, 0.0, 0.2, 0.2])], ["option_1"])
    different = record(
        "c",
        [("option_0", [0.0, 0.0, 0.2, 0.2])],
        ["option_0"],
        interval=(1_000_000, 2_000_000),
    )

    report = audit_records([first, same, different])

    assert report["unique_model_inputs"] == 2
    assert report["contradictory_input_groups"] == 0


def test_identity_audit_detects_positive_targets_indistinguishable_by_crop_boxes():
    row = record(
        "a",
        [("option_0", [0.0, 0.0, 0.2, 0.2]), ("option_1", [0.0, 0.0, 0.2, 0.2])],
        ["option_0"],
    )

    report = audit_records([row])

    assert report["partial_positive_crop_records"] == 1
