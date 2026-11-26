import numpy as np
import pytest

from tiny_omni_decision.perception_visibility_validation import (
    build_validation_manifest,
    paired_metric_bootstrap,
)


def fixed_inputs():
    events = []
    media_rows = []
    for index in range(200):
        video_id = f"video_{index:04d}"
        target = index // 100
        start = 0.4 if index == 0 else 4.0 + index / 1000
        end = start + 0.6
        events.append(
            {
                "split": "valid",
                "video_id": video_id,
                "event_id": index,
                "target": target,
                "start_event_s": start,
                "end_event_s": end,
                "audio_duration_s": 20.0,
                "video_duration_s": 20.0,
            }
        )
        media_rows.append(
            {
                "video_id": video_id,
                "target": target,
                "audio_sha256": f"a{index}",
                "video_sha256": f"v{index}",
                "audio_duration_s": 20.0,
                "video_duration_s": 20.0,
                "event_start_s": start,
                "event_end_s": end,
            }
        )
    selection = {
        "status": "PREFLIGHT_ONLY_NOT_EVALUATED",
        "source_split": "official validation only",
        "official_test_split_opened": False,
        "events": events,
    }
    media = {
        "status": "PASS_MEDIA_HASH_AND_DECODE_METADATA",
        "selection_sha256": "selection-hash",
        "examples_manifest": media_rows,
    }
    return selection, media


def test_validation_manifest_keeps_all_groups_and_records_boundary_shift():
    selection, media = fixed_inputs()
    result = build_validation_manifest(
        selection,
        media,
        selection_sha256="selection-hash",
        media_validation_sha256="media-hash",
    )

    assert len(result["validation"]) == 200
    assert result["train"] == []
    assert result["metadata"]["boundary_shift_count"] == 1
    assert result["validation"][0]["video_window_s"] == [0.0, 2.0]
    assert len(result["validation"][0]["video_frame_times_s"]) == 8
    assert result["metadata"]["labels_used_for_selection"] is True
    assert result["metadata"]["validation_is_blind"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_split", "train", "official validation cohort required"),
        ("official_test_split_opened", True, "official test must remain unopened"),
        ("status", "PASS_EVALUATED", "frozen preflight cohort"),
    ],
)
def test_validation_manifest_rejects_wrong_or_opened_cohort(field, value, message):
    selection, media = fixed_inputs()
    selection[field] = value
    with pytest.raises(ValueError, match=message):
        build_validation_manifest(
            selection,
            media,
            selection_sha256="selection-hash",
            media_validation_sha256="media-hash",
        )


def test_validation_manifest_rejects_media_identity_or_target_mismatch():
    selection, media = fixed_inputs()
    media["examples_manifest"][0]["target"] = 1
    with pytest.raises(ValueError, match="media target differs"):
        build_validation_manifest(
            selection,
            media,
            selection_sha256="selection-hash",
            media_validation_sha256="media-hash",
        )


def test_paired_metric_bootstrap_is_deterministic_and_reports_b_minus_a():
    targets = np.asarray([0, 1, 0, 1])
    weaker = np.asarray([-0.2, -0.1, 0.3, 0.2])
    stronger = np.asarray([-1.0, 1.0, -0.5, 0.8])
    first = paired_metric_bootstrap(weaker, stronger, targets, repetitions=100)
    second = paired_metric_bootstrap(weaker, stronger, targets, repetitions=100)

    assert first == second
    assert first["delta_b_minus_a"]["accuracy"] == 0.5
    assert first["delta_b_minus_a"]["nll"] < 0
    assert first["bootstrap_repetitions"] == 100
