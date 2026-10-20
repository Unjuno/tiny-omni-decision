from scripts.audit_holoassist_window_label_alignment import audit_split


def _fixture():
    annotation = {
        "taskId": "task-1",
        "video_name": "session-1",
        "taskType": "assemble",
        "events": [
            {
                "id": 10,
                "label": "Fine grained action",
                "start": 2.0,
                "end": 5.0,
                "attributes": {"Action Correctness": "Correct Action"},
            },
            {
                "id": 11,
                "label": "Fine grained action",
                "start": 8.0,
                "end": 12.0,
                "attributes": {
                    "Action Correctness": "Wrong Action, corrected by instructor verbally",
                    "Incorrect Action Corrected by": "21",
                },
            },
            {
                "id": 21,
                "label": "Conversation",
                "start": 11.0,
                "end": 13.0,
                "attributes": {
                    "Conversation Purpose": (
                        "instructor-start-conversation_correct the wrong action"
                    ),
                    "sequence": "10, 11, 12",
                    "Transcription": "No, that one.",
                },
            },
            {
                "id": 12,
                "label": "Fine grained action",
                "start": 31.0,
                "end": 34.0,
                "attributes": {"Action Correctness": "Wrong Action, not corrected"},
            },
        ],
    }
    event_rows = [
        {
            "id": "task-1:10",
            "session_id": "session-1",
            "split": "validation",
            "target": 0,
            "action_start_seconds": 2.0,
            "action_end_seconds": 5.0,
        },
        {
            "id": "task-1:11",
            "session_id": "session-1",
            "split": "validation",
            "target": 1,
            "action_start_seconds": 8.0,
            "action_end_seconds": 12.0,
        },
        {
            "id": "task-1:12",
            "session_id": "session-1",
            "split": "validation",
            "target": 1,
            "action_start_seconds": 31.0,
            "action_end_seconds": 34.0,
        },
    ]
    window_rows = [
        {
            "id": "session-1:window-0000",
            "session_id": "session-1",
            "split": "validation",
            "window_index": 0,
            "window_start_seconds": 0.0,
            "window_end_seconds": 30.0,
            "target": 1,
            "loss_mask": True,
            "event_ids": ["task-1:10", "task-1:11"],
            "action_count": 2,
            "correct_action_count": 1,
            "incorrect_action_count": 1,
        },
        {
            "id": "session-1:window-0001",
            "session_id": "session-1",
            "split": "validation",
            "window_index": 1,
            "window_start_seconds": 30.0,
            "window_end_seconds": 60.0,
            "target": 1,
            "loss_mask": True,
            "event_ids": ["task-1:12"],
            "action_count": 1,
            "correct_action_count": 0,
            "incorrect_action_count": 1,
        },
    ]
    return {"session-1": annotation}, event_rows, window_rows


def test_label_alignment_counts_verbal_correction_inside_window():
    annotations, events, windows = _fixture()

    result = audit_split("validation", annotations, events, windows)

    assert result["event_count"] == 3
    assert result["window_count"] == 2
    assert result["positive_window_count"] == 2
    assert result["positive_windows_with_linked_instructor_speech"] == 1
    assert result["positive_windows_with_explicit_correction_speech"] == 1
    assert result["wrong_action_subtypes"]["Wrong Action, corrected by instructor verbally"] == 1
    assert result["wrong_action_subtypes"]["Wrong Action, not corrected"] == 1


def test_correction_after_window_cutoff_is_not_counted_as_audio_cue():
    annotations, events, windows = _fixture()
    annotations["session-1"]["events"][2]["start"] = 30.2
    annotations["session-1"]["events"][2]["end"] = 31.0

    result = audit_split("validation", annotations, events, windows)

    assert result["positive_windows_with_linked_instructor_speech"] == 0


def test_label_mismatch_fails_closed():
    annotations, events, windows = _fixture()
    events[1]["target"] = 0

    try:
        audit_split("validation", annotations, events, windows)
    except ValueError as exc:
        assert "label differs" in str(exc)
    else:
        raise AssertionError("label mismatch was accepted")




def test_prediction_strata_separate_explicit_correction_cues():
    from scripts.audit_holoassist_window_label_alignment import summarize_prediction_strata

    predictions = [
        {"id": "with-cue", "target": 1, "prediction": 1, "probabilities": [0.1, 0.9]},
        {"id": "without-cue", "target": 1, "prediction": 0, "probabilities": [0.8, 0.2]},
        {"id": "negative", "target": 0, "prediction": 0, "probabilities": [0.9, 0.1]},
    ]

    result = summarize_prediction_strata(
        predictions, {"with-cue"}, {"with-cue"}
    )

    assert result["positive_with_explicit_correction_speech"]["n"] == 1
    assert result["positive_without_explicit_correction_speech"]["accuracy"] == 0.0


def test_saved_prediction_hash_mismatch_fails_closed():
    from scripts.audit_holoassist_window_label_alignment import _require_hash

    try:
        _require_hash("unexpected", "pinned", "prediction")
    except ValueError as exc:
        assert "hash differs" in str(exc)
    else:
        raise AssertionError("unrecognized prediction artifact was accepted")


def test_run_metadata_requires_completed_pinned_run_without_sealed_audit():
    import pytest

    from scripts.audit_holoassist_window_label_alignment import (
        CONFIG_SHA256,
        TRAINING_SOURCE_COMMIT,
        validate_run_metadata_document,
    )

    metadata = {
        "source_commit": TRAINING_SOURCE_COMMIT,
        "config_sha256": CONFIG_SHA256,
        "status": "complete",
        "sealed_audit_loaded": False,
    }
    validate_run_metadata_document(metadata)
    metadata["sealed_audit_loaded"] = True
    with pytest.raises(ValueError, match="fixed completed run"):
        validate_run_metadata_document(metadata)
