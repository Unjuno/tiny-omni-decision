from __future__ import annotations

from tiny_omni_decision.audio_precision import _compare, _request_fields


def test_request_projection_ignores_targets_and_labels() -> None:
    source = {
        "id": "speaker-chapter-utterance",
        "audio_path": "LibriSpeech/dev-clean/speaker/chapter/sample.flac",
        "audio_sha256": "a" * 64,
        "duration_seconds": 4.5,
        "options_by_condition": {
            "random_distractors": ["a", "b", "c", "d"],
            "hard_distractors": ["e", "f", "g", "h"],
        },
        "target_index": 3,
        "text": "label-bearing transcript",
        "speaker_id": "not-needed",
    }

    projected = _request_fields(source)

    assert set(projected) == {
        "sample_id",
        "audio_path",
        "audio_sha256",
        "duration_seconds",
        "options_by_condition",
    }
    assert "target_index" not in projected
    assert "text" not in projected


def test_numeric_compare_reports_probability_and_decision_changes() -> None:
    cpu_rows = [
        {
            "sample_id": "x",
            "audio_sha256": "a" * 64,
            "probabilities": [0.7, 0.1, 0.1, 0.1],
        }
    ]
    cuda_rows = [
        {
            "sample_id": "x",
            "audio_sha256": "a" * 64,
            "probabilities": [0.700001, 0.099999, 0.1, 0.1],
        }
    ]

    report = _compare(
        {"prediction_data": {"random_distractors": cpu_rows, "hard_distractors": cpu_rows}},
        {"prediction_data": {"random_distractors": cuda_rows, "hard_distractors": cuda_rows}},
    )

    assert report["random_distractors"]["decision_count"] == 1
    assert report["random_distractors"]["argmax_changed_decisions"] == 0
    assert report["random_distractors"]["strict_probability_parity_passed"] is True


def test_request_projection_rejects_changed_choice_contract() -> None:
    source = {
        "id": "id",
        "audio_path": "audio.flac",
        "audio_sha256": "a" * 64,
        "duration_seconds": 1.0,
        "options_by_condition": {
            "random_distractors": ["a", "a", "b", "c"],
            "hard_distractors": ["d", "e", "f", "g"],
        },
    }

    try:
        _request_fields(source)
    except ValueError as error:
        assert "four unique options" in str(error)
    else:
        raise AssertionError("duplicate options should fail closed")
