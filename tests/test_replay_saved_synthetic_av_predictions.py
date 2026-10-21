import math

import pytest

from scripts.replay_saved_synthetic_av_predictions import (
    _assert_paired_rows,
    _metrics,
    _probability_matrix,
    _scene_bootstrap,
    _swap_metrics,
)


def _row(sample_id, pair_id, scene_id, target_index, predicted_index=None):
    predicted_index = target_index if predicted_index is None else predicted_index
    probabilities = [0.9, 0.1] if predicted_index == 0 else [0.1, 0.9]
    return {
        "id": sample_id,
        "pair_id": pair_id,
        "scene_group_id": scene_id,
        "target": "before" if target_index == 0 else "after",
        "target_index": target_index,
        "prediction": "before" if predicted_index == 0 else "after",
        "probabilities": [
            {"option": "before", "probability": probabilities[0]},
            {"option": "after", "probability": probabilities[1]},
        ],
    }


def test_recompute_metrics_from_saved_probabilities():
    rows = [_row("a", "p1", "s1", 0), _row("b", "p2", "s2", 1)]
    probabilities, targets = _probability_matrix(rows)
    metrics = _metrics(probabilities, targets)
    assert metrics["count"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] == pytest.approx(-math.log(0.9))
    assert metrics["brier"] == pytest.approx(0.02)


def test_paired_rows_reject_reordered_or_mismatched_ids():
    rows = [_row("a", "p1", "s1", 0), _row("b", "p1", "s1", 1)]
    with pytest.raises(ValueError, match="identity/target mismatch"):
        _assert_paired_rows(rows, list(reversed(rows)), name="test")


def test_scene_bootstrap_resamples_scenes_and_is_deterministic():
    rows = [
        _row("a1", "p1", "s1", 0), _row("a2", "p1", "s1", 1),
        _row("b1", "p2", "s2", 0), _row("b2", "p2", "s2", 1),
    ]
    joint, targets = _probability_matrix(rows)
    audio_rows = [
        _row("a1", "p1", "s1", 0, 1), _row("a2", "p1", "s1", 1, 0),
        _row("b1", "p2", "s2", 0, 1), _row("b2", "p2", "s2", 1, 0),
    ]
    audio, audio_targets = _probability_matrix(audio_rows)
    result = _scene_bootstrap(
        rows, joint, targets, audio, audio_targets, repetitions=100, seed=17
    )
    assert result["scene_count"] == 2
    assert result["event_pair_count"] == 2
    assert result["metrics"]["accuracy"]["difference"] == 1.0
    assert result == _scene_bootstrap(
        rows, joint, targets, audio, audio_targets, repetitions=100, seed=17
    )


def test_audio_swap_tracks_original_donor_targets_and_prediction_sensitivity():
    original_rows = [
        _row("a", "p1", "s1", 0), _row("b", "p1", "s1", 1)
    ]
    original, _ = _probability_matrix(original_rows)
    swapped_rows = [
        _row("a", "p1", "s1", 0, 1), _row("b", "p1", "s1", 1, 0)
    ]
    swapped, _ = _probability_matrix(swapped_rows)
    result = _swap_metrics(original_rows, original, swapped_rows, swapped)
    assert result["swapped_accuracy_on_original_target"] == 0.0
    assert result["swapped_accuracy_on_opposite_donor_target"] == 1.0
    assert result["prediction_flip_rate"] == 1.0


def test_probability_validation_fails_closed():
    rows = [_row("a", "p1", "s1", 0), _row("a", "p2", "s2", 1)]
    with pytest.raises(ValueError, match="unique"):
        _probability_matrix(rows)
