from __future__ import annotations

import pytest

from scripts.compare_video_candidate_probe_predictions import paired_scene_bootstrap


def _predictions():
    rows = {}
    for scene in (10, 20, 30):
        for question in (0, 1):
            sample_id = f"scene-{scene}-q{question}"
            rows[sample_id] = {
                "id": sample_id,
                "scene_index": scene,
                "question_type": "temporal_descriptive" if question else "static_descriptive",
                "target": "yes",
                "options": ["no", "yes"],
                "probabilities": [0.25, 0.75],
            }
    return rows


def test_paired_scene_bootstrap_is_deterministic_and_zero_for_identical_models():
    predictions = _predictions()

    first = paired_scene_bootstrap(predictions, predictions, replicates=100, seed=17)
    second = paired_scene_bootstrap(predictions, predictions, replicates=100, seed=17)

    assert first == second
    assert first["independent_scenes"] == 3
    assert first["overall"]["accuracy"]["percentile_95_ci"] == [0.0, 0.0]
    assert first["overall"]["nll"]["percentile_95_ci"] == [0.0, 0.0]


def test_paired_scene_bootstrap_rejects_mismatched_questions():
    reference = _predictions()
    candidate = _predictions()
    del candidate["scene-10-q0"]

    with pytest.raises(ValueError, match="sample IDs differ"):
        paired_scene_bootstrap(reference, candidate, replicates=10)
