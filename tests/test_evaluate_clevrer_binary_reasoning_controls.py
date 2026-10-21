from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="optional Path A model-evaluation dependency")
pytest.importorskip("torch", reason="optional Path A model-evaluation dependency")
pytest.importorskip("transformers", reason="optional Path A model-evaluation dependency")
pytest.importorskip("av", reason="optional video-decoding dependency")

from scripts.evaluate_clevrer_binary_reasoning_controls import (
    _paired_scene_bootstrap,
    _scene_derangement,
)


def test_video_shuffle_is_seeded_derangement() -> None:
    scenes = [11, 12, 13, 14, 15]
    mapping = _scene_derangement(scenes, 17)

    assert mapping == _scene_derangement(scenes, 17)
    assert set(mapping) == set(scenes)
    assert set(mapping.values()) == set(scenes)
    assert all(scene != donor for scene, donor in mapping.items())


def test_scene_bootstrap_is_paired_and_deterministic() -> None:
    rows = [
        {"scene_index": scene, "question_type": kind}
        for scene in (1, 2, 3)
        for kind in ("explanatory", "predictive", "counterfactual")
    ]
    baseline = []
    improved = []
    for row in rows:
        baseline.append(
            {
                "scene_index": row["scene_index"],
                "question_type": row["question_type"],
                "target_index": 1,
                "probabilities": [0.6, 0.4],
            }
        )
        improved.append(
            {
                "scene_index": row["scene_index"],
                "question_type": row["question_type"],
                "target_index": 1,
                "probabilities": [0.2, 0.8],
            }
        )

    args = ({"baseline": baseline, "improved": improved}, rows)
    first = _paired_scene_bootstrap(*args, replicates=200, seed=17)
    second = _paired_scene_bootstrap(*args, replicates=200, seed=17)

    assert first == second
    interval = first["delta_intervals"]["improved_minus_baseline"]
    assert interval["accuracy"] == [1.0, 1.0]
    assert interval["nll"][1] < 0
    assert interval["brier"][1] < 0
