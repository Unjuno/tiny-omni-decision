from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from scripts.replay_source_unimodal_cv import compare_prediction  # noqa: E402


def test_replay_checks_identity_predictions_and_metrics():
    example = SimpleNamespace(
        record_id="r",
        video_id="v",
        none_visible=False,
        candidate_ids=("a",),
        target_candidate_ids=frozenset({"a"}),
    )
    saved = {
        "record_id": "r",
        "video_id": "v",
        "none_visible": False,
        "candidate_ids": ["a"],
        "target_candidate_ids": ["a"],
        "correct": True,
        "probabilities": [0.7, 0.3],
        "nll": 0.4,
        "brier": 0.18,
    }
    assert compare_prediction(saved, example, [0.7, 0.3], True, 0.4, 0.18) == 0
    with pytest.raises(ValueError):
        compare_prediction(saved, example, [0.6, 0.4], True, 0.4, 0.18)
    with pytest.raises(ValueError):
        compare_prediction(saved, example, [0.7, 0.3], False, 0.4, 0.18)
