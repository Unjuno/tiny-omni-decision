import pytest

from scripts.bootstrap_perception_source_candidate_controls import paired_group_bootstrap


def prediction(record_id, video_id, *, correct, confidence, nll, candidate_probability):
    return {
        "record_id": record_id,
        "video_id": video_id,
        "candidate_ids": ["candidate"],
        "target_candidate_ids": ["candidate"],
        "none_visible": False,
        "predicted": "candidate" if correct else "none_visible",
        "top1_correct": correct,
        "top1_confidence": confidence,
        "nll": nll,
        "candidate_probabilities": {"candidate": candidate_probability},
        "none_visible_probability": 1 - candidate_probability,
    }


def test_paired_bootstrap_resamples_videos_and_is_deterministic():
    joint = [
        prediction(
            "a1", "video-a", correct=True, confidence=0.8, nll=0.2, candidate_probability=0.8
        ),
        prediction(
            "a2", "video-a", correct=True, confidence=0.8, nll=0.2, candidate_probability=0.8
        ),
        prediction(
            "b1", "video-b", correct=False, confidence=0.7, nll=1.2, candidate_probability=0.3
        ),
    ]
    control = [
        prediction(
            "a1", "video-a", correct=False, confidence=0.7, nll=1.2, candidate_probability=0.3
        ),
        prediction(
            "a2", "video-a", correct=False, confidence=0.7, nll=1.2, candidate_probability=0.3
        ),
        prediction(
            "b1", "video-b", correct=False, confidence=0.7, nll=1.2, candidate_probability=0.3
        ),
    ]

    first = paired_group_bootstrap(joint, control, resamples=200, seed=17)
    second = paired_group_bootstrap(joint, control, resamples=200, seed=17)

    assert first == second
    assert first["records"] == 3
    assert first["independent_video_groups"] == 2
    assert first["metrics"]["accuracy"]["reference_minus_comparison"] == pytest.approx(2 / 3)
    assert first["metrics"]["accuracy"]["bootstrap_probability_reference_better"] == pytest.approx(
        0.74
    )
    assert first["metrics"]["nll"]["reference_minus_comparison"] < 0


def test_paired_bootstrap_rejects_misaligned_video_rows():
    first = prediction(
        "a1", "video-a", correct=True, confidence=0.8, nll=0.2, candidate_probability=0.8
    )
    second = prediction(
        "a1", "video-b", correct=True, confidence=0.8, nll=0.2, candidate_probability=0.8
    )

    with pytest.raises(ValueError, match="video_id"):
        paired_group_bootstrap([first], [second], resamples=10)
