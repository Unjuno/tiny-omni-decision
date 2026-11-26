import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.fit_perception_temporal_match_sequence_readouts import (  # noqa: E402
    AudioCandidateScorer,
    TemporalCandidateScorer,
    categorical_metrics,
)


def test_categorical_metrics_reports_correct_choice_and_calibration_values():
    scores = np.asarray([[np.log(3.0), 0.0], [0.0, np.log(3.0)]])
    targets = np.asarray([0, 1])

    metrics = categorical_metrics(scores, targets)

    assert metrics["count"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] == pytest.approx(np.log(4.0 / 3.0))
    assert metrics["brier"] == pytest.approx(0.125)
    assert metrics["ece_10"] == pytest.approx(0.25)


def test_temporal_candidate_scorer_returns_one_score_per_option():
    model = TemporalCandidateScorer(video_dim=8, audio_dim=5, hidden_dim=4)
    video = torch.randn(3, 8, 8)
    audio = torch.randn(3, 2, 5)

    scores = model(video, audio)

    assert scores.shape == (3, 2)
    assert torch.isfinite(scores).all()


def test_temporal_candidate_scorer_can_pool_time_without_changing_option_shape():
    model = TemporalCandidateScorer(video_dim=8, audio_dim=5, hidden_dim=4, pool_video=True)
    video = torch.randn(3, 8, 8)
    audio = torch.randn(3, 2, 5)

    scores = model(video, audio)

    assert scores.shape == (3, 2)
    assert torch.isfinite(scores).all()


def test_audio_candidate_scorer_returns_one_score_per_option():
    model = AudioCandidateScorer(audio_dim=5, hidden_dim=4)
    audio = torch.randn(3, 2, 5)

    scores = model(audio)

    assert scores.shape == (3, 2)
    assert torch.isfinite(scores).all()


def test_categorical_metrics_rejects_invalid_scores():
    with pytest.raises(ValueError, match="finite"):
        categorical_metrics(np.asarray([[np.nan, 0.0]]), np.asarray([0]))
