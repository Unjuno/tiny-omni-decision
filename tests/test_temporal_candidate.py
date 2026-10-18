import pytest

torch = pytest.importorskip("torch")


def test_temporal_candidate_scorer_supports_variable_option_counts():
    from tiny_omni_decision.temporal_candidate import (
        TemporalCandidateScorer,
        temporal_candidate_parameter_count,
    )

    model = TemporalCandidateScorer(
        text_feature_size=12,
        video_feature_size=6,
        temporal_hidden_size=4,
        scorer_hidden_size=5,
    )
    video = torch.randn(2, 8, 6)
    options = torch.randn(2, 4, 12)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

    logits = model(video, options, mask)

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert logits[0, 2] < -1e30
    assert logits[0, 0] != logits[0, 1]
    assert temporal_candidate_parameter_count(model) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_temporal_candidate_scorer_rejects_all_masked_example():
    from tiny_omni_decision.temporal_candidate import TemporalCandidateScorer

    model = TemporalCandidateScorer(
        text_feature_size=12,
        video_feature_size=6,
        temporal_hidden_size=4,
        scorer_hidden_size=5,
    )

    with pytest.raises(ValueError, match="at least one valid option"):
        model(torch.randn(1, 8, 6), torch.randn(1, 2, 12), torch.zeros(1, 2, dtype=torch.bool))


def test_temporal_candidate_scorer_rejects_non_finite_features():
    from tiny_omni_decision.temporal_candidate import TemporalCandidateScorer

    model = TemporalCandidateScorer(
        text_feature_size=12,
        video_feature_size=6,
        temporal_hidden_size=4,
        scorer_hidden_size=5,
    )
    video = torch.randn(1, 8, 6)
    video[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        model(video, torch.randn(1, 2, 12))
