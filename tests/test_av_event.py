import pytest
import torch

from tiny_omni_decision.av_event import AVEventSourceScorer


def test_audio_only_is_uniform_over_opaque_options():
    scorer = AVEventSourceScorer(audio_size=8, visual_size=6, hidden_size=4)
    first = scorer(
        condition="audio_only", option_count=5, audio_features=torch.randn(8)
    )
    second = scorer(
        condition="audio_only", option_count=5, audio_features=torch.randn(8)
    )
    assert torch.equal(first, torch.zeros(5))
    assert torch.equal(first, second)
    assert torch.equal(first.softmax(-1), torch.full((5,), 0.2))


@pytest.mark.parametrize("condition", ["video_only", "joint"])
def test_option_permutation_permutes_logits_without_changing_scores(condition):
    torch.manual_seed(17)
    scorer = AVEventSourceScorer(audio_size=8, visual_size=6, hidden_size=4).eval()
    visual = torch.randn(4, 6)
    audio = torch.randn(8)
    kwargs = {"audio_features": audio} if condition == "joint" else {}
    first = scorer(condition=condition, visual_features=visual, **kwargs)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = scorer(
        condition=condition,
        visual_features=visual[permutation],
        **kwargs,
    )
    assert torch.equal(permuted, first[permutation])
    assert torch.allclose(first.softmax(-1).sum(), torch.tensor(1.0))


def test_joint_condition_uses_audio_and_returns_bounded_option_gates():
    torch.manual_seed(17)
    scorer = AVEventSourceScorer(audio_size=8, visual_size=6, hidden_size=4).eval()
    visual = torch.randn(3, 6)
    audio = torch.randn(8)
    logits, gates = scorer(
        condition="joint",
        visual_features=visual,
        audio_features=audio,
        return_gate=True,
    )
    changed_audio_logits = scorer(
        condition="joint",
        visual_features=visual,
        audio_features=-audio,
    )
    assert gates is not None and gates.shape == (3, 4)
    assert torch.all((gates > 0) & (gates < 1))
    assert logits.shape == (3,) and torch.isfinite(logits).all()
    assert not torch.equal(logits, changed_audio_logits)


def test_conditions_fail_closed_on_missing_or_cross_condition_features():
    scorer = AVEventSourceScorer(audio_size=8, visual_size=6, hidden_size=4)
    audio = torch.randn(8)
    visual = torch.randn(3, 6)
    with pytest.raises(ValueError, match="at least two"):
        scorer(condition="audio_only", option_count=1, audio_features=audio)
    with pytest.raises(ValueError, match="forbids visual"):
        scorer(
            condition="audio_only",
            option_count=3,
            audio_features=audio,
            visual_features=visual,
        )
    with pytest.raises(ValueError, match="forbids audio"):
        scorer(condition="video_only", visual_features=visual, audio_features=audio)
    with pytest.raises(ValueError, match="requires both"):
        scorer(condition="joint", visual_features=visual)


def test_scorer_rejects_nonfinite_or_wrongly_shaped_options():
    scorer = AVEventSourceScorer(audio_size=8, visual_size=6, hidden_size=4)
    with pytest.raises(ValueError, match="finite"):
        scorer(condition="video_only", visual_features=torch.full((2, 6), float("nan")))
    with pytest.raises(ValueError, match="shape"):
        scorer(condition="video_only", visual_features=torch.randn(2, 5))
    with pytest.raises(ValueError, match="audio-only.*finite"):
        scorer(
            condition="audio_only",
            option_count=2,
            audio_features=torch.full((8,), float("inf")),
        )
