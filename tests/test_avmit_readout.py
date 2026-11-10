from __future__ import annotations

import pytest
import torch

from tiny_omni_decision.avmit_readout import AVMITFeatureDecisionScorer


@pytest.mark.parametrize("policy", ["mean", "gru"])
def test_avmit_readout_is_finite_trainable_and_option_order_equivariant(policy):
    torch.manual_seed(17)
    model = AVMITFeatureDecisionScorer(
        audio_width=3, video_width=4, temporal_policy=policy, hidden_width=5, scorer_width=7
    )
    audio = torch.randn(3, 5, 3)
    video = torch.randn(3, 7, 4)
    audio_lengths = torch.tensor([5, 3, 1], dtype=torch.long)
    video_lengths = torch.tensor([7, 2, 1], dtype=torch.long)
    original_audio, original_video = audio.clone(), video.clone()
    options, valid = model.option_features(3, device=audio.device)

    logits = model(audio, audio_lengths, video, video_lengths, options, valid)
    reversed_logits = model(
        audio, audio_lengths, video, video_lengths, options.flip(1), valid.flip(1)
    )
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(reversed_logits, logits.flip(1))
    torch.testing.assert_close(audio, original_audio)
    torch.testing.assert_close(video, original_video)

    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]))
    loss.backward()
    assert model.parameter_count() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("policy", ["mean", "gru"])
def test_avmit_readout_ignores_right_padding(policy):
    torch.manual_seed(3)
    model = AVMITFeatureDecisionScorer(
        audio_width=2, video_width=2, temporal_policy=policy, hidden_width=4, scorer_width=5
    )
    audio = torch.randn(1, 4, 2)
    video = torch.randn(1, 3, 2)
    audio_lengths = torch.tensor([2], dtype=torch.long)
    video_lengths = torch.tensor([1], dtype=torch.long)
    audio_changed = audio.clone()
    video_changed = video.clone()
    audio_changed[:, 2:] = 1000
    video_changed[:, 1:] = -1000
    options, valid = model.option_features(1, device=audio.device)

    logits = model(audio, audio_lengths, video, video_lengths, options, valid)
    padded_logits = model(
        audio_changed, audio_lengths, video_changed, video_lengths, options, valid
    )
    torch.testing.assert_close(padded_logits, logits)


def test_avmit_readout_validates_shapes_and_round_trips(tmp_path):
    model = AVMITFeatureDecisionScorer(
        audio_width=2, video_width=3, temporal_policy="mean", hidden_width=4, scorer_width=5
    )
    audio = torch.ones(1, 2, 2)
    video = torch.ones(1, 2, 3)
    options, valid = model.option_features(1, device=audio.device)
    args = (audio, torch.tensor([2]), video, torch.tensor([1]), options, valid)
    expected = model(*args)
    checkpoint = tmp_path / "readout.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = AVMITFeatureDecisionScorer(
        audio_width=2, video_width=3, temporal_policy="mean", hidden_width=4, scorer_width=5
    )
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    torch.testing.assert_close(restored(*args), expected)

    with pytest.raises(ValueError, match="valid sequence lengths"):
        model(audio, torch.tensor([3]), video, torch.tensor([1]), options, valid)
