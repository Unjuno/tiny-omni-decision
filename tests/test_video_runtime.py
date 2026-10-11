from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="optional Path A model-evaluation dependency")
pytest.importorskip("transformers", reason="optional Path A model-evaluation dependency")

from tiny_omni_decision.video_runtime import (  # noqa: E402
    SCORER_FEATURE_SIZE,
    binary_candidate_features,
)


def test_binary_video_features_follow_training_layout() -> None:
    video = torch.ones(768)
    question = torch.full((384,), 2.0)
    options = torch.stack([torch.zeros(384), torch.full((384,), 3.0)])

    result = binary_candidate_features(video, question, options)

    assert result.shape == (2, SCORER_FEATURE_SIZE)
    assert torch.equal(result[:, :768], video.expand(2, -1))
    assert torch.equal(result[:, 768:1152], question.expand(2, -1))
    assert torch.equal(result[:, 1152:1536], options)
    assert torch.equal(result[:, 1536:1920], (question.expand(2, -1) - options).abs())
    assert torch.equal(result[:, 1920:2304], question.expand(2, -1) * options)


@pytest.mark.parametrize(
    ("video", "question", "options"),
    [
        (torch.zeros(767), torch.zeros(384), torch.zeros(2, 384)),
        (torch.zeros(768), torch.zeros(383), torch.zeros(2, 384)),
        (torch.zeros(768), torch.zeros(384), torch.zeros(3, 384)),
        (torch.full((768,), float("inf")), torch.zeros(384), torch.zeros(2, 384)),
    ],
)
def test_binary_video_features_reject_invalid_inputs(
    video: torch.Tensor, question: torch.Tensor, options: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        binary_candidate_features(video, question, options)
