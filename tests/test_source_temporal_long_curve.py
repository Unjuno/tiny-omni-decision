import pytest

torch = pytest.importorskip("torch")

from scripts.train_source_temporal_long_curve import CHECKPOINTS, EVALUATIONS, score  # noqa: E402
from scripts.train_source_unimodal_cv import CandidateModel  # noqa: E402


def test_fixed_evaluation_and_checkpoint_schedule():
    assert EVALUATIONS == (1, 2, 4, 7, 8, 16, 24, 32)
    assert CHECKPOINTS == (7, 16, 32)


def test_score_baseline_dispatch_and_unknown_mode():
    model = CandidateModel()
    video, audio, mask = (
        torch.randn(1, 2, 1536),
        torch.randn(1, 384),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.equal(
        score(model, "baseline", video, audio, mask, [], {}), model(video, audio, mask, "joint")
    )
    with pytest.raises(ValueError):
        score(model, "unknown", video, audio, mask, [], {})
