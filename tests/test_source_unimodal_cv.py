import pytest

torch = pytest.importorskip("torch")

from scripts.train_source_unimodal_cv import CandidateModel, fold_assignment  # noqa: E402


def test_folds_keep_groups_together_and_are_order_independent():
    ids = ["a", "b", "c", "a", "d", "e"]
    assert fold_assignment(ids) == fold_assignment(list(reversed(ids)))
    assert set(fold_assignment(ids).values()) == {0, 1, 2}


def test_audio_only_visible_candidates_tied_but_none_has_own_scorer():
    torch.manual_seed(17)
    model = CandidateModel()
    video = torch.randn(2, 3, 1536)
    audio = torch.randn(2, 384)
    scores = model(video, audio, torch.ones(2, 3, dtype=torch.bool), "audio")
    assert torch.allclose(scores[:, 0], scores[:, 1])
    assert torch.allclose(scores[:, 1], scores[:, 2])
    assert not torch.allclose(scores[:, 0], scores[:, -1])


def test_video_only_ignores_audio():
    model = CandidateModel()
    video = torch.randn(1, 2, 1536)
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert torch.equal(
        model(video, torch.zeros(1, 384), mask, "video"),
        model(video, torch.randn(1, 384), mask, "video"),
    )
