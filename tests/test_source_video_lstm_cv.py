import pytest

torch = pytest.importorskip("torch")

from scripts.train_source_unimodal_cv import CandidateModel  # noqa: E402
from scripts.train_source_video_lstm_cv import TemporalCandidateModel  # noqa: E402


def test_zero_initialized_residual_matches_baseline():
    torch.manual_seed(17)
    baseline = CandidateModel().eval()
    torch.manual_seed(17)
    temporal = TemporalCandidateModel().eval()
    video, audio = torch.randn(2, 3, 1536), torch.randn(2, 384)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    sequence = torch.randn(2, 3, 4, 768)
    lengths = torch.tensor([[1, 3, 0], [4, 1, 2]])
    assert torch.equal(
        baseline(video, audio, mask, "joint"), temporal(video, audio, mask, sequence, lengths)
    )


def test_single_frame_temporal_residual_is_zero_even_when_trained():
    model = TemporalCandidateModel().eval()
    torch.nn.init.normal_(model.temporal_residual.weight)
    video, audio = torch.randn(1, 2, 1536), torch.randn(1, 384)
    mask = torch.ones(1, 2, dtype=torch.bool)
    lengths = torch.ones(1, 2, dtype=torch.long)
    sequence = torch.randn(1, 2, 4, 768)
    out1 = model(video, audio, mask, sequence, lengths)
    out2 = model(video, audio, mask, torch.randn_like(sequence), lengths)
    assert torch.equal(out1, out2)
