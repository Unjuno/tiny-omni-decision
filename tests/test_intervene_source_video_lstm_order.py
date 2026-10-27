import pytest

torch = pytest.importorskip("torch")

from scripts.intervene_source_video_lstm_order import reverse_valid  # noqa: E402


def test_reverses_only_valid_frames_and_keeps_padding():
    seq = torch.tensor([[[[1.0], [2.0], [99.0]], [[5.0], [88.0], [77.0]]]])
    result = reverse_valid(seq, torch.tensor([[2, 1]]))
    assert result.tolist() == [[[[2.0], [1.0], [99.0]], [[5.0], [88.0], [77.0]]]]
    assert seq[0, 0, 0, 0] == 1


def test_invalid_lengths_fail():
    with pytest.raises(ValueError):
        reverse_valid(torch.zeros(1, 1, 2, 1), torch.tensor([[3]]))
