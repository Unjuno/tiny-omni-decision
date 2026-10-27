import pytest

torch = pytest.importorskip("torch")
from scripts.source_set_decision import decode_set, set_loss  # noqa: E402


def test_multitarget_gradients_and_padding_invariance():
    logits = torch.tensor([[1.0, -0.5, 8.0, -1.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    targets = torch.tensor([[True, True, False, False]])
    loss = set_loss(logits, mask, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0
    assert logits.grad[0, 2] == 0 and logits.grad[0, 3] > 0
    changed = logits.detach().clone()
    changed[0, 2] = -999
    assert torch.allclose(loss.detach(), set_loss(changed, mask, targets))


def test_none_has_no_candidate_loss_and_decode_is_exclusive():
    logits = torch.tensor([[10.0, 10.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[True, True]])
    targets = torch.tensor([[False, False, True]])
    set_loss(logits, mask, targets).backward()
    assert torch.equal(logits.grad[0, :2], torch.zeros(2))
    assert decode_set(logits.detach(), mask).tolist() == [[False, False, True]]


def test_visible_selects_multiple_or_falls_back_to_best():
    mask = torch.tensor([[True, True, False], [True, True, False]])
    logits = torch.tensor([[1.0, 2.0, 99.0, -2.0], [-2.0, -1.0, 99.0, -2.0]])
    assert decode_set(logits, mask).tolist() == [
        [True, True, False, False],
        [False, True, False, False],
    ]


def test_reject_inconsistent_targets_and_empty_candidate_rows():
    mask = torch.tensor([[True, False]])
    logits = torch.zeros(1, 3)
    for targets in (
        torch.tensor([[True, False, True]]),
        torch.tensor([[False, False, False]]),
        torch.tensor([[False, True, False]]),
    ):
        with pytest.raises(ValueError):
            set_loss(logits, mask, targets)
    with pytest.raises(ValueError):
        decode_set(logits, torch.tensor([[False, False]]))


def test_audio_only_candidate_logits_are_symmetric():
    from scripts.train_source_unimodal_cv import CandidateModel

    torch.manual_seed(17)
    model = CandidateModel()
    video = torch.randn(1, 3, 1536)
    audio = torch.randn(1, 384)
    mask = torch.tensor([[True, True, True]])
    output = model(video, audio, mask, "audio")
    assert torch.allclose(output[0, :3], output[0, :1].expand(3))


def test_tied_fallback_selects_all_maxima_independent_of_order():
    mask = torch.tensor([[True, True, True]])
    logits = torch.tensor([[-1.0, -2.0, -1.0, -2.0]])
    assert decode_set(logits, mask).tolist() == [[True, False, True, False]]
    tied = torch.tensor([[-1.0, -1.0, -1.0, -2.0]])
    assert decode_set(tied, mask).tolist() == [[True, True, True, False]]
