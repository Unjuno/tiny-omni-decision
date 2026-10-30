import pytest

torch = pytest.importorskip("torch")
from scripts.train_role_supervised_qa import auxiliary_loss  # noqa: E402
from tiny_omni_decision.cover_point_readout import balanced_membership_loss  # noqa: E402


def test_missing_auxiliary_label_disables_only_its_auxiliary_gradient():
    logits = torch.tensor([[1.0, 0.0], [2.0, 3.0]], requires_grad=True)
    labels = torch.tensor([[True, False], [False, False]])
    available = torch.tensor([True, False])
    loss = auxiliary_loss(logits, labels, available)
    torch.testing.assert_close(loss, balanced_membership_loss(logits[:1], labels[:1]))
    loss.backward()
    assert logits.grad[0].abs().sum() > 0 and logits.grad[1].abs().sum() == 0
    empty = auxiliary_loss(logits, labels, torch.zeros(2, dtype=torch.bool))
    assert empty == 0


def test_checkpoint_nll_uses_stable_log_probabilities_without_flooring():
    from scripts.train_role_supervised_qa import qa_summary

    targets = torch.tensor([0, 0])
    a = torch.tensor([[-40.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    b = torch.tensor([[-30.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    results = []
    for logits in (a, b):
        m = qa_summary(logits.softmax(-1).numpy(), targets.numpy(), logits.log_softmax(-1).numpy())
        assert m["nll"] == pytest.approx(torch.nn.functional.cross_entropy(logits, targets).item())
        assert m["initial"]["nll"] == pytest.approx(-logits.log_softmax(-1)[0, 0].item())
        assert m["final"]["nll"] == pytest.approx(-logits.log_softmax(-1)[1, 0].item())
        assert m["unchanged"]["nll"] == pytest.approx(m["nll"])
        results.append(m)
    assert results[1]["nll"] < results[0]["nll"]


def test_reload_compares_original_saved_epoch_outputs(tmp_path):
    import numpy as np

    from scripts.train_role_supervised_qa import verify_saved_epoch

    values = (np.ones((2, 3)), np.ones((1, 2)), np.ones((2, 3)))
    path = tmp_path / "epoch.npz"
    np.savez(path, train=values[0], train_cover=values[1], train_logp=values[2])
    verify_saved_epoch(path, {"train": values})
    wrong = (values[0] * 2, values[1], values[2])
    with pytest.raises(AssertionError):
        verify_saved_epoch(path, {"train": wrong})
