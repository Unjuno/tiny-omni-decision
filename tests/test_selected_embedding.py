import pytest

torch = pytest.importorskip("torch")


def test_selected_embedding_exact_init_frozen_base_and_isolation():
    from tiny_omni_decision.selected_embedding import SelectedEmbeddingDelta

    base = torch.nn.Embedding(10, 4)
    before = base.weight.detach().clone()
    m = SelectedEmbeddingDelta(base, [2, 7])
    ids = torch.tensor([[0, 2, 7, 9]])
    torch.testing.assert_close(m(ids), base(ids), rtol=0, atol=0)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 8
    opt = torch.optim.AdamW([m.delta], lr=0.1)
    m(ids).sum().backward()
    assert base.weight.grad is None
    assert torch.isfinite(m.delta.grad).all()
    opt.step()
    torch.testing.assert_close(base.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(m(ids)[:, [0, 3]], base(ids)[:, [0, 3]], rtol=0, atol=0)
    assert not torch.equal(m(ids)[:, [1, 2]], base(ids)[:, [1, 2]])
    with pytest.raises(ValueError):
        SelectedEmbeddingDelta(torch.nn.Embedding(10, 4), [2, 2])
