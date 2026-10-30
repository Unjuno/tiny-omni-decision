import pytest

torch = pytest.importorskip("torch")

def test_gru_readout_scores_four_candidates_and_preserves_option_equivariance():
    from scripts.train_egoviz_gru_temporal import OrderedCrossModalGRUScorer

    torch.manual_seed(17)
    model = OrderedCrossModalGRUScorer()
    video = torch.randn(3, 8, 768)
    sensor = torch.randn(3, 4, 6, 16, 512)

    logits = model(video, sensor)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(video, sensor[:, permutation])

    assert logits.shape == (3, 4)
    torch.testing.assert_close(permuted, logits[:, permutation], rtol=1e-5, atol=1e-6)


def test_gru_readout_is_small_and_all_parameters_receive_gradients():
    from scripts.train_egoviz_gru_temporal import OrderedCrossModalGRUScorer

    model = OrderedCrossModalGRUScorer()
    video = torch.randn(2, 8, 768)
    sensor = torch.randn(2, 4, 6, 16, 512)

    model(video, sensor).sum().backward()

    assert sum(parameter.numel() for parameter in model.parameters()) == 123_521
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
