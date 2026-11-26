import numpy as np
import pytest

from tiny_omni_decision.videoprism_spatial_features import readout_features


def test_shared_readout_matched_capacity_initialization_and_permutation():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer

    cells = np.random.default_rng(17).normal(size=(2, 17, 768)).astype(np.float32)
    states = []
    for condition in ("global", "spatial"):
        torch.manual_seed(17)
        model = BilinearCandidateScorer(13056, 384, 128)
        assert sum(p.numel() for p in model.parameters()) == 1720576
        states.append(model.state_dict())
        observation = torch.from_numpy(readout_features(cells, condition))
        candidates = torch.randn(2, 3, 384)
        inputs = torch.cat((observation[:, None].expand(-1, 3, -1), candidates), -1)
        scores = model(inputs)
        torch.testing.assert_close(model(inputs[:, [2, 0, 1]]), scores[:, [2, 0, 1]])
        scores.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])


def test_global_control_discards_cell_order_but_spatial_retains_it():
    cells = np.random.default_rng(17).normal(size=(2, 17, 768)).astype(np.float32)
    permuted = cells[:, [0, *range(16, 0, -1)]]
    np.testing.assert_allclose(readout_features(cells, "global"),
                               readout_features(permuted, "global"), atol=1e-7)
    assert not np.array_equal(readout_features(cells, "spatial"),
                              readout_features(permuted, "spatial"))
