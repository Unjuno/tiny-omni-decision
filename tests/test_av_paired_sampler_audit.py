import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.audit_av_paired_sampler import pair_indexes, sampler_orders  # noqa: E402


def test_pair_index_requires_opposite_targets_same_scene_and_unique_rows():
    ids = np.array(["a", "b", "a", "b"])
    scenes = np.array(["x", "y", "x", "y"])
    targets = np.array([1, 0, 0, 1])
    assert pair_indexes(ids, scenes, targets).tolist() == [[2, 0], [1, 3]]
    with pytest.raises(ValueError):
        pair_indexes(ids, scenes, np.zeros(4, dtype=int))
    with pytest.raises(ValueError):
        pair_indexes(ids, np.array(["x", "x", "y", "y"]), targets)


def test_samplers_visit_identical_unique_examples_each_epoch_without_loss():
    pairs = torch.tensor([[2, 0], [1, 3], [4, 5]])
    for paired in [True, False]:
        first = sampler_orders(pairs, paired=paired, epochs=3, seed=17)
        second = sampler_orders(pairs, paired=paired, epochs=3, seed=17)
        for a, b in zip(first, second, strict=True):
            assert torch.equal(a, b)
            assert sorted(a.tolist()) == list(range(6))
            if paired:
                assert {frozenset(row) for row in a.reshape(-1, 2).tolist()} == {
                    frozenset(row) for row in pairs.tolist()}
    with pytest.raises(ValueError):
        sampler_orders(torch.tensor([[0, 1], [1, 2]]), paired=True, epochs=1, seed=17)
