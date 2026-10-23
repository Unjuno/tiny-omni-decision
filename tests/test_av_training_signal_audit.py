import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.audit_av_training_signal import (  # noqa: E402
    make_model,
    select_train_pairs,
    validate_pair_tensors,
)


def metadata():
    return {
        "sample_ids": np.array(["v0", "v1", "b0", "b1", "a1", "a0"]),
        "splits": np.array(["validation"] * 2 + ["train"] * 4),
        "scene_ids": np.array(["v"] * 2 + ["b"] * 2 + ["a"] * 2),
        "pair_ids": np.array(["vp"] * 2 + ["bp"] * 2 + ["ap"] * 2),
        "targets": np.array([0, 1, 0, 1, 1, 0]),
    }


def test_select_train_only_balanced_pairs_deterministically_without_order_dependence():
    bank = metadata()
    ix = select_train_pairs(bank, pairs=2, seed=17)
    assert len(ix) == 4
    assert set(bank["splits"][ix]) == {"train"}
    for offset in [0, 2]:
        assert bank["targets"][ix[offset:offset + 2]].tolist() == [0, 1]
    shuffled = {k: v[::-1] for k, v in bank.items()}
    other = select_train_pairs(shuffled, pairs=2, seed=17)
    assert bank["sample_ids"][ix].tolist() == shuffled["sample_ids"][other].tolist()
    with pytest.raises(ValueError, match="distinct train scenes"):
        select_train_pairs(bank, pairs=3, seed=17)


def test_fail_closed_for_split_scene_or_target_mismatch():
    for key, index, value in [("splits", 2, "validation"),
                              ("scene_ids", 2, "bad"), ("targets", 2, 1)]:
        bank = metadata()
        bank[key] = bank[key].astype(object)
        bank[key][index] = value
        with pytest.raises(ValueError):
            select_train_pairs(bank, pairs=2, seed=17)


def test_pair_features_must_match_except_audio_and_target():
    values = {"audio": torch.tensor([[[0.], [1.]], [[1.], [0.]]]),
              "video": torch.zeros(2, 2, 768), "text": torch.zeros(2, 384),
              "targets": torch.tensor([0, 1])}
    validate_pair_tensors(values)
    broken = copy.deepcopy(values)
    broken["audio"][1] = broken["audio"][0]
    with pytest.raises(ValueError, match="audio"):
        validate_pair_tensors(broken)
    broken = copy.deepcopy(values)
    broken["video"][1, 0, 0] = 1
    with pytest.raises(ValueError, match="video"):
        validate_pair_tensors(broken)


def test_query_freeze_is_only_initial_arm_difference():
    original, frozen = make_model(False), make_model(True)
    for (name, first), (other, second) in zip(
        original.named_parameters(), frozen.named_parameters(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert first.requires_grad
        assert second.requires_grad == (not name.startswith("baseline.query."))
