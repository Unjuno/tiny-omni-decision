import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.robomnist_digit_decision import (  # noqa: E402
    condition_folds,
    multiclass_metrics,
    option_ordered_probabilities,
    parse_group,
)


def groups():
    return {
        f"Ar{arm}_V{speed}_Ac{digit}_Rp{rep:02d}_UNC123"
        for arm in (1, 2)
        for speed in ("low", "med", "high")
        for digit in range(10)
        for rep in range(1, 5)
    }


def test_condition_split_is_deterministic_and_holds_full_conditions():
    ids = groups()
    fold = condition_folds(ids)
    assert fold == condition_folds(set(reversed(sorted(ids))))
    assert set(fold.values()) == set(range(5))
    for group, f in fold.items():
        condition, _ = parse_group(group)
        assert all(fold[other] == f for other in ids if parse_group(other)[0] == condition)
    for f in range(5):
        assert {parse_group(g)[1] for g in ids if fold[g] == f} == set(range(10))


def test_parser_rejects_malformed_and_out_of_domain_label():
    with pytest.raises(ValueError):
        parse_group("Ar1_Vhigh_Ac10_Rp01_UNC123")
    with pytest.raises(ValueError):
        parse_group("malformed")


def test_supplied_option_order_is_preserved():
    logits = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    canonical = option_ordered_probabilities(logits, list(range(10)))
    perm = [4, 1, 9, 0, 8, 2, 7, 3, 6, 5]
    reordered = option_ordered_probabilities(logits, perm)
    assert torch.equal(reordered, canonical[:, perm])
    with pytest.raises(ValueError):
        option_ordered_probabilities(logits, [0] * 10)


def test_ten_way_metrics_and_class_counts():
    p = np.full((10, 10), 0.01)
    p[np.arange(10), np.arange(10)] = 0.91
    m = multiclass_metrics(p, np.arange(10))
    assert m["accuracy"] == 1 and m["class_count"]["9"] == 1
    with pytest.raises(ValueError):
        multiclass_metrics(np.ones((2, 2)), np.array([0, 1]))
