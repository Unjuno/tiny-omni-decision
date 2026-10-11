from collections import Counter

from scripts.evaluate_numeric_training_fit import choose_groups


def test_fixed_group_sample_is_balanced_order_independent_and_target_blind():
    rows = [
        dict(group=f"{s}:{g}", numeric_transform={"scale": s}, target=str(q))
        for s in (1, 2, 4, 8)
        for g in range(8)
        for q in range(8)
    ]
    chosen = choose_groups(rows, per_scale=2)
    assert len(chosen) == 8
    assert Counter(g.split(":")[0] for g in chosen) == {"1": 2, "2": 2, "4": 2, "8": 2}
    assert chosen == choose_groups(rows[::-1], per_scale=2)
    assert chosen == choose_groups([dict(r, target="changed") for r in rows], per_scale=2)
