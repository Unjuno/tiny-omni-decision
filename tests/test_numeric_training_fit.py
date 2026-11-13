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


def test_warmstart_fit_requires_completion_and_consumed_examples():
    import pytest

    from scripts.evaluate_numeric_training_fit import require_completed_warmstart

    config = dict(
        initial_checkpoint="encoder_coverage",
        curriculum_training="numeric_range",
        trainable_policy="all_encoder_blocks",
        train_ids=[str(i) for i in range(16384)],
    )
    result = dict(status="PASS", steps=2048)
    require_completed_warmstart(config, result, ["0"])
    with pytest.raises(ValueError):
        require_completed_warmstart(config, dict(result, status="RUNNING"), ["0"])
    with pytest.raises(ValueError):
        require_completed_warmstart(config, result, ["unseen"])
    with pytest.raises(ValueError):
        require_completed_warmstart(dict(config, initial_checkpoint="epoch3"), result, ["0"])
