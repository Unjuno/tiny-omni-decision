import pytest

from scripts.evaluate_vocabulary_transfer import require_selection


def test_completed_earliest_selection_only():
    history = [dict(step=s, selector=1 if s in (128, 256) else 2) for s in range(0, 2049, 128)]
    result = dict(status="PASS", steps=2048, best_step=128, smoke=False)
    config = dict(
        curriculum_training="comparison_vocabulary", initial_checkpoint="encoder_coverage", seed=17
    )
    assert require_selection(result, history, config) == 128
    for key, value in [("status", "RUNNING"), ("steps", 128), ("best_step", 256), ("smoke", True)]:
        with pytest.raises(ValueError):
            require_selection(dict(result, **{key: value}), history, config)
    with pytest.raises(ValueError):
        require_selection(result, history[:-1], config)
    with pytest.raises(ValueError):
        require_selection(result, history, dict(config, seed=18))
    with pytest.raises(ValueError):
        require_selection(result, [dict(r, selector=float("nan")) for r in history], config)


def test_opposing_pairs_use_candidate_identity_not_option_position():
    from scripts.build_conditional_curriculum import build
    from scripts.evaluate_vocabulary_transfer import opposing_pairs

    rows = build(167, 1, "test")
    predictions = [
        dict(
            id=r["id"],
            options=r["options"],
            target=r["target"],
            probabilities=[float(o == r["target"]) for o in r["options"]],
        )
        for r in rows
    ]
    result = opposing_pairs(rows, predictions)
    assert set(result) == {"cost", "risk", "time"}
    assert all(r == dict(pairs=1, both_correct=1, prediction_changed=1) for r in result.values())
    with pytest.raises(ValueError):
        opposing_pairs(rows, predictions[::-1])
    with pytest.raises(ValueError):
        opposing_pairs(rows[1:], predictions[1:])
