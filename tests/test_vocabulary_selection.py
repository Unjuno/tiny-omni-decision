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
