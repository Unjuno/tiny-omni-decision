import copy

import pytest

from scripts.evaluate_numeric_range_transfer import selected_step


def test_transfer_requires_completed_frozen_selection():
    history = [dict(step=s, selector=1.0) for s in range(0, 2049, 128)]
    result = dict(status="PASS", steps=2048, best_step=0)
    assert selected_step(history, result) == 0
    with pytest.raises(ValueError):
        selected_step(history, dict(result, best_step=2048))
    with pytest.raises(ValueError):
        selected_step(history[:-1], result)
    with pytest.raises(ValueError):
        selected_step(history, dict(result, status="RUNNING"))
    broken = copy.deepcopy(history)
    broken[2]["selector"] = float("nan")
    with pytest.raises(ValueError):
        selected_step(broken, result)
