from __future__ import annotations

import torch

from scripts.pretrained_reuse.train_haptile_decision import _fit_temperature


def test_temperature_fit_is_deterministic_and_uses_macro_groups() -> None:
    logits = torch.tensor([
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
    ])
    targets = torch.tensor([0, 1, 2, 3])
    groups = ["task-a", "task-a", "task-b", "task-b"]
    first = _fit_temperature(logits, targets, groups)
    second = _fit_temperature(logits, targets, groups)
    assert first == second
    assert first["objective"] == "minimum calibration macro task-group NLL"
    assert first["calibration_task_groups"] == ["task-a", "task-b"]
    assert first["calibration_examples"] == 4
    assert first["temperature"] == 0.25


def test_temperature_fit_rejects_single_group_and_nonfinite_logits() -> None:
    logits = torch.zeros((2, 7))
    targets = torch.tensor([0, 1])
    try:
        _fit_temperature(logits, targets, ["only", "only"])
    except ValueError as error:
        assert "multiple calibration task groups" in str(error)
    else:
        raise AssertionError("single-group calibration was accepted")

    logits[0, 0] = float("nan")
    try:
        _fit_temperature(logits, targets, ["task-a", "task-b"])
    except ValueError as error:
        assert "invalid" in str(error)
    else:
        raise AssertionError("nonfinite calibration logits were accepted")
