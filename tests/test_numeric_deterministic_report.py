from copy import deepcopy

import pytest

from scripts.report_numeric_deterministic import verify_pair


def test_pair_rejects_unplanned_data_or_training_changes():
    control = dict(
        condition="control",
        steps=2048,
        smoke=False,
        deterministic=True,
        train_ids=["a", "b"],
        validation_ids=[["v"]],
        lr=2e-5,
        trace_sha256="controltrace",
        disk_output_allowance=280,
    )
    numeric = dict(
        deepcopy(control),
        condition="numeric",
        trace_sha256="numerictrace",
        disk_output_allowance=140,
    )
    verify_pair(control, numeric)
    for key, value in [
        ("train_ids", ["b", "a"]),
        ("lr", 1e-5),
        ("deterministic", False),
        ("steps", 512),
        ("smoke", True),
        ("validation_ids", [["changed"]]),
        ("unexpected", 1),
    ]:
        with pytest.raises(ValueError):
            verify_pair(control, dict(numeric, **{key: value}))
    with pytest.raises(ValueError):
        verify_pair(numeric, control)
