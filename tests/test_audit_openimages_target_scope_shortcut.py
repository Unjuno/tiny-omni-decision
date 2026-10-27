import pytest

pytest.importorskip("torch")
from scripts.audit_openimages_target_scope_shortcut import audit  # noqa: E402


def test_whitelist_shortcut_and_tie_expectation():
    whitelist = {f"c{i}" for i in range(11)}
    rows = [
        {"sample_id": "1", "options": ["z", "c0", "y", "x"], "target": "c0", "prediction": "c0"},
        {"sample_id": "2", "options": ["c0", "c1", "y", "x"], "target": "c1", "prediction": "c1"},
    ]
    result = audit(rows, whitelist)
    assert result["target_whitelist_option_count"] == {1: 1, 2: 1}
    assert result["random_uniform_within_whitelist_expected_accuracy"] == 0.75
    assert result["first_whitelist_option_accuracy"] == 0.5
    assert result["saved_model_accuracy"] == 1.0
    assert audit(rows, whitelist) == result


def test_refuse_target_outside_train_scope():
    row = {"sample_id": "1", "options": ["a", "b", "c", "d"], "target": "d", "prediction": "a"}
    with pytest.raises(ValueError, match="target-scope"):
        audit([row], {"a", *[f"c{i}" for i in range(10)]})
