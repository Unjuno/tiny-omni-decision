import copy

import pytest

pytest.importorskip("torch")
pytest.importorskip("av")
pytest.importorskip("transformers")

from scripts.probe_native_mask_memory_4h import validate_amendment  # noqa: E402


def reference():
    contract = dict(
        source_commit="old", source_hashes={"model": "fixed"}, frames=32, max_main_seconds=7200
    )
    rows = [
        dict(arm=a, cache_size=c, seconds_by_seed=[10, 6])
        for a, c in [("actual", 1), ("actual", 32), ("initial_repeat", 32)]
    ]
    return dict(
        contract=contract,
        status="PASS_NATIVE_MEMORY_SMOKE_ONLY",
        rows=rows,
        projected_main_seconds=2 * (6 * 696 * 1.25 + 10 * 8),
    )


def test_explicit_resource_amendment_keeps_old_gate_failure():
    prior = reference()
    current = copy.deepcopy(prior["contract"])
    current["source_commit"] = "new"
    result = validate_amendment(prior, current)
    assert result["original_scale_gate_passed"] is False
    assert result["amended_scale_gate_passed"] is True
    assert result["original_main_limit_seconds"] == 7200
    assert result["main_limit_seconds"] == 14400
    assert prior["contract"]["source_commit"] == "old"


@pytest.mark.parametrize("key,value", [("frames", 16), ("source_hashes", {"model": "changed"})])
def test_no_condition_or_inherited_pin_change_is_admitted(key, value):
    prior = reference()
    current = copy.deepcopy(prior["contract"])
    current[key] = value
    with pytest.raises(ValueError):
        validate_amendment(prior, current)


@pytest.mark.parametrize("value", [float("nan"), 15000, 1])
def test_projection_is_recomputed_and_bounded(value):
    prior = reference()
    prior["projected_main_seconds"] = value
    with pytest.raises(ValueError):
        validate_amendment(prior, prior["contract"])
