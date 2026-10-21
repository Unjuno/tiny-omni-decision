from copy import deepcopy

import pytest


def test_error_attribution_separates_condition_selection_and_value_errors():
    import json

    from scripts.publish_condition_contrast import error_attribution

    state = json.dumps(dict(color="red", count=2))
    base = dict(state=state, option_ids=["color:red", "color:blue", "count:2", "count:3"])
    rows = [dict(base, target_id="color:red") for _ in range(4)]
    actual = error_attribution(rows, ["color:red", "color:blue", "count:2", "count:3"])
    assert actual == dict(
        questions=4,
        correct=1,
        wrong_field=2,
        correct_field_wrong_value=1,
        wrong_field_value_matches_state=1,
        target_to_predicted_field={"color": {"color": 2, "count": 2}},
    )
    with pytest.raises(ValueError):
        error_attribution(rows, ["color:red"])
    with pytest.raises(ValueError, match="candidate"):
        error_attribution(rows[:1], ["color:unknown"])


def test_run_provenance_rejects_self_consistent_stale_inputs_and_code(tmp_path):
    from scripts.evaluate_semantic_state_bridge import sha
    from scripts.publish_condition_contrast import verify_run_provenance

    data = tmp_path / "corpus.json"
    code = tmp_path / "runner.py"
    data.write_text("current corpus")
    code.write_text("frozen code")
    inputs, codes = {str(data): sha(data)}, {str(code): sha(code)}
    run = dict(input_pins=inputs, code_pins=codes)
    verify_run_provenance(run, inputs, codes)
    for key in ("input_pins", "code_pins"):
        stale = deepcopy(run)
        stale[key] = {"stale": "0" * 64}
        with pytest.raises(ValueError, match="provenance"):
            verify_run_provenance(stale, inputs, codes)
    data.write_text("changed after manifest creation")
    with pytest.raises(ValueError):
        verify_run_provenance(run, inputs, codes)


def test_transfer_provenance_requires_exact_current_fit_manifests_and_artifacts(tmp_path):
    from scripts.evaluate_semantic_state_bridge import sha
    from scripts.publish_condition_contrast import fit_provenance, verify_run_provenance

    fit = tmp_path / "fit"
    fit.mkdir()
    (fit / "result.json").write_text("current fit manifest")
    (fit / "final.pt").write_text("current weights")
    manifest = dict(artifacts={"final.pt": sha(fit / "final.pt")})
    pins = fit_provenance(fit, manifest)
    code = tmp_path / "runner.py"
    code.write_text("frozen code")
    codes = {str(code): sha(code)}
    assert pins == {str(fit / p): sha(fit / p) for p in ("result.json", "final.pt")}
    verify_run_provenance(dict(input_pins=pins, code_pins=codes), pins, codes)
    stale = dict(pins)
    stale[str(fit / "result.json")] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        verify_run_provenance(dict(input_pins=stale, code_pins=codes), pins, codes)
    with pytest.raises(ValueError, match="provenance"):
        verify_run_provenance(dict(input_pins={}, code_pins=codes), pins, codes)


def test_mechanism_gate_requires_both_conditions_and_retention():
    from scripts.publish_condition_contrast import mechanism_failures

    base = dict(accuracy=0.5, nll=1.0, brier=0.6, ece=0.1)
    improved = dict(accuracy=0.6, nll=0.8, brier=0.5, ece=0.1)
    names = ("focus_descriptions", "complement_descriptions")
    models = {
        label: {name: dict(overall=dict(m)) for name in names}
        for label, m in [("control", base), ("candidate", improved)]
    }
    intervals = {name: dict(lower=[0.02, -0.3]) for name in names}
    assert mechanism_failures(models, intervals, True) == []
    assert mechanism_failures(models, intervals, False) == ["final_retention"]
    bad = deepcopy(models)
    bad["candidate"]["complement_descriptions"]["overall"]["ece"] = 0.13
    assert "complement_descriptions:ece" in mechanism_failures(bad, intervals, True)
    bad = deepcopy(intervals)
    bad["complement_descriptions"]["lower"][0] = 0
    assert "complement_descriptions:paired_lower" in mechanism_failures(models, bad, True)


def test_fit_curve_rejects_wrong_counts_steps_and_selector():
    import pytest

    from scripts.publish_condition_contrast import verify_curve

    rows = [
        dict(
            step=s,
            eligible=True,
            binding_nll=2 - s / 512,
            consumed=s * 8,
            unique_questions=s * 8,
            source_counts={} if not s else dict(binding=s * 4, original=s * 2, curriculum=s * 2),
        )
        for s in (0, 128, 256, 384, 512)
    ]
    verify_curve(rows, 512)
    for key, value in [("consumed", 123), ("unique_questions", 123), ("step", 511)]:
        changed = deepcopy(rows)
        changed[-1][key] = value
        with pytest.raises(ValueError):
            verify_curve(changed, 512)
    with pytest.raises(ValueError):
        verify_curve(rows, 0)
