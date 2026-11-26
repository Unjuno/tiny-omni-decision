import copy

import pytest


def fixture():
    metric = dict(accuracy=0.8, nll=0.4, brier=0.2, ece=0.02)
    baseline = dict(metric, accuracy=0.81)
    return dict(
        policies={
            p: dict(
                baseline=baseline,
                calibrated=dict(train=dict(metric, accuracy=0.9), validation=metric),
                strata=[],
                eligible_for_runtime_trial=False,
            )
            for p in ["random", "hard"]
        },
        eligible_for_runtime_trial=False,
    )


def test_failed_gate_is_preserved_and_gap_uses_actual_metrics():
    from scripts.report_av_slot_weights import decision

    data = fixture()
    summary = decision(data)
    assert not summary["eligible_for_runtime_trial"]
    assert summary["action"] == "CLOSE_SIX_COEFFICIENT_COMPARISON_KEEP_PARENT"
    assert summary["policies"]["random"]["train_minus_validation_accuracy_pp"] == pytest.approx(10)
    assert summary["policies"]["random"]["candidate_minus_baseline"]["accuracy"] == pytest.approx(
        -0.01
    )
    corrupted = copy.deepcopy(data)
    corrupted["eligible_for_runtime_trial"] = True
    with pytest.raises(ValueError, match="eligibility"):
        decision(corrupted)


def test_declared_hash_collision_is_rejected():
    from scripts.report_av_slot_weights import merge_pins

    pins = {"same": "a" * 64}
    merge_pins(pins, {"new": "b" * 64})
    assert len(pins) == 2
    with pytest.raises(ValueError, match="conflicting"):
        merge_pins(pins, {"same": "b" * 64})


def test_publication_requires_actual_replay_and_phase_separation():
    from scripts.report_av_slot_weights import admit

    common = dict(
        status="PASS_FIXED_SLOT_WEIGHT_COMPARISON",
        smoke=False,
        iterations=512,
        fit_parameter_count=6,
        base_model_updates=0,
        gpu_used=False,
        encoders_loaded=False,
        sealed_audit_opened=False,
        source_and_base_weights_unchanged=True,
        weights_frozen_before_calibration=True,
        readout_frozen_before_validation=True,
        fresh_process_replay=False,
    )
    replay = dict(common, fresh_process_replay=True)
    smoke = dict(
        status="PASS_TRAIN_ONLY_SLOT_WEIGHT_SMOKE",
        calibration_evaluated=False,
        validation_evaluated=False,
    )
    admit(common, replay, smoke)
    with pytest.raises(ValueError):
        admit(common, common, smoke)
    with pytest.raises(ValueError):
        admit(dict(common, weights_frozen_before_calibration=False), replay, smoke)
    with pytest.raises(ValueError):
        admit(common, replay, dict(smoke, validation_evaluated=True))
