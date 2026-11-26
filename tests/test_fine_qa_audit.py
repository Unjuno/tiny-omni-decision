import numpy as np
import pytest

from scripts import audit_fine_grouped_qa as module
from scripts.audit_fine_grouped_qa import metrics, scene_bootstrap


def test_metrics_and_changed_condition_pair_counts_are_independent():
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], np.float64)
    y = np.array([0, 1, 2, 2], np.int64)
    m = metrics(p, np.log(p), y)
    assert m["accuracy"] == 0.75 and m["both_correct"] == 0.5
    assert m["changed"]["both_correct"] == 1 and m["unchanged"]["both_correct"] == 0
    assert m["initial"]["accuracy"] == 0.5 and m["final"]["accuracy"] == 1
    assert abs(m["ece"] - 0.05) < 1e-12
    assert abs(m["nll"] - (-3 * np.log(0.8) - np.log(0.1)) / 4) < 1e-12


def test_scene_bootstrap_keeps_two_conditions_together():
    a = np.tile([0.8, 0.1, 0.1], (4, 1))
    b = np.tile([0.1, 0.8, 0.1], (4, 1))
    y = np.array([0, 0, 1, 1], np.int64)
    m = scene_bootstrap(a, np.log(a), b, np.log(b), y)
    assert m["accuracy"]["difference"] == 0
    assert m["accuracy"]["interval95"] == [-1, 1]
    assert m["videos"] == 2 and m["repetitions"] == 10000 and m["seed"] == 17
    with pytest.raises(ValueError):
        metrics(a, np.log(a), y[:3])


def test_strict_replay_rejects_dtype_and_signed_zero_changes():
    a = np.array([0.0, 1.0], np.float32)
    module.require_bytes(a, a.copy())
    with pytest.raises(ValueError):
        module.require_bytes(a, a.astype(np.float64))
    b = a.copy()
    b[0] = -0.0
    with pytest.raises(ValueError):
        module.require_bytes(a, b)


def test_bound_config_checks_effective_mode_digest_split_and_terminal_references(tmp_path):
    import json

    (tmp_path / "split.json").write_text("fixed split")
    contract = dict(split_sha256=module.sha(tmp_path / "split.json"), seed=17)
    config = dict(contract=contract, mode="replay")
    (tmp_path / "config.json").write_text(json.dumps(config))
    evidence = dict(
        contract=contract,
        contract_sha256=module.digest(contract),
        config_sha256=module.sha(tmp_path / "config.json"),
        arms={},
    )
    (tmp_path / "result.json").write_text(json.dumps(evidence))
    pins = module.evidence_pins(tmp_path, evidence, "replay")
    module.verify_pins(pins)
    with pytest.raises(ValueError):
        module.evidence_pins(tmp_path, evidence, "main")
    (tmp_path / "split.json").write_text("modified")
    with pytest.raises(ValueError):
        module.verify_pins(pins)
    with pytest.raises(ValueError):
        module.evidence_pins(tmp_path, evidence, "replay")
