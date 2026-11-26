import copy

import pytest

from scripts.condition_relevance_postfit import (
    freeze_comparison,
    require_completed_pair,
    transfer_contract,
)


def pair():
    common = {
        "status": "PASS_MATCHED_RELEVANCE_FIT",
        "smoke": False,
        "steps": 1024,
        "source_commit": "6fb617f233752f582af7daf0dd131ba7cc281a5c",
        "config_sha256": "942e86187fa587689165269b92e424d70fe5d615fbc3301342af48204fe1b5db",
        "config": {
            "seed": 17,
            "steps": 1024,
            "evaluation_interval": 256,
            "questions_per_update": [8, 8, 2],
            "relevance_coefficients": {"ce": 0.0, "relevance": 0.1},
            "corpus_sha256": "7593e36769996d07e3654694013a989a8cdcd8a4b58442a29bde4209feb26964",
            "revision": "262ae02f29173eec1c250f90804dc7edc677dcff",
            "auxiliary_parameters": 385,
        },
        "selected_reload_exact": ["validation", "fixed-train"],
        "frozen_and_sources_unchanged": True,
        "native_forward_equivalence": True,
        "sealed_audit_loaded": False,
        "transfer_loaded": False,
        "exact_resume": False,
        "best_step": 0,
        "selected_sha256": "a" * 64,
        "consumed": {
            "consumed": 18432,
            "unique_questions": 18432,
            "repeated_questions": 0,
            "unique_states": 2626,
            "native_assets": 0,
        },
        "selected_consumed": {"consumed": 0, "unique_questions": 0, "repeated_questions": 0},
        "input_pins": {"corpus/result.json": "b" * 64},
        "code_pins": {"trainer.py": "c" * 64},
        "frozen_sha256": "d" * 64,
        "artifacts": {
            name: "a" * 64
            for name in [
                "initial-delta.pt",
                "initial-retention-scores.npy",
                "sample-order.json",
                "sample-accounting.json",
                "token-preflight.json",
                "effective-config.json",
                "step-0-delta.pt",
                "step-1024-delta.pt",
                "best-delta.pt",
                "final-delta.pt",
                "validation-0-metrics.json",
                "validation-0-predictions.jsonl",
                "validation-0-scores.npy",
                "fixed-train-0-metrics.json",
                "fixed-train-0-predictions.jsonl",
                "fixed-train-0-scores.npy",
            ]
        },
    }
    ce = dict(copy.deepcopy(common), arm="ce", relevance_coefficient=0.0)
    aux = dict(copy.deepcopy(common), arm="relevance", relevance_coefficient=0.1)
    aux["best_step"] = 512
    aux["selected_consumed"].update(consumed=9216, unique_questions=9216)
    aux["artifacts"]["step-512-delta.pt"] = "e" * 64
    aux["artifacts"]["best-delta.pt"] = aux["selected_sha256"] = "e" * 64
    aux["artifacts"]["step-1024-delta.pt"] = "f" * 64
    aux["artifacts"]["final-delta.pt"] = "0" * 64
    for arm, run in [("ce", ce), ("relevance", aux)]:
        run["command"] = [
            "/repo/scripts/train_condition_relevance.py",
            "--arm",
            arm,
            "--output",
            "/runs/" + arm,
        ]
    return ce, aux


def effective_pair(ce, aux):
    values = {}
    for arm, run in [("ce", ce), ("relevance", aux)]:
        values[arm] = {
            key: copy.deepcopy(run[key])
            for key in (
                "config",
                "config_sha256",
                "input_pins",
                "code_pins",
                "frozen_sha256",
                "smoke",
                "arm",
                "relevance_coefficient",
                "command",
            )
        }
        values[arm].update(
            environment={
                "python": "3.11",
                "torch": "2.6",
                "cuda": "12.4",
                "gpu": "fixture",
                "dtype": "torch.float32",
                "attention": "eager",
                "packages": {"transformers": "5.6.2"},
            },
            trainable_names=["deberta.encoder.layer.0.attention.self.query_proj.weight"],
            auxiliary_parameters=385,
            initial_state="pinned_pretrained_no_delta",
        )
    return values


def freeze_pair(ce, aux, hashes):
    return freeze_comparison(ce, aux, hashes, effective_configs=effective_pair(ce, aux))


def transfer_pair(frozen, ce, aux, hashes, data, corpus_sha):
    return transfer_contract(
        frozen, ce, aux, hashes, data, corpus_sha, effective_configs=effective_pair(ce, aux)
    )


HASHES = {"ce": "1" * 64, "relevance": "2" * 64}


def test_complete_pair_freezes_all_roles_and_keeps_endpoint_separate():
    ce, aux = pair()
    require_completed_pair(ce, aux)
    frozen = freeze_pair(ce, aux, HASHES)
    assert frozen == freeze_pair(copy.deepcopy(ce), copy.deepcopy(aux), dict(HASHES))
    assert set(frozen["roles"]) == {
        "initial",
        "ce_endpoint",
        "relevance_endpoint",
        "ce_selected",
        "relevance_selected",
    }
    assert frozen["roles"]["initial"]["step"] == 0
    assert frozen["roles"]["ce_selected"]["step"] == 0
    assert frozen["roles"]["relevance_selected"]["step"] == 512
    assert frozen["roles"]["relevance_endpoint"]["step"] == 1024
    assert frozen["roles"]["relevance_selected"]["sha256"] == "e" * 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "RUNNING"),
        ("status", "PASS_SELECTED_RELEVANCE_REPLAY"),
        ("steps", 512),
        ("steps", True),
        ("smoke", True),
        ("sealed_audit_loaded", True),
        ("transfer_loaded", True),
        ("exact_resume", True),
        ("frozen_and_sources_unchanged", False),
        ("native_forward_equivalence", False),
        ("selected_reload_exact", ["validation"]),
        ("arm", "ce"),
        ("relevance_coefficient", 0.2),
        ("best_step", 128),
        ("best_step", False),
        ("selected_sha256", "9" * 64),
    ],
)
def test_incomplete_or_changed_run_rejected(field, value):
    ce, aux = pair()
    aux[field] = value
    with pytest.raises(ValueError):
        require_completed_pair(ce, aux)


@pytest.mark.parametrize("field", ["config", "input_pins", "code_pins", "frozen_sha256"])
def test_common_settings_and_frozen_parameters_must_match(field):
    ce, aux = pair()
    aux[field] = {"changed": True} if isinstance(aux[field], dict) else "9" * 64
    with pytest.raises(ValueError):
        require_completed_pair(ce, aux)


@pytest.mark.parametrize(
    "name",
    [
        "initial-delta.pt",
        "sample-order.json",
        "sample-accounting.json",
        "initial-retention-scores.npy",
        "token-preflight.json",
        "validation-0-scores.npy",
        "fixed-train-0-predictions.jsonl",
    ],
)
def test_initial_conditions_must_match(name):
    ce, aux = pair()
    aux["artifacts"][name] = "9" * 64
    with pytest.raises(ValueError):
        require_completed_pair(ce, aux)


def test_wrong_consumption_and_missing_selected_checkpoint_rejected():
    ce, aux = pair()
    aux["consumed"]["repeated_questions"] = 1
    with pytest.raises(ValueError):
        require_completed_pair(ce, aux)
    ce, aux = pair()
    aux["selected_consumed"]["consumed"] -= 1
    with pytest.raises(ValueError):
        require_completed_pair(ce, aux)
    ce, aux = pair()
    del aux["artifacts"]["step-512-delta.pt"]
    with pytest.raises(ValueError):
        require_completed_pair(ce, aux)


def corpus():
    return dict(
        status="PASS_FROZEN_CONDITION_RELEVANCE",
        sealed_audit_loaded=False,
        new_transfer_model_evaluated=False,
        old_transfer_payload_opened=False,
        old_state_partitions_opened=False,
        calibration_loaded_for_fit=False,
        artifacts={"transfer.jsonl": "3" * 64},
    )


def test_transfer_contract_requires_exact_freeze_and_corpus_hash():
    ce, aux = pair()
    frozen = freeze_pair(ce, aux, HASHES)
    expected = ce["config"]["corpus_sha256"]
    contract = transfer_pair(frozen, ce, aux, HASHES, corpus(), expected)
    assert contract == {
        "filename": "transfer.jsonl",
        "sha256": "3" * 64,
        "rows": 16384,
        "states": 256,
    }
    frozen["roles"]["ce_endpoint"]["step"] = 512
    with pytest.raises(ValueError):
        transfer_pair(frozen, ce, aux, HASHES, corpus(), expected)
    with pytest.raises(ValueError):
        transfer_pair(freeze_pair(ce, aux, HASHES), ce, aux, HASHES, corpus(), "4" * 64)


def test_incomplete_pair_rejected_before_corpus_inspection():
    class Unreadable(dict):
        def __getitem__(self, key):
            raise AssertionError("corpus metadata accessed before completion gate")

        def get(self, *args):
            raise AssertionError("corpus metadata accessed before completion gate")

    ce, aux = pair()
    frozen = freeze_pair(ce, aux, HASHES)
    aux["status"] = "RUNNING"
    with pytest.raises(ValueError):
        transfer_pair(frozen, ce, aux, HASHES, Unreadable(), ce["config"]["corpus_sha256"])


@pytest.mark.parametrize("change", ["missing_transfer", "audit_seen", "transfer_seen", "bad_hash"])
def test_unapproved_corpus_metadata_rejected(change):
    ce, aux = pair()
    data = corpus()
    if change == "missing_transfer":
        data["artifacts"] = {"sealed_audit.jsonl": "3" * 64}
    if change == "audit_seen":
        data["sealed_audit_loaded"] = True
    if change == "transfer_seen":
        data["new_transfer_model_evaluated"] = True
    if change == "bad_hash":
        data["artifacts"]["transfer.jsonl"] = "not-a-hash"
    with pytest.raises(ValueError):
        transfer_pair(
            freeze_pair(ce, aux, HASHES), ce, aux, HASHES, data, ce["config"]["corpus_sha256"]
        )


def test_arm_specific_effective_config_hashes_are_pinned_separately():
    ce, aux = pair()
    ce["artifacts"]["effective-config.json"] = "8" * 64
    aux["artifacts"]["effective-config.json"] = "9" * 64
    frozen = freeze_pair(ce, aux, HASHES)
    assert frozen["effective_config_hashes"] == {"ce": "8" * 64, "relevance": "9" * 64}
    assert "effective-config.json" not in frozen["common_artifact_hashes"]
    del aux["artifacts"]["effective-config.json"]
    with pytest.raises(ValueError):
        freeze_pair(ce, aux, HASHES)


@pytest.mark.parametrize(
    "flag",
    ["old_transfer_payload_opened", "old_state_partitions_opened", "calibration_loaded_for_fit"],
)
@pytest.mark.parametrize("value", [True, None, 0, "missing"])
def test_historical_access_provenance_must_be_explicitly_false(flag, value):
    ce, aux = pair()
    data = corpus()
    if value == "missing":
        del data[flag]
    else:
        data[flag] = value
    with pytest.raises(ValueError):
        transfer_pair(
            freeze_pair(ce, aux, HASHES), ce, aux, HASHES, data, ce["config"]["corpus_sha256"]
        )


def test_effective_contents_are_mandatory_for_freeze():
    ce, aux = pair()
    with pytest.raises(TypeError):
        freeze_comparison(ce, aux, HASHES)


def test_valid_common_effective_fields_are_frozen():
    ce, aux = pair()
    values = effective_pair(ce, aux)
    frozen = freeze_comparison(ce, aux, HASHES, effective_configs=values)
    assert len(frozen["common_effective_sha256"]) == 64
    changed = copy.deepcopy(values)
    for arm in changed:
        changed[arm]["environment"]["gpu"] = "other"
    assert frozen != freeze_comparison(ce, aux, HASHES, effective_configs=changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment", {"dtype": "torch.float16"}),
        ("trainable_names", ["deberta.encoder.rel_embeddings.weight"]),
        ("initial_state", "warm_start"),
        ("arm", "ce"),
        ("relevance_coefficient", 0.0),
        ("auxiliary_parameters", 386),
        ("config_sha256", "9" * 64),
        ("command", ["fit.py", "--smoke"]),
    ],
)
def test_effective_mismatch_or_override_is_rejected(field, value):
    ce, aux = pair()
    values = effective_pair(ce, aux)
    values["relevance"][field] = value
    with pytest.raises(ValueError):
        freeze_comparison(ce, aux, HASHES, effective_configs=values)


@pytest.mark.parametrize(
    "change",
    [
        "missing_field",
        "unknown_field",
        "missing_arm",
        "same_output",
        "extra_cli",
        "frozen_embeddings",
        "duplicate_names",
    ],
)
def test_effective_schema_and_execution_contract_are_fail_closed(change):
    ce, aux = pair()
    values = effective_pair(ce, aux)
    if change == "missing_field":
        del values["relevance"]["trainable_names"]
    elif change == "unknown_field":
        values["relevance"]["override"] = True
    elif change == "missing_arm":
        del values["relevance"]
    elif change in ("same_output", "extra_cli"):
        cmd = values["relevance"]["command"]
        if change == "same_output":
            cmd[-1] = values["ce"]["command"][-1]
        else:
            cmd += ["--replay-reference", "/old"]
        aux["command"] = copy.deepcopy(cmd)
    elif change in ("frozen_embeddings", "duplicate_names"):
        for arm in values:
            names = values[arm]["trainable_names"]
            names.append(
                "deberta.embeddings.word_embeddings.weight"
                if change == "frozen_embeddings"
                else names[0]
            )
    with pytest.raises(ValueError):
        freeze_comparison(ce, aux, HASHES, effective_configs=values)
