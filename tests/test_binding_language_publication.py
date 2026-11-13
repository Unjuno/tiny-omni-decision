import copy

import pytest

from scripts.binding_language import CONDITIONS
from scripts.publish_binding_language import check_run_modes, cross_run_checks


def fixture():
    selected = {
        f"reload-{source}-{suffix}"
        for source in ("binding", "original", "curriculum", "fixed_train")
        for suffix in ("scores.npy", "probabilities.npy", "metrics.json")
    }
    transfer = {
        f"{model}-{condition}-{suffix}"
        for model in ("control", "candidate")
        for condition in CONDITIONS
        for suffix in ("scores.npy", "probabilities.npy", "predictions.jsonl")
    }
    transfer |= {"metrics.json", "effective-config.json"}
    fit = dict(
        input_pins={"input": "a"},
        code_pins={"code": "b"},
        best_step=512,
        selected_sha256="c",
        artifacts=dict.fromkeys(selected, "d"),
        selected_reload_exact=[
            f"validation-512-{n}-scores.npy" for n in ("binding", "original", "curriculum")
        ]
        + ["fixed-train-512-scores.npy"],
    )
    replay = dict(copy.deepcopy(fit), fresh_replay_exact=sorted(selected))
    evaluated = dict(
        input_pins={"input": "a", "fit/result.json": "fit-hash"},
        code_pins={"code": "b"},
        artifacts=dict.fromkeys(transfer, "e"),
    )
    rerun = dict(copy.deepcopy(evaluated), fresh_replay_files_exact=sorted(transfer))
    return dict(fit=fit, selected_replay=replay, transfer=evaluated, transfer_replay=rerun)


def test_publication_links_exact_results_sources_and_artifact_sets():
    records = fixture()
    cross_run_checks(records, "fit/result.json", "fit-hash")
    for run, key, replacement in (
        ("selected_replay", "input_pins", {"input": "wrong"}),
        ("selected_replay", "best_step", 384),
        ("transfer_replay", "code_pins", {"code": "different"}),
        ("transfer", "input_pins", {"input": "a", "fit/result.json": "other-fit"}),
        ("selected_replay", "fresh_replay_exact", ["wrong"] * 12),
        ("transfer_replay", "fresh_replay_files_exact", ["wrong"] * 44),
    ):
        changed = copy.deepcopy(records)
        changed[run][key] = replacement
        with pytest.raises(ValueError):
            cross_run_checks(changed, "fit/result.json", "fit-hash")


def test_publication_rejects_inconsistent_run_modes():
    modes = {
        "smoke": dict(steps=2),
        "fit": dict(steps=512, transfer_evaluated=False),
        "selected_replay": dict(steps=0, transfer_evaluated=False),
        "transfer": dict(model_updates=0),
        "transfer_replay": dict(model_updates=0),
    }
    for value in modes.values():
        value.update(cloud_cost=0, sealed_audit_loaded=False)
    check_run_modes(modes)
    for run, key, value in (
        ("smoke", "steps", 3),
        ("fit", "transfer_evaluated", True),
        ("selected_replay", "steps", 1),
        ("transfer", "model_updates", 1),
        ("transfer_replay", "cloud_cost", 1),
        ("transfer", "sealed_audit_loaded", True),
    ):
        changed = copy.deepcopy(modes)
        changed[run][key] = value
        with pytest.raises(ValueError):
            check_run_modes(changed)


def test_publication_rejects_changed_replay_payload_hashes():
    records = fixture()
    for run in ("selected_replay", "transfer_replay"):
        changed = copy.deepcopy(records)
        filename = next(iter(changed[run]["artifacts"]))
        changed[run]["artifacts"][filename] = "wrong"
        with pytest.raises(ValueError):
            cross_run_checks(changed, "fit/result.json", "fit-hash")
