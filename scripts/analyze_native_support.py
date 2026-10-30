"""Audit the fixed positive-support intervention using saved development predictions."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.analyze_native_long_budget import per_example
from scripts.native_consumed_evaluation import common_task_gap
from scripts.native_fit import accounting, choose_best, summarize
from scripts.native_long_budget import EVALUATION_STEPS, MAX_STEPS, learning_rate
from scripts.native_support_sampling import repair_support
from scripts.native_video_cache import sha
from scripts.semantic_state_diagnostic import log_probabilities

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-support-comparison-01"
NAMES = ("accuracy", "nll", "brier", "ece")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recording_bootstrap(rows, first, second, seed=17, resamples=5000):
    """Paired cluster resampling; all windows/questions from a recording stay together."""
    indices = [i for i, row in enumerate(rows) if row["source"] == "starss"]
    groups = sorted({rows[i]["observation"].rsplit(":", 1)[0] for i in indices})
    if not indices or len(groups) < 2:
        raise ValueError("at least two STARSS recording clusters required")
    by_group = [
        [i for i in indices if rows[i]["observation"].rsplit(":", 1)[0] == g] for g in groups
    ]
    sizes = np.array([len(ix) for ix in by_group])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), (resamples, len(groups)))
    result = dict(
        questions=len(indices),
        recordings=len(groups),
        seed=seed,
        resamples=resamples,
        interval="paired recording clusters; percentile95%; linear",
        caveat="Observed development validation; checkpoint selection bias applies.",
    )
    for name in ("accuracy", "nll", "brier"):
        delta = np.asarray(second[name]) - np.asarray(first[name])
        if delta.shape != (len(rows),) or not np.isfinite(delta).all():
            raise ValueError("finite paired per-example vectors required")
        sums = np.array([delta[ix].sum() for ix in by_group])
        values = sums[draws].sum(1) / sizes[draws].sum(1)
        result[name] = dict(
            candidate_minus_control=float(delta[indices].mean()),
            low=float(np.quantile(values, 0.025, method="linear")),
            high=float(np.quantile(values, 0.975, method="linear")),
        )
    return result


def non_speech_balanced(metrics):
    events = metrics["event_classes"]
    defined = {
        k: v["balanced_accuracy"]
        for k, v in events.items()
        if k != "speech" and v["balanced_accuracy"] is not None
    }
    return dict(
        mean=float(np.mean(list(defined.values()))),
        included=list(defined),
        undefined=[
            k for k, v in events.items() if k != "speech" and v["balanced_accuracy"] is None
        ],
    )


def event_condition_diagnostic(rows, logits):
    """Post-hoc description of answers to different event questions on the same audio."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["source"] == "starss":
            if row["options"] != ["no", "yes"]:
                raise ValueError("canonical event options required")
            groups[row["observation"]].append(i)
    if not groups:
        raise ValueError("event observations required")
    classes = {rows[i]["task"] for indices in groups.values() for i in indices}
    probabilities = np.exp(log_probabilities(logits))[:, 1]
    prediction = np.asarray(logits).argmax(1)
    mixed, uniform_predictions, exact, spans, aucs = 0, 0, 0, [], []
    for indices in groups.values():
        if len(indices) != len(classes) or {rows[i]["task"] for i in indices} != classes:
            raise ValueError("all event questions once per observation required")
        target = np.array([rows[i]["target_index"] for i in indices])
        prob = probabilities[indices]
        pred = prediction[indices]
        exact += bool(np.array_equal(pred, target))
        spans.append(float(prob.max() - prob.min()))
        if target.any() and not target.all():
            mixed += 1
            uniform_predictions += len(set(pred)) == 1
            differences = prob[target == 1, None] - prob[None, target == 0]
            aucs.append(float(((differences > 0) + 0.5 * (differences == 0)).mean()))
    return dict(
        purpose="Post-hoc condition diagnostic; never used for selection or further tuning.",
        observations=len(groups),
        classes=len(classes),
        mixed_target_observations=mixed,
        all_event_questions_correct=exact,
        all_event_questions_accuracy=exact / len(groups),
        mixed_target_uniform_prediction_fraction=uniform_predictions / mixed if mixed else None,
        median_within_window_yes_probability_span=float(np.median(spans)),
        mixed_target_mean_event_ranking_auc=float(np.mean(aucs)) if aucs else None,
        caveat="Uniform binary answers do not alone prove the question was ignored. "
        "Cross-question yes scores are uncalibrated; ranking is descriptive only.",
    )


def delta_metrics(control, candidate):
    return dict(
        macro={k: candidate["macro"][k] - control["macro"][k] for k in NAMES},
        modalities={
            m: {k: candidate["modality_macro"][m][k] - v[k] for k in NAMES}
            for m, v in control["modality_macro"].items()
        },
        sources={
            s: {k: candidate["source_macro"][s][k] - v[k] for k in NAMES}
            for s, v in control["source_macro"].items()
        },
        non_speech_balanced=(
            non_speech_balanced(candidate)["mean"] - non_speech_balanced(control)["mean"]
        ),
    )


def main():
    fit = ROOT / "native-support-fit-01"
    parent = ROOT / "native-long-budget-fit-01"
    replay_dir = ROOT / "native-support-replay-01"
    result = load(fit / "result.json")
    replay = load(replay_dir / "result.json")
    previous = load(parent / "result.json")
    assert result["status"] == "PASS_NATIVE_SUPPORT_TRAIN"
    assert replay["status"] == "PASS_NATIVE_SUPPORT_REPLAY"
    assert previous["status"] == "PASS_NATIVE_LONG_BUDGET_TRAIN"
    assert result["step"] == replay["step"] == previous["step"] == MAX_STEPS
    assert replay["reference_sha256"] == sha(fit / "result.json")
    assert replay["replay_max_abs"] == replay["parent_original_train_replay_max_abs"] == 0
    resources = {}
    for phase in ("smoke", "fit", "replay"):
        folder = ROOT / f"native-support-{phase}-01"
        receipt = load(folder / "result.json")
        assert not receipt["sealed_audit_loaded"] and not receipt["calibration_loaded"]
        assert receipt["encoder_updates"] == 0 and receipt["cloud_cost"] == 0
        for name, digest in receipt["artifacts"].items():
            assert sha(folder / name) == digest, (phase, name)
        resources[phase] = {
            k: receipt[k]
            for k in (
                "seconds",
                "training_seconds",
                "evaluation_seconds",
                "peak_vram_bytes",
                "peak_working_set_bytes",
                "peak_vram_reserved_bytes",
                "started_utc",
                "finished_utc",
            )
        }
        resources[phase]["receipt_sha256"] = sha(folder / "result.json")
    for name, digest in result["validated_hashes"].items():
        assert sha(Path(name)) == digest, name
    for name, digest in previous["artifacts"].items():
        assert sha(parent / name) == digest, name
    train = [
        json.loads(s)
        for s in (ROOT / "native-common-corpus-02/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows = [
        json.loads(s)
        for s in (ROOT / "native-common-corpus-02/validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected = [
        dict(id=r["id"], option_ids=r["option_ids"], target_index=r["target_index"]) for r in rows
    ]
    assert load(fit / "validation-ids.json") == load(parent / "validation-ids.json") == expected
    by_id = {r["id"]: r for r in train}
    probe = [by_id[i] for i in load(fit / "train-probe-ids.json")]
    assert len(probe) == 6656
    sampling = load(fit / "sampling.json")
    old_sampling = load(parent / "sampling.json")
    plan = repair_support(train, old_sampling["order"])
    assert plan == load(ROOT / "native-support-plan-01/plan.json")
    assert sampling["order"] == plan["order"]
    assert sampling["occurrences"] == old_sampling["occurrence_plan"]["occurrences"]
    order = sampling["order"]
    unchanged_nonstarss = changed = 0
    for old, new in zip(old_sampling["order"], order, strict=True):
        for i, j in zip(old, new, strict=True):
            if train[i]["source"] != "starss":
                assert i == j
                unchanged_nonstarss += 1
            changed += i != j
    assert unchanged_nonstarss == 229376 and changed == 9344
    assert len(plan["replacements"]) == 292
    assert sha(fit / "initial.pt") == sha(parent / "initial.pt")
    assert (
        load(fit / "effective-config.json")["training"]
        == load(parent / "effective-config.json")["training"]
    )
    np.testing.assert_array_equal(
        np.load(fit / "validation-0000.npy", allow_pickle=False),
        np.load(parent / "validation-0000.npy", allow_pickle=False),
    )
    history = [json.loads(s) for s in (fit / "training-history.jsonl").read_text().splitlines()]
    assert len(history) == MAX_STEPS
    assert (
        load(fit / "lr-plan.json")
        == load(parent / "lr-plan.json")
        == [learning_rate(s) for s in range(1, MAX_STEPS + 1)]
    )
    for step, h in enumerate(history, 1):
        assert h["step"] == step and h["lr"] == learning_rate(step)
        assert h["sample_ids"] == [train[i]["id"] for i in order[step - 1]]
        assert h["occurrence_ids"] == sampling["occurrences"][step - 1]
        assert np.isfinite([h["ce"], h["gradient_norm_before_clip"]]).all()
    assert result["actual_accounting"] == accounting([train[i] for b in order for i in b])
    assert {i for h in history for i in h["sample_ids"]} == {r["id"] for r in probe}
    curve = load(fit / "learning-curve.json")
    parent_curve = load(parent / "learning-curve.json")
    assert (
        tuple(h["step"] for h in curve)
        == tuple(h["step"] for h in parent_curve)
        == EVALUATION_STEPS
    )
    best = best_step = None
    for h in curve:
        step = h["step"]
        scores = np.load(fit / f"validation-{step:04}.npy", allow_pickle=False)
        train_scores = np.load(fit / f"train-probe-{step:04}.npy", allow_pickle=False)
        assert h["validation"] == summarize(rows, scores)
        assert h["train_probe"] == summarize(probe, train_scores)
        assert h["common_task_gap"] == common_task_gap(probe, train_scores, rows, scores)
        assert h["accounting"] == accounting([train[i] for b in order[:step] for i in b])
        take = choose_best(best, h["validation"]["macro"]["nll"])
        assert h["selected"] == take
        if take:
            best, best_step = h["validation"]["macro"]["nll"], step
        assert h["best_step"] == best_step
    assert best_step == result["best_step"]
    assert result["selected_accounting"] == accounting(
        [train[i] for b in order[:best_step] for i in b]
    )
    parent_steps = {h["step"]: h for h in parent_curve}
    candidate_steps = {h["step"]: h for h in curve}
    comparisons = {}
    for label, old_step, new_step in (
        ("selected", previous["best_step"], best_step),
        ("equal8192", 8192, 8192),
        ("equal32768", MAX_STEPS, MAX_STEPS),
    ):
        a = parent_steps[old_step]["validation"]
        b = candidate_steps[new_step]["validation"]
        comparisons[label] = dict(
            control_step=old_step,
            candidate_step=new_step,
            control=a,
            candidate=b,
            delta=delta_metrics(a, b),
            non_speech_control=non_speech_balanced(a),
            non_speech_candidate=non_speech_balanced(b),
            posthoc_condition_diagnostic={
                name: event_condition_diagnostic(
                    rows, np.load(directory / f"validation-{step:04}.npy", allow_pickle=False)
                )
                for name, directory, step in (
                    ("control", parent, old_step),
                    ("candidate", fit, new_step),
                )
            },
            starss_paired_bootstrap=recording_bootstrap(
                rows,
                per_example(
                    rows, np.load(parent / f"validation-{old_step:04}.npy", allow_pickle=False)
                ),
                per_example(
                    rows, np.load(fit / f"validation-{new_step:04}.npy", allow_pickle=False)
                ),
            ),
        )
    selected = comparisons["selected"]
    drops = {m: x["accuracy"] for m, x in selected["delta"]["modalities"].items()}
    drops["lombard"] = selected["delta"]["sources"]["lombard"]["accuracy"]
    retention = dict(
        macro_nll_improved=selected["delta"]["macro"]["nll"] < 0,
        non_speech_balanced_improved=selected["delta"]["non_speech_balanced"] > 0,
        no_accuracy_drop_over_2pp=all(v >= -0.02 for v in drops.values()),
        accuracy_deltas=drops,
        product_promoted=False,
    )
    retention["retain_for_further_research"] = all(
        retention[k]
        for k in ("macro_nll_improved", "non_speech_balanced_improved", "no_accuracy_drop_over_2pp")
    )
    union = load(ROOT / "native-support-plan-01/union-train-ids.json")
    union_rows = [by_id[i] for i in union["union_ids"]]
    union_results = {}
    for name, directory, file in (
        ("control_best", replay_dir, "parent-best-train-union.npy"),
        ("control_final", replay_dir, "parent-final-train-union.npy"),
        ("candidate_final", fit, "endpoint-train-union.npy"),
    ):
        values = np.load(directory / file, allow_pickle=False)
        subsets = {"union": list(range(len(union_rows)))}
        for key in ("parent_ids", "additional_ids", "consumed_ids"):
            included = set(union[key])
            subsets[key] = [i for i, r in enumerate(union_rows) if r["id"] in included]
        union_results[name] = {
            k: summarize([union_rows[i] for i in ix], values[ix]) for k, ix in subsets.items()
        }
    segments = []
    start = 0
    for step in EVALUATION_STEPS[1:]:
        hs = history[start:step]
        norms = np.array([h["gradient_norm_before_clip"] for h in hs])
        segments.append(
            dict(
                start_exclusive=start,
                end=step,
                rolling_ce=float(np.mean([h["ce"] for h in hs])),
                rolling_accuracy=float(np.mean([h["accuracy"] for h in hs])),
                clipped_fraction=float(np.mean(norms > 1)),
                median_gradient_norm=float(np.median(norms)),
                max_gradient_norm=float(norms.max()),
                training_seconds=sum(h["seconds"] for h in hs),
            )
        )
        start = step
    preserved = load(ROOT / "native-support-verification-01/preserved-artifacts.json")
    for name, digest in preserved.items():
        assert sha(Path(name)) == digest, name
    summary = dict(
        status="PASS_SUPPORT_EXECUTION_AND_REPLAY_AUDIT",
        goal_complete=False,
        product_promoted=False,
        blind_evaluation=False,
        run={
            k: result[k]
            for k in (
                "source_commit",
                "command",
                "config_sha256",
                "initialization",
                "step",
                "best_step",
                "best_score",
                "actual_accounting",
                "selected_accounting",
                "final_metrics",
                "selected_metrics",
                "total_parameters",
                "trainable_parameters",
                "python",
                "versions",
                "cuda",
                "device",
                "dtype",
                "tf32",
            )
        },
        checkpoint_sha256={name: sha(fit / name) for name in ("initial.pt", "best.pt", "final.pt")},
        resources=resources,
        curve=curve,
        parent_curve=parent_curve,
        segments=segments,
        comparisons=comparisons,
        research_retention=retention,
        train_union=union_results,
        endpoint_state_metrics=load(fit / "endpoint-state-metrics.json"),
        endpoint_state_membership=load(fit / "endpoint-state-membership.json"),
        support_events=plan["events"],
        preserved_artifact_count=len(preserved),
        invariant_checks=dict(
            unchanged_nonstarss_presentations=unchanged_nonstarss,
            changed_presentations=changed,
            replaced_unique=292,
            initial_validation_max_abs=0,
            best_final_replay_max_abs=0,
            parent_original_train_replay_max_abs=0,
        ),
        unique_counts={key: dict(Counter(r[key] for r in probe)) for key in ("modality", "source")},
    )
    with (OUT / "comparison.json").open("x", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(summary["status"], "best", best_step, "retention", retention, flush=True)


if __name__ == "__main__":
    main()
