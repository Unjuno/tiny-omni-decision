"""CPU-only accounting of the frozen condition-binding fit and inference replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import retention_ok, select_checkpoint
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.semantic_state_diagnostic import log_probabilities


def grouped_interval(differences, groups, repetitions=10_000, seed=17):
    """Paired question-weighted mean; resample whole states, keeping every row."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError("finite aligned difference matrix required")
    labels = sorted(set(groups))
    if len(labels) < 2 or repetitions < 1:
        raise ValueError("at least two groups and a positive repetition count required")
    index = {label: i for i, label in enumerate(labels)}
    sums = np.zeros((len(labels), values.shape[1]))
    counts = np.zeros(len(labels))
    for row, label in zip(values, groups, strict=True):
        sums[index[label]] += row
        counts[index[label]] += 1
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repetitions, 200):
        drawn = rng.integers(0, len(labels), (min(200, repetitions - start), len(labels)))
        samples.append(sums[drawn].sum(1) / counts[drawn].sum(1)[:, None])
    samples = np.concatenate(samples)
    bounds = np.quantile(samples, [0.025, 0.975], axis=0, method="linear")
    return dict(
        groups=len(labels),
        questions=len(values),
        repetitions=repetitions,
        seed=seed,
        method="paired whole-state bootstrap; question-weighted mean; percentile 95%, linear",
        difference=values.mean(0).tolist(),
        lower=bounds[0].tolist(),
        upper=bounds[1].tolist(),
    )


def score_difference(rows, old, new):
    if old.shape != new.shape or len(rows) != len(old):
        raise ValueError("aligned score arrays required")
    target = np.array([r["options"].index(r["target"]) for r in rows])
    nll_old = -log_probabilities(old)[np.arange(len(rows)), target]
    nll_new = -log_probabilities(new)[np.arange(len(rows)), target]
    correct_old = (old.argmax(1) == target).astype(float)
    correct_new = (new.argmax(1) == target).astype(float)
    differences = np.column_stack((correct_new - correct_old, nll_new - nll_old))
    return dict(
        columns=["accuracy_new_minus_parent", "nll_new_minus_parent"],
        **grouped_interval(differences, [r["group"] for r in rows]),
    )


def gate_failures(current, baseline):
    failures = []
    for source in ("original", "curriculum"):
        now = current["validation"][source]
        old = baseline["validation"][source]
        for task, metric in now["tasks"].items():
            change = metric["accuracy"] - old["tasks"][task]["accuracy"]
            if change < -0.01 - 1e-12:
                failures.append(
                    dict(source=source, task=task, metric="accuracy", change=change, limit=-0.01)
                )
        for metric, limit in (("nll", 0.02), ("brier", 0.01), ("ece", 0.02)):
            change = now["overall"][metric] - old["overall"][metric]
            if change > limit + 1e-12:
                failures.append(dict(source=source, metric=metric, change=change, limit=limit))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--primary-sha256", required=True)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary = args.root / "condition-binding-fit-01"
    replay = args.root / "condition-binding-selected-replay-01"
    corpus = args.root / "condition-binding-corpus-01"
    verify(
        {
            str(primary / "result.json"): args.primary_sha256,
            str(replay / "result.json"): args.replay_sha256,
        }
    )
    fit, repeat = read(primary / "result.json"), read(replay / "result.json")
    if fit["status"] != "PASS_FIXED_BINDING_TRAINING" or fit["steps"] != 512:
        raise ValueError("primary training incomplete")
    if repeat["status"] != "PASS_SELECTED_MODEL_REPLAY" or repeat["steps"] != 0:
        raise ValueError("fresh selected inference replay incomplete")
    for root, result in ((primary, fit), (replay, repeat)):
        verify({str(root / p): h for p, h in result["artifacts"].items()})
        verify(result["input_pins"])
        verify(result["code_pins"])
    if fit["selected_sha256"] != repeat["selected_sha256"]:
        raise ValueError("selected checkpoint changed")
    history = read(primary / "history.json")
    if [r["step"] for r in history] != [0, 128, 256, 384, 512]:
        raise ValueError("incomplete learning curve")
    if select_checkpoint(history) != fit["best_step"]:
        raise ValueError("selector mismatch")
    rows = lines(corpus / "validation.jsonl")
    old = np.load(primary / "validation-0-binding-scores.npy", allow_pickle=False)
    baseline = history[0]
    curve = []
    for item in history:
        metrics = {s: item["validation"][s]["overall"] for s in ("original", "curriculum")}
        tasks = {
            s + ":" + t: m["accuracy"]
            for s in metrics
            for t, m in item["validation"][s]["tasks"].items()
        }
        original = {s: baseline["validation"][s]["overall"] for s in metrics}
        original_tasks = {
            s + ":" + t: m["accuracy"]
            for s in metrics
            for t, m in baseline["validation"][s]["tasks"].items()
        }
        failures = gate_failures(item, baseline)
        if (not failures) != item["eligible"] or retention_ok(
            metrics, tasks, original, original_tasks
        ) != item["eligible"]:
            raise ValueError("retention gate reconstruction mismatch")
        new = np.load(primary / f"validation-{item['step']}-binding-scores.npy", allow_pickle=False)
        curve.append(
            dict(
                **item,
                gate_failures=failures,
                paired_binding_difference=score_difference(rows, old, new),
            )
        )
    transfer_intervals = {}
    for q in (0, 1):
        for o in (0, 1):
            condition = f"q{q}-o{o}"
            data = lines(corpus / f"transfer-{condition}.jsonl")
            old = np.load(primary / f"parent-{condition}-scores.npy", allow_pickle=False)
            new = np.load(primary / f"selected-{condition}-scores.npy", allow_pickle=False)
            transfer_intervals[condition] = score_difference(data, old, new)
    updates = read(primary / "updates.json")
    if [r["step"] for r in updates] != list(range(1, 513)):
        raise ValueError("incomplete update records")
    selected = next(r for r in curve if r["step"] == fit["best_step"])
    environment = read(primary / "effective-config.json")
    result = dict(
        status="PASS_COMPLETED_FIXED_COMPARISON",
        new_checkpoint_retained=fit["best_step"] != 0,
        product_goal_met=False,
        primary=fit,
        fresh_inference_replay=repeat,
        corpus=read(corpus / "result.json"),
        environment=environment["environment"],
        initialization=environment["initialized_from"],
        hashes={
            str(primary / "result.json"): args.primary_sha256,
            str(replay / "result.json"): args.replay_sha256,
            str(corpus / "result.json"): sha(corpus / "result.json"),
        },
        learning_curve=curve,
        selected_consumed_questions=selected["consumed"],
        full_consumed_questions=curve[-1]["consumed"],
        optimizer_updates_seconds=sum(r["seconds"] for r in updates),
        scheduled_validation_seconds=sum(r["evaluation_seconds"] for r in curve),
        gradient_norm_min=min(r["gradient_norm"] for r in updates),
        gradient_norm_max=max(r["gradient_norm"] for r in updates),
        transfer=read(primary / "transfer-metrics.json"),
        transfer_paired_difference=transfer_intervals,
        permutation=read(primary / "permutation.json"),
        uncertainty_scope="Development evidence, single seed; selector bias; one authored transfer "
        "wording family, no multiplicity correction or blind release inference.",
        summarizer_sha256=sha(__file__),
    )
    if args.output.exists():
        raise ValueError("refusing to overwrite a publication")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                sha256=sha(args.output),
                best_step=fit["best_step"],
                consumed=curve[-1]["consumed"],
            )
        )
    )


if __name__ == "__main__":
    main()
