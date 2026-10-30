"""Verify the fixed native long curve from saved TRAIN/development predictions."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analyze_native_temporal_fit import bootstrap
from scripts.native_consumed_evaluation import common_task_gap
from scripts.native_fit import accounting, choose_best, summarize
from scripts.native_long_budget import EVALUATION_STEPS, MAX_STEPS, learning_rate
from scripts.native_video_cache import sha
from scripts.semantic_state_diagnostic import log_probabilities

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-long-budget-comparison-01"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def per_example(rows, logits):
    """Use the same stable log-softmax as the registered metric implementation."""
    assert logits.shape == (len(rows), 40) and np.isfinite(logits).all()
    logp = log_probabilities(logits)
    p = np.exp(logp)
    y = np.array([r["target_index"] for r in rows])
    prob = p[np.arange(len(y)), y]
    return dict(
        accuracy=(p.argmax(1) == y).astype(float),
        nll=-logp[np.arange(len(y)), y],
        brier=np.square(p).sum(1) - 2 * prob + 1,
    )


def main():
    fit = ROOT / "native-long-budget-fit-01"
    result = load(fit / "result.json")
    replay = load(ROOT / "native-long-budget-replay-01/result.json")
    assert result["status"] == "PASS_NATIVE_LONG_BUDGET_TRAIN"
    assert replay["status"] == "PASS_NATIVE_LONG_BUDGET_REPLAY"
    assert result["step"] == replay["step"] == MAX_STEPS
    assert replay["reference_sha256"] == sha(fit / "result.json")
    resources = {}
    for phase in ("smoke", "fit", "replay"):
        folder = ROOT / f"native-long-budget-{phase}-01"
        receipt = load(folder / "result.json")
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
    assert load(fit / "validation-ids.json") == [
        dict(id=r["id"], option_ids=r["option_ids"], target_index=r["target_index"]) for r in rows
    ]
    by_id = {r["id"]: r for r in train}
    probe = [by_id[i] for i in load(fit / "train-probe-ids.json")]
    assert len(probe) == 6656
    sampling = load(fit / "sampling.json")
    order = sampling["order"]
    occurrences = sampling["occurrence_plan"]["occurrences"]
    history = [json.loads(s) for s in (fit / "training-history.jsonl").read_text().splitlines()]
    assert len(history) == MAX_STEPS
    assert load(fit / "lr-plan.json") == [learning_rate(s) for s in range(1, MAX_STEPS + 1)]
    for step, h in enumerate(history, 1):
        assert h["step"] == step and h["lr"] == learning_rate(step)
        assert h["sample_ids"] == [train[i]["id"] for i in order[step - 1]]
        assert h["occurrence_ids"] == occurrences[step - 1]
        assert np.isfinite([h["ce"], h["gradient_norm_before_clip"]]).all()
    assert result["actual_accounting"] == accounting([train[i] for b in order for i in b])
    assert {i for h in history for i in h["sample_ids"]} == {r["id"] for r in probe}
    curve = load(fit / "learning-curve.json")
    assert tuple(h["step"] for h in curve) == EVALUATION_STEPS
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
    np.testing.assert_array_equal(
        np.load(fit / "validation-0000.npy"),
        np.load(ROOT / "native-coverage-repeat-fit-01/validation-0000.npy"),
    )
    vectors = [
        per_example(rows, np.load(fit / f"validation-{step:04}.npy")) for step in (2048, MAX_STEPS)
    ]
    uncertainty = bootstrap(rows, *vectors)
    for v in uncertainty.values():
        for metric in ("accuracy", "nll", "brier"):
            v[metric]["final_minus_2048"] = v[metric].pop("lstm_minus_baseline")
    curve_by_step = {h["step"]: h for h in curve}
    baseline = curve_by_step[2048]["validation"]
    deltas = {}
    for step in (8192, 16384, 24576, MAX_STEPS):
        m = curve_by_step[step]["validation"]
        deltas[step] = dict(
            macro={k: m["macro"][k] - v for k, v in baseline["macro"].items()},
            modalities={
                mod: {k: v - baseline["modality_macro"][mod][k] for k, v in value.items()}
                for mod, value in m["modality_macro"].items()
            },
        )
    segments = []
    previous = 0
    for step in EVALUATION_STEPS[1:]:
        hs = history[previous:step]
        norms = [h["gradient_norm_before_clip"] for h in hs]
        segments.append(
            dict(
                start_exclusive=previous,
                end=step,
                rolling_ce=float(np.mean([h["ce"] for h in hs])),
                rolling_accuracy=float(np.mean([h["accuracy"] for h in hs])),
                clipped_fraction=float(np.mean(np.asarray(norms) > 1)),
                median_gradient_norm=float(np.median(norms)),
                max_gradient_norm=max(norms),
                training_seconds=sum(h["seconds"] for h in hs),
            )
        )
        previous = step
    preserved = load(ROOT / "native-long-budget-verification-01/preserved-artifacts.json")
    for name, digest in preserved.items():
        assert sha(Path(name)) == digest, name
    common_keys = sorted(
        set(curve[0]["train_probe"]["tasks"]) & set(curve[0]["validation"]["tasks"])
    )
    task_timelines = {}
    for key in common_keys:
        task_timelines[key] = dict(
            points=[
                dict(
                    step=h["step"],
                    train=h["train_probe"]["tasks"][key],
                    validation=h["validation"]["tasks"][key],
                )
                for h in curve
            ],
            first_observed_train_95=next(
                (h["step"] for h in curve if h["train_probe"]["tasks"][key]["accuracy"] >= 0.95),
                None,
            ),
            first_observed_validation_95=next(
                (h["step"] for h in curve if h["validation"]["tasks"][key]["accuracy"] >= 0.95),
                None,
            ),
            caveat=(
                "Scheduled observations only; crossing95% does not prove grokking. "
                "Event imbalance remains explicit."
            ),
        )
    summary = dict(
        status="PASS_LONG_CURVE_EXECUTION_AND_REPLAY_AUDIT",
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
        segments=segments,
        within_run_deltas_from_2048=deltas,
        video_scene_bootstrap=uncertainty,
        historical_repeat_2048=load(ROOT / "native-coverage-repeat-fit-01/result.json")[
            "final_metrics"
        ],
        historical_caveat="Different cosine horizon; not a steps-only causal comparison.",
        preserved_artifact_count=len(preserved),
        endpoint_state_metrics=load(fit / "endpoint-state-metrics.json"),
        endpoint_train_metrics=load(fit / "endpoint-train-metrics.json"),
        all_common_task_timelines=task_timelines,
        unique_counts={key: dict(Counter(r[key] for r in probe)) for key in ("modality", "source")},
    )
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(summary["status"], "best", best_step, "all original artifacts unchanged", len(preserved))


if __name__ == "__main__":
    main()
