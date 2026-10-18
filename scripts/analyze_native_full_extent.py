"""Verify the fixed full-extent intervention from saved development predictions."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analyze_native_long_budget import per_example
from scripts.analyze_native_support import delta_metrics
from scripts.native_consumed_evaluation import common_task_gap
from scripts.native_fit import accounting, choose_best, summarize
from scripts.native_long_budget import EVALUATION_STEPS, MAX_STEPS, learning_rate
from scripts.native_video_cache import sha

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-full-extent-comparison-01"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def retention(control, candidate):
    delta = {
        m: candidate["modality_macro"][m]["accuracy"] - value["accuracy"]
        for m, value in control["modality_macro"].items()
    }
    checks = dict(
        selected_macro_nll_improves=candidate["macro"]["nll"] < control["macro"]["nll"],
        mean_visual_accuracy_improves=(delta["image"] + delta["video"]) / 2 > 0,
        no_modality_declines_over_two_pp=min(delta.values()) >= -0.02,
    )
    return dict(
        checks=checks,
        retain_for_research=all(checks.values()),
        product_promoted=False,
        modality_accuracy_deltas=delta,
    )


def visual_bootstrap(rows, first, second, seed=17, resamples=5000):
    """Paired image/scene clusters, all questions on an asset resampled together."""
    vectors = {}
    for metric in ("accuracy", "nll", "brier"):
        delta = np.asarray(second[metric]) - np.asarray(first[metric])
        if delta.shape != (len(rows),) or not np.isfinite(delta).all():
            raise ValueError("finite paired per-example vectors required")
        vectors[metric] = delta
    result = {}
    for subset in ("image", "video", "video_temporal"):
        indices = [
            i
            for i, r in enumerate(rows)
            if r["modality"] == subset.split("_")[0]
            and (subset != "video_temporal" or r["task"] == "temporal_descriptive")
        ]
        groups = sorted({rows[i]["observation"] for i in indices})
        if len(groups) < 2:
            raise ValueError("at least two underlying assets per visual subset required")
        by_group = [[i for i in indices if rows[i]["observation"] == g] for g in groups]
        sizes = np.array([len(ix) for ix in by_group])
        draws = np.random.default_rng(seed).integers(0, len(groups), (resamples, len(groups)))
        item = dict(
            questions=len(indices),
            assets=len(groups),
            seed=seed,
            resamples=resamples,
            interval="paired image/scene clusters; percentile95%; NumPy linear quantile",
            caveat="Observed development validation; checkpoint selection bias applies.",
        )
        for metric, delta in vectors.items():
            sums = np.array([delta[ix].sum() for ix in by_group])
            values = sums[draws].sum(1) / sizes[draws].sum(1)
            item[metric] = dict(
                candidate_minus_control=float(delta[indices].mean()),
                low=float(np.quantile(values, 0.025, method="linear")),
                high=float(np.quantile(values, 0.975, method="linear")),
            )
        result[subset] = item
    return result


def main():
    fit = ROOT / "native-full-extent-fit-01"
    parent = ROOT / "native-long-budget-fit-01"
    replay_dir = ROOT / "native-full-extent-replay-01"
    result, previous, replay = (load(p / "result.json") for p in (fit, parent, replay_dir))
    assert result["status"] == "PASS_NATIVE_FULL_EXTENT_TRAIN"
    assert previous["status"] == "PASS_NATIVE_LONG_BUDGET_TRAIN"
    assert replay["status"] == "PASS_NATIVE_FULL_EXTENT_REPLAY"
    assert result["step"] == previous["step"] == replay["step"] == MAX_STEPS
    assert replay["reference_sha256"] == sha(fit / "result.json")
    assert replay["replay_max_abs"] == result["initial_nonvisual_parity_max_abs"] == 0
    assert result["control_sampling_exact"] and result["visual_banks_replaced"] == 458
    resources = {}
    for phase in ("smoke", "fit", "replay"):
        folder = ROOT / (
            "native-full-extent-readout-smoke-01"
            if phase == "smoke"
            else f"native-full-extent-{phase}-01"
        )
        receipt = load(folder / "result.json")
        assert not receipt["sealed_audit_loaded"] and not receipt["calibration_loaded"]
        assert receipt["encoder_updates"] == receipt["cloud_cost"] == 0
        for name, digest in receipt["artifacts"].items():
            assert sha(folder / name) == digest, (phase, name)
        resources[phase] = {
            k: receipt[k]
            for k in (
                "seconds",
                "training_seconds",
                "evaluation_seconds",
                "peak_vram_bytes",
                "peak_vram_reserved_bytes",
                "peak_working_set_bytes",
                "started_utc",
                "finished_utc",
            )
        }
        resources[phase]["receipt_sha256"] = sha(folder / "result.json")
    for path, digest in result["validated_hashes"].items():
        assert sha(Path(path)) == digest, path
    for name, digest in previous["artifacts"].items():
        assert sha(parent / name) == digest, name
    encoder = {}
    for label, folder in (
        ("failed_smoke", ROOT / "native-full-extent-smoke-01"),
        ("smoke", ROOT / "native-full-extent-smoke-02"),
        ("full", ROOT / "native-full-extent-cache-01"),
    ):
        # Encoder smoke and readout smoke have distinct directory names below.
        receipt = load(folder / "result.json")
        encoder[label] = {
            k: v
            for k, v in receipt.items()
            if k not in ("source_pins", "validated_hashes", "artifacts", "traceback")
        }
        encoder[label]["receipt_sha256"] = sha(folder / "result.json")
        for name, digest in receipt.get("artifacts", {}).items():
            assert sha(folder / name) == digest, (label, name)
    assert encoder["full"]["status"] == "PASS_FULL_EXTENT_FULL"
    train = [
        json.loads(s)
        for s in (ROOT / "native-common-corpus-02/train.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ]
    rows = [
        json.loads(s)
        for s in (ROOT / "native-common-corpus-02/validation.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ]
    by_id = {r["id"]: r for r in train}
    probe = [by_id[i] for i in load(fit / "train-probe-ids.json")]
    assert len(probe) == 6656 and len(rows) == 24346
    for name in ("sampling.json", "train-probe-ids.json", "validation-ids.json", "lr-plan.json"):
        assert sha(fit / name) == sha(parent / name), name
    assert load(fit / "validation-ids.json") == [
        dict(id=r["id"], option_ids=r["option_ids"], target_index=r["target_index"]) for r in rows
    ]
    sampling = load(fit / "sampling.json")
    order, occurrences = sampling["order"], sampling["occurrence_plan"]["occurrences"]
    history = [json.loads(s) for s in (fit / "training-history.jsonl").read_text().splitlines()]
    assert len(history) == MAX_STEPS
    for step, h in enumerate(history, 1):
        assert h["step"] == step and h["lr"] == learning_rate(step)
        assert h["sample_ids"] == [train[i]["id"] for i in order[step - 1]]
        assert h["occurrence_ids"] == occurrences[step - 1]
        assert np.isfinite([h["ce"], h["gradient_norm_before_clip"]]).all()
    assert result["actual_accounting"] == accounting([train[i] for b in order for i in b])
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
    assert result["selected_metrics"] == next(
        h["validation"] for h in curve if h["step"] == best_step
    )
    initial, original_initial = (np.load(p / "validation-0000.npy") for p in (fit, parent))
    nonvisual = [i for i, r in enumerate(rows) if r["modality"] in ("text", "audio")]
    np.testing.assert_array_equal(initial[nonvisual], original_initial[nonvisual])
    parent_curve = load(parent / "learning-curve.json")
    original_steps = {h["step"]: h["validation"] for h in parent_curve}
    comparisons = {}
    for label, a, b in (
        ("selected", previous["best_step"], best_step),
        ("equal8192", 8192, 8192),
        ("equal32768", MAX_STEPS, MAX_STEPS),
    ):
        control = original_steps[a]
        candidate = next(h["validation"] for h in curve if h["step"] == b)
        vectors = [
            per_example(rows, np.load(p / f"validation-{s:04}.npy"))
            for p, s in ((parent, a), (fit, b))
        ]
        comparisons[label] = dict(
            control_step=a,
            candidate_step=b,
            control=control,
            candidate=candidate,
            deltas=delta_metrics(control, candidate),
            paired_visual_bootstrap=visual_bootstrap(rows, *vectors),
        )
    preserve = load(ROOT / "native-full-extent-verification-01/preserved-artifacts.json")
    for path, digest in preserve.items():
        assert sha(Path(path)) == digest, path
    selected = comparisons["selected"]
    summary = dict(
        status="PASS_FULL_EXTENT_EXECUTION_AND_REPLAY_AUDIT",
        goal_complete=False,
        product_promoted=False,
        blind_evaluation=False,
        retention=retention(selected["control"], selected["candidate"]),
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
                "visual_cache_sha256",
                "control_receipt_sha256",
            )
        },
        checkpoint_sha256={name: sha(fit / name) for name in ("initial.pt", "best.pt", "final.pt")},
        resources=resources,
        encoder=encoder,
        curve=curve,
        control_curve=parent_curve,
        comparisons=comparisons,
        preserved_artifact_count=len(preserve),
        unique_counts={key: dict(Counter(r[key] for r in probe)) for key in ("modality", "source")},
        endpoint_state_metrics=load(fit / "endpoint-state-metrics.json"),
        endpoint_train_metrics=load(fit / "endpoint-train-metrics.json"),
        caveat="One seed, observed development validation and checkpoint selection. Full extent "
        "also changes object scale/padding. No general capability ceiling or product claim.",
    )
    OUT.mkdir(exist_ok=False)
    (OUT / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(summary["status"], "best", best_step, summary["retention"])


if __name__ == "__main__":
    main()
