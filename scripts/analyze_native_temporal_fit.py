"""Pinned native baseline/LSTM result comparison; never loads models or audit data."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-common-comparison-01"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def per_example(rows, logits):
    assert logits.shape == (len(rows), 40) and np.isfinite(logits).all()
    z = logits.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    y = np.array([r["target_index"] for r in rows])
    prob = p[np.arange(len(y)), y]
    return dict(
        accuracy=(p.argmax(1) == y).astype(float),
        nll=-np.log(prob),
        brier=np.square(p).sum(1) - 2 * prob + 1,
    )


def bootstrap(rows, first, second):
    answer = {}
    for subset in ("all", "temporal_descriptive"):
        ids = [
            i
            for i, r in enumerate(rows)
            if r["modality"] == "video" and (subset == "all" or r["task"] == subset)
        ]
        groups = sorted({rows[i]["group"] for i in ids})
        by_group = [[i for i in ids if rows[i]["group"] == g] for g in groups]
        sizes = np.array([len(v) for v in by_group])
        rng = np.random.default_rng(17)
        draws = rng.integers(0, len(groups), (5000, len(groups)))
        item = dict(
            questions=len(ids),
            groups=len(groups),
            seed=17,
            resamples=5000,
            interval="percentile95%, NumPy linear quantile, paired underlying-scene clusters",
        )
        for metric in first:
            delta = second[metric] - first[metric]
            sums = np.array([delta[index].sum() for index in by_group])
            values = sums[draws].sum(1) / sizes[draws].sum(1)
            item[metric] = dict(
                lstm_minus_baseline=float(delta[ids].mean()),
                low=float(np.quantile(values, 0.025, method="linear")),
                high=float(np.quantile(values, 0.975, method="linear")),
            )
        answer[subset] = item
    return answer


def main():
    plan = load(OUT / "comparison-plan.json")
    assert (
        sha(OUT / "comparison-plan.json")
        == "7a06775ffc3fddf3c62f05d30928480b5acd4a8097fa7b1cb476eff386c9218a"
    )
    rows = [
        json.loads(s)
        for s in (ROOT / "native-common-corpus-02/validation.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ]
    image_labels = {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in (ROOT / "clevr4-sample-seed17-1024-512/val-labels.jsonl").open()
        )
    }
    text_relevance = {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in (ROOT / "condition-relevance-corpus-01/validation.jsonl").open()
        )
    }
    probe_ids = load(ROOT / "native-common-baseline-fit-01/train-probe-ids.json")
    probe_set = set(probe_ids)
    train_map = {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in (ROOT / "native-common-corpus-02/train.jsonl").open(encoding="utf8")
        )
        if r["id"] in probe_set
    }
    probe = [train_map[i] for i in probe_ids]
    runs = {}
    receipts = {}
    curves = {}
    vectors = {}
    coverage = {}
    error_analysis = {}
    for variant in ("baseline", "lstm"):
        run = ROOT / f"native-common-{variant}-fit-01"
        replay = ROOT / f"native-common-{variant}-replay-01"
        r = load(run / "result.json")
        rp = load(replay / "result.json")
        assert r["status"] == "PASS_NATIVE_FIT_TRAIN" and rp["status"] == "PASS_NATIVE_FIT_REPLAY"
        assert rp["reference_sha256"] == sha(run / "result.json") and r["step"] == 2048
        assert r["config_sha256"] == plan["training_config_sha256"] and r["variant"] == variant
        for directory, record in ((run, r), (replay, rp)):
            for name, digest in record["artifacts"].items():
                assert sha(directory / name) == digest, (variant, name)
        assert load(run / "validation-ids.json") == [
            dict(id=x["id"], option_ids=x["option_ids"], target_index=x["target_index"])
            for x in rows
        ]
        curve = load(run / "learning-curve.json")
        assert [h["step"] for h in curve] == list(range(0, 2049, 256))
        trajectory = [
            json.loads(s) for s in (run / "training-history.jsonl").read_text().splitlines()
        ]
        assert [h["step"] for h in trajectory] == list(range(1, 2049))
        assert all(
            np.isfinite(h["gradient_norm_before_clip"]) and np.isfinite(h["ce"]) for h in trajectory
        )
        seen = set()
        diagnostic = []
        for item in trajectory:
            seen.update(item["sample_ids"])
            if item["step"] % 256 == 0:
                values = per_example(
                    probe, np.load(run / f"train-probe-{item['step']:04}.npy", allow_pickle=False)
                )
                subsets = {}
                for modality in ("text", "image", "audio", "video"):
                    for used in (False, True):
                        ix = [
                            i
                            for i, r in enumerate(probe)
                            if r["modality"] == modality and (r["id"] in seen) == used
                        ]
                        subsets[f"{modality}:consumed={used}"] = dict(
                            count=len(ix),
                            metrics={key: float(val[ix].mean()) for key, val in values.items()}
                            if ix
                            else None,
                        )
                diagnostic.append(
                    dict(
                        step=item["step"],
                        probe_questions_consumed=len(probe_set & seen),
                        subsets=subsets,
                    )
                )
        coverage[variant] = diagnostic
        optimizer_seconds = [h["seconds"] for h in trajectory]
        receipts[variant] = r
        curves[variant] = curve
        runs[variant] = {
            k: r[k]
            for k in (
                "source_commit",
                "config_sha256",
                "started_utc",
                "finished_utc",
                "step",
                "best_step",
                "best_score",
                "selected_metrics",
                "final_metrics",
                "actual_accounting",
                "selected_accounting",
                "trainable_parameters",
                "total_parameters",
                "training_seconds",
                "evaluation_seconds",
                "seconds",
                "peak_vram_bytes",
                "peak_working_set_bytes",
                "device",
                "versions",
                "python",
                "cuda",
                "dtype",
                "tf32",
            )
        }
        runs[variant].update(
            path=str(run),
            receipt_sha256=sha(run / "result.json"),
            replay_receipt_sha256=sha(replay / "result.json"),
            replay_seconds=rp["seconds"],
            median_seconds_per_update=float(np.median(optimizer_seconds)),
            all_predictions_replayed_exactly=True,
            checkpoint_bytes={
                k: (run / k).stat().st_size for k in ("initial.pt", "best.pt", "final.pt")
            },
        )
        vectors[variant] = {
            label: per_example(rows, np.load(run / f"validation-{step:04}.npy", allow_pickle=False))
            for label, step in [("final", 2048), ("selected", r["best_step"])]
        }
        errors = {
            "image": {},
            "text_relevance": {},
            "status": "POSTHOC_SAVED_PREDICTION_DIAGNOSTIC_NO_SELECTION_CHANGE",
        }
        predicted = np.load(run / f"validation-{r['best_step']:04}.npy", allow_pickle=False).argmax(
            1
        )
        for row, j in zip(rows, predicted, strict=True):
            if row["source"] == "clevr4":
                field = row["task"]
                candidate = row["options"][j]
                pf, pv = candidate.split(": ", 1)
                truth = image_labels[row["group"]]
                c = errors["image"].setdefault(field, Counter())
                c["questions"] += 1
                c["correct"] += int(j == row["target_index"])
                c["requested_field"] += int(pf == field)
                c["true_but_unrequested"] += int(pf != field and str(truth[pf]) == pv)
            elif row["source"] == "text:condition_relevance_v1":
                original = text_relevance[row["original_id"]]
                assert row["options"] == original["options"]
                labels = original["supervision"]
                c = errors["text_relevance"].setdefault(original["field"], Counter())
                c["questions"] += 1
                c["correct"] += int(j == row["target_index"])
                c["requested_field"] += int(labels["request_relevant"][j])
                c["true_but_unrequested"] += int(
                    labels["fact_true"][j] and not labels["request_relevant"][j]
                )
        error_analysis[variant] = errors
        for label in vectors[variant]:
            stored = r["final_metrics" if label == "final" else "selected_metrics"]["overall"]
            for metric, array in vectors[variant][label].items():
                assert abs(float(array.mean()) - stored[metric]) < 1e-10, (variant, label, metric)
    assert receipts["baseline"]["validated_hashes"] == receipts["lstm"]["validated_hashes"]
    for f in ("sampling.json", "validation-ids.json", "train-probe-ids.json"):
        assert receipts["baseline"]["artifacts"][f] == receipts["lstm"]["artifacts"][f], f
    assert receipts["baseline"]["actual_accounting"] == receipts["lstm"]["actual_accounting"]
    np.testing.assert_array_equal(
        np.load(ROOT / "native-common-baseline-fit-01/validation-0000.npy"),
        np.load(ROOT / "native-common-lstm-fit-01/validation-0000.npy"),
    )
    report = dict(
        status="PASS_PAIRED_COMPARISON_AUDIT",
        plan=plan,
        plan_sha256=sha(OUT / "comparison-plan.json"),
        analysis_sha256=sha(__file__),
        runs=runs,
        curves=curves,
        posthoc_errors=error_analysis,
        train_probe_coverage=coverage,
        video_uncertainty={
            label: bootstrap(rows, vectors["baseline"][label], vectors["lstm"][label])
            for label in ("final", "selected")
        },
        initial_validation_logits_exactly_equal=True,
        source_pins_identical=True,
        sampler_and_validation_ids_identical=True,
        unique_validation_groups_by_source={
            s: len({r["group"] for r in rows if r["source"] == s})
            for s in sorted({r["source"] for r in rows})
        },
        limitations=[
            "Validation and checkpoint-selection conditional diagnostic, not a blind audit.",
            "One training seed; cluster bootstrap does not include training variation.",
            "Native readout runtime uses frozen cached features, excludes encoder execution.",
            "Parameter-matched nonrecurrent control has not run.",
            "No product promotion; all-task95-98% and joint/OOD gates unchanged.",
        ],
    )
    (OUT / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps({k: report[k] for k in ("status", "video_uncertainty")}, indent=2))


if __name__ == "__main__":
    main()
