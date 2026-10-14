"""Independent NumPy arithmetic over all saved predictions; no model/data inference."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar_metrics(p, logp, y):
    p, logp = np.asarray(p, np.float64), np.asarray(logp, np.float64)
    if p.shape != (len(y), 3) or logp.shape != p.shape or not len(y) or not np.isfinite(logp).all():
        raise ValueError("complete finite three-option outputs required")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities required")
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    np.testing.assert_allclose(np.exp(logp), p, atol=1e-6, rtol=1e-6)
    pred = p.argmax(1)
    confidence = p.max(1)
    correct = pred == y
    indices = np.minimum((confidence * 15).astype(int), 14)
    ece = 0.0
    for i in range(15):
        mask = indices == i
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return dict(
        count=len(y),
        accuracy=float(correct.mean()),
        nll=float(-logp[np.arange(len(y)), y].mean()),
        brier=float(((p - np.eye(3)[y]) ** 2).sum(1).mean()),
        ece=float(ece),
        mean_confidence=float(confidence.mean()),
        prediction_counts=np.bincount(pred, minlength=3).tolist(),
    )


def metrics(p, logp, y):
    y = np.asarray(y)
    if len(y) == 0 or len(y) % 2 or y.dtype != np.int64 or np.any(y < 0) or np.any(y > 2):
        raise ValueError("original paired int64 three-option targets required")
    result = scalar_metrics(p, logp, y)
    hits = (p.argmax(1) == y).reshape(-1, 2)
    result.update(
        both_correct=float(hits.all(1).mean()),
        both_correct_count=int(hits.all(1).sum()),
        pair_count=len(y) // 2,
    )
    for name, index in [("initial", slice(None, None, 2)), ("final", slice(1, None, 2))]:
        result[name] = scalar_metrics(p[index], logp[index], y[index])
    changed = y[::2] != y[1::2]
    for name, mask in [("changed", changed), ("unchanged", ~changed)]:
        rows = np.repeat(mask, 2)
        result[name] = (
            dict(
                scalar_metrics(p[rows], logp[rows], y[rows]),
                both_correct=float(hits[mask].all(1).mean()),
            )
            if mask.any()
            else None
        )
    return result


def scene_bootstrap(a, loga, b, logb, y):
    metrics(a, loga, y)
    metrics(b, logb, y)
    vectors = []
    for p, logs in [(a, loga), (b, logb)]:
        vectors.append(
            dict(
                accuracy=(p.argmax(1) == y).reshape(-1, 2).mean(1),
                nll=(-np.asarray(logs, np.float64)[np.arange(len(y)), y]).reshape(-1, 2).mean(1),
                brier=((np.asarray(p, np.float64) - np.eye(3)[y]) ** 2)
                .sum(1)
                .reshape(-1, 2)
                .mean(1),
            )
        )
    draws = np.random.default_rng(17).integers(0, len(y) // 2, (10000, len(y) // 2))
    result = dict(
        videos=len(y) // 2, repetitions=10000, seed=17, method="percentile", unit="video ID"
    )
    for key in vectors[0]:
        d = vectors[1][key] - vectors[0][key]
        result[key] = dict(
            difference=float(d.mean()),
            interval95=np.percentile(d[draws].mean(1), [2.5, 97.5]).tolist(),
        )
    return result


def compare(got, saved):
    for key in ("count", "accuracy", "nll", "brier", "ece"):
        np.testing.assert_allclose(got[key], saved[key], rtol=0, atol=1e-6)
    for key in ("initial", "final", "changed", "unchanged"):
        if got[key] is not None:
            compare_scalar(got[key], saved[key])
    np.testing.assert_allclose(got["both_correct"], saved["both_correct"], rtol=0, atol=0)


def compare_scalar(got, saved):
    for key in ("count", "accuracy", "nll", "brier", "ece"):
        np.testing.assert_allclose(got[key], saved[key], rtol=0, atol=1e-6)


def audit(run, run_sha, replay, replay_sha):
    if sha(run / "result.json") != run_sha or sha(replay / "result.json") != replay_sha:
        raise ValueError("immutable run/replay evidence changed")
    r = json.loads((run / "result.json").read_text())
    fresh = json.loads((replay / "result.json").read_text())
    if (
        r["status"] != "PASS_FIXED_GROUPED_FINE_QA_COMPARISON_ONLY"
        or fresh["status"] != "PASS_FRESH_ALL_FOUR_WEIGHT_REPLAYS_NOT_QUALITY"
        or fresh["contract"] != r["contract"]
        or sha(run / "config.json") != r["config_sha256"]
        or sha(run / "split.json") != r["contract"]["split_sha256"]
    ):
        raise ValueError("complete bound comparison/fresh replay required")
    split = json.loads((run / "split.json").read_text())
    a, b = split["fit"], split["selector"]
    for key in ("video_ids", "media_sha256"):
        if len(a[key]) != 79 or len(b[key]) != 20 or len(set(a[key] + b[key])) != 99:
            raise ValueError("cross-split content/identity overlap")
    labels = {}
    for name in ("fit", "selector"):
        with np.load(run / f"{name}-labels.npz", allow_pickle=False) as d:
            labels[name] = d["qa"].copy()
            if d["identities"].tolist() != split[name]["video_ids"]:
                raise ValueError("saved target identities changed")
        expected = [q["target"] for p in split[name]["original_questions"] for q in p["questions"]]
        np.testing.assert_array_equal(labels[name], expected)
    results = {}
    arms = ("coarse16", "fine32")
    orders_list = []
    for arm in arms:
        folder = run / arm
        for file, pin in r["arms"][arm]["hashes"].items():
            if sha(folder / file) != pin:
                raise ValueError("saved training artifact changed")
        orders = json.loads((folder / "orders.json").read_text())
        history = json.loads((folder / "history.json").read_text())
        if (
            len(orders) != 50
            or len(history) != 50
            or any(sorted(o) != list(range(79)) for o in orders)
        ):
            raise ValueError("incomplete fixed budget/sampling")
        orders_list.append(orders)
        curve = []
        for epoch, h in enumerate(history, 1):
            if (
                h["epoch"] != epoch
                or h["updates"] != epoch * 10
                or h["cumulative_video_presentations"] != epoch * 79
            ):
                raise ValueError("incorrect cumulative optimizer/exposure count")
            with np.load(folder / f"epoch-{epoch:02d}-predictions.npz", allow_pickle=False) as p:
                row = {s: metrics(p[s], p[s + "_logp"], labels[s]) for s in ("fit", "selector")}
            for s in row:
                compare(row[s], h[s])
            curve.append(
                dict(
                    epoch=epoch,
                    updates=epoch * 10,
                    **row,
                    accuracy_gap=row["fit"]["accuracy"] - row["selector"]["accuracy"],
                    nll_gap=row["selector"]["nll"] - row["fit"]["nll"],
                )
            )
        best = min(range(50), key=lambda i: curve[i]["selector"]["nll"]) + 1
        if best != r["arms"][arm]["best_epoch"]:
            raise ValueError("earliest strict NLL selector mismatch")
        for label, epoch in [("best", best), ("final", 50)]:
            path = replay / f"{arm}-{label}-replay.npz"
            if sha(path) != fresh["arms"][arm]["hashes"][path.name]:
                raise ValueError("fresh predictions changed")
            with (
                np.load(path, allow_pickle=False) as p,
                np.load(folder / f"epoch-{epoch:02d}-predictions.npz", allow_pickle=False) as o,
            ):
                for k in o.files:
                    np.testing.assert_array_equal(p[k], o[k])
        controls = {}
        with np.load(folder / "selected-controls.npz", allow_pickle=False) as p:
            for control in ("normal", "zero", "mismatch", "initial", "reverse"):
                controls[control] = metrics(p[control], p[control + "_logp"], labels["selector"])
                compare(controls[control], r["arms"][arm]["controls"][control])
        results[arm] = dict(
            best_epoch=best,
            final_epoch=50,
            updates=500,
            curve=curve,
            controls=controls,
            run_video_presentations=3950,
            selected_video_presentations=79 * best,
            unique_fit_video_assets=79,
            repeated_video_presentations=3871,
            unique_questions=158,
            run_question_presentations=7900,
            selected_question_presentations=158 * best,
        )
    if orders_list[0] != orders_list[1]:
        raise ValueError("matched full orders changed")
    bootstraps = {}
    for label in ("final", "selected"):
        values = []
        for arm in arms:
            ep = 50 if label == "final" else results[arm]["best_epoch"]
            with np.load(run / arm / f"epoch-{ep:02d}-predictions.npz", allow_pickle=False) as p:
                values.extend([p["selector"].copy(), p["selector_logp"].copy()])
        got = scene_bootstrap(*values, labels["selector"])
        for key in ("accuracy", "nll", "brier"):
            np.testing.assert_allclose(
                got[key]["difference"],
                r["paired_scene_bootstrap"][label][key]["difference"],
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                got[key]["interval95"],
                r["paired_scene_bootstrap"][label][key]["interval95"],
                rtol=0,
                atol=1e-6,
            )
        bootstraps[label] = got
    return dict(
        status="PASS_INDEPENDENT_PREDICTION_ACCOUNTING_AUDIT_NOT_QUALITY",
        run_result_sha256=run_sha,
        replay_result_sha256=replay_sha,
        arms=results,
        bootstrap=bootstraps,
        metrics_tolerance=1e-6,
        arithmetic="float64 independent NumPy; original stable log probabilities",
        limitation=(
            "known TRAIN scene selector; weak auxiliary labels not independently rebuilt; not blind"
        ),
        no_torch_or_model_loaded=True,
        official_validation_or_sealed_arrays_opened=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--replay-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    result = audit(args.run, args.run_sha, args.replay, args.replay_sha)
    result.update(seconds=time.perf_counter() - start, auditor_sha256=sha(Path(__file__)))
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(result["status"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
