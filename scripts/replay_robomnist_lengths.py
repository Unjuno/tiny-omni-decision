"""Fresh full directory-length replay plus independent scalar metric audit."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.diagnose_robomnist_lengths import (
    ARMS,
    PARENT,
    PARENT_SHA,
    PLAN,
    PROTOCOL,
    TASKS,
    calculate,
    load_rows,
    peak_bytes,
    require_products,
    sha,
    write,
)
from scripts.robomnist_length_probe import task_logp, task_mapping


def require_scope(r):
    if (
        r["status"] != "PASS_FIXED_DIRECTORY_LENGTH_SHORTCUT_DIAGNOSTIC_ONLY"
        or r["unique_source_groups"] != 2135
        or r["split_counts"] != {"train": 1475, "validation": 660}
        or r["metadata_estimator_fit"] is not True
        or r["training_corpus_admission"] != "NOT_ADMITTED"
        or any(
            r[k] is not False
            for k in [
                "pretrained_models_loaded",
                "teacher_training",
                "gpu_used",
                "new_payloads_opened",
                "position_gt_opened",
                "sealed_audit_opened",
                "native_modality_quality_measured",
            ]
        )
        or not 0 < r["elapsed_seconds"] <= 120
        or not 0 < r["peak_cpu_bytes"] <= 2 * 2**30
    ):
        raise ValueError("complete directory-only shortcut scope required; never native quality")


def scalar_metrics(logp, targets):
    p = np.exp(logp)
    correct, confidences, briers, losses = [], [], [], []
    bins = [[] for _ in range(15)]
    for lp, row, target in zip(logp, p, targets, strict=True):
        pred = max(range(len(row)), key=lambda i: row[i])
        conf = float(row[pred])
        hit = int(pred == target)
        correct.append(hit)
        confidences.append(conf)
        briers.append(math.fsum((float(v) - int(i == target)) ** 2 for i, v in enumerate(row)))
        losses.append(-float(lp[target]))
        bins[min(int(conf * 15), 14)].append(hit - conf)
    n = len(targets)
    return dict(
        count=n,
        accuracy=sum(correct) / n,
        nll=math.fsum(losses) / n,
        brier=math.fsum(briers) / n,
        ece=math.fsum(abs(math.fsum(b)) for b in bins) / n,
        mean_confidence=math.fsum(confidences) / n,
    )


def compare_metrics(expected, actual):
    if set(expected) != set(actual) or any(
        not math.isclose(expected[k], actual[k], rel_tol=0, abs_tol=1e-12) for k in expected
    ):
        raise ValueError("independent scalar calibration/Accuracy/count metrics differ")


def accounting(rows):
    return dict(
        split_counts=dict(Counter(r["split"] for r in rows)),
        split_repetition_blocks={
            s: sorted({r["repetition"] for r in rows if r["split"] == s})
            for s in ["train", "validation"]
        },
        class_counts={
            s: {
                str(k): v
                for k, v in Counter(r["class_target"] for r in rows if r["split"] == s).items()
            }
            for s in ["train", "validation"]
        },
        group_order_sha256=hashlib.sha256(
            "\n".join(r["group_id"] for r in rows).encode()
        ).hexdigest(),
        split_assignment_sha256=hashlib.sha256(
            "\n".join(f"{r['group_id']}:{r['split']}" for r in rows).encode()
        ).hexdigest(),
    )


def require_accounting(report, rows):
    if any(report.get(k) != v for k, v in accounting(rows).items()):
        raise ValueError("actual source/split/order/class accounting differs")


def read_report(path):
    blob = path.read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def recheck_bindings(pins):
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError("source/artifact/report evidence changed during replay")


def audit(root):
    r, report_sha = read_report(root / "result.json")
    require_scope(r)
    c = json.loads((root / "config.json").read_bytes())
    if (
        sha(root / "config.json") != r["config_sha256"]
        or c["source_commit"] != r["source_commit"]
        or c["arms"] != ARMS
        or c["tasks"] != TASKS
        or c["seed"] != 17
        or c["bootstrap_draws"] != 2000
        or c["bootstrap_seed"] != 17
        or c["ece_bins"] != 15
        or c["nll"] != "stable log posterior, no clipping"
        or c["estimator"]
        != "train-standardized centroids; -0.5 distance squared; temperature1; no tuning"
    ):
        raise ValueError("exact fixed effective config and creation hash required")
    rows, x, parent = load_rows()
    require_accounting(r, rows)
    required = {**parent["validated_hashes"], str(PARENT): PARENT_SHA}
    for p in [
        Path("scripts/diagnose_robomnist_lengths.py"),
        Path("scripts/robomnist_length_probe.py"),
        Path("scripts/fetch_robomnist_input_smoke.py"),
        Path("scripts/replay_robomnist_directories.py"),
        Path("tests/test_robomnist_length_probe.py"),
        PROTOCOL,
        PLAN,
        Path(np.__file__),
        Path("src/tiny_omni_decision/decision_math.py"),
    ]:
        required[str(p.resolve())] = sha(p)
    if any(c["source_hashes"].get(p) != h for p, h in required.items()):
        raise ValueError("mandatory source/code/library/protocol pins changed or omitted")
    predictions, results = calculate(rows, x)
    arrays = dict(
        diagnostic_log_lengths=x,
        class_targets=np.array([v["class_target"] for v in rows], np.int64),
        **predictions,
    )
    require_products(root, r["artifacts"], c, rows, arrays)
    if results != r["results"]:
        raise ValueError("all replayed train/validation metrics and block intervals must match")
    y = arrays["class_targets"]
    train = np.array([v["split"] == "train" for v in rows])
    scalar_checks = 0
    for arm, posterior in predictions.items():
        for task in TASKS:
            lp, target = task_logp(posterior, task), task_mapping(task)[y]
            for split, indices in [("train", train), ("validation", ~train)]:
                compare_metrics(
                    results[arm][task][split], scalar_metrics(lp[indices], target[indices])
                )
                scalar_checks += 1
    for p, h in c["source_hashes"].items():
        if sha(p) != h:
            raise ValueError("source pin changed during replay")
    required.update({str(root / name): h for name, h in r["artifacts"].items()})
    required[str(root / "result.json")] = report_sha
    recheck_bindings(required)
    return dict(
        status="PASS_FRESH_LENGTH_ARRAY_REPLAY_AND_SCALAR_METRICS_ONLY",
        main_source_commit=r["source_commit"],
        config_sha256=r["config_sha256"],
        unique_source_groups=len(rows),
        split_counts=r["split_counts"],
        group_order_sha256=r["group_order_sha256"],
        split_assignment_sha256=r["split_assignment_sha256"],
        arrays_exact=len(arrays),
        all_log_probabilities_byte_equal=True,
        scalar_metric_checks=scalar_checks,
        scalar_absolute_tolerance=1e-12,
        results=results,
        validated_hashes=required,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_corpus_admission="NOT_ADMITTED",
        native_modality_quality_measured=False,
        new_payloads_opened=False,
        pretrained_models_loaded=False,
        gpu_used=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        cloud_cost=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new separate replay file required")
    tick = time.perf_counter()
    try:
        r = audit(args.main)
        seconds, peak = time.perf_counter() - tick, peak_bytes()
        if seconds > 60 or peak > 2 * 2**30:
            raise RuntimeError("fresh CPU replay resource bound exhausted")
        r.update(replay_seconds=seconds, replay_peak_cpu_bytes=peak)
    except Exception as exc:
        r = dict(
            status="FAIL_INCOMPLETE_LENGTH_REPLAY",
            error=repr(exc),
            replay_seconds=time.perf_counter() - tick,
        )
    write(args.output, r)
    print(r["status"], "seconds", r["replay_seconds"], flush=True)
    if r["status"] != "PASS_FRESH_LENGTH_ARRAY_REPLAY_AND_SCALAR_METRICS_ONLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
