"""Full-source directory-only diagnostic; no native media/encoder quality claim."""

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.fetch_robomnist_input_smoke import peak_bytes
from scripts.replay_robomnist_directories import verify_bank
from scripts.robomnist_length_probe import (
    centroid_logp,
    length_features,
    metrics,
    paired_block_interval,
    parse_group,
    split_blocks,
    task_logp,
    task_mapping,
)

PARENT = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-all-directories-replay-02.json")
PARENT_SHA = "b34bc54f57d3c434947d673ed32cd9a9c6e4f82bfc10f9b02213b397a76a29a4"
PROTOCOL = Path("docs/pretrained_reuse/R2_ROBOMNIST_LENGTH_PROTOCOL_20261015.md")
PLAN = Path("docs/superpowers/plans/2026-10-15-robomnist-source-length-diagnostic.md")
ARMS = {
    "camera_length": [0, 1, 2],
    "microphone_length": [3, 4, 5],
    "csi_length": [6],
    "all_lengths": list(range(7)),
}
TASKS = ["activity60", "robot", "velocity", "trajectory"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_parent(parent):
    if (
        parent["status"] != "PASS_COMPLETE_DIRECTORY_GROUP_ACCOUNTING_ONLY"
        or parent["archives"] != 60
        or parent["unique_repetition_groups"] != 2135
        or any(
            parent[k] is not False
            for k in ["media_members_opened", "training", "sealed_audit_opened"]
        )
    ):
        raise ValueError("complete directory-only60-class/2135-group source accounting required")


def load_rows():
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("immutable complete directory replay required")
    parent = json.loads(PARENT.read_bytes())
    require_parent(parent)
    for p, h in parent["validated_hashes"].items():
        if sha(p) != h:
            raise ValueError("parent source/hash changed")
    rows, features, seen, class_rp = [], [], set(), set()
    banks = sorted(
        p
        for p in parent["validated_hashes"]
        if Path(p).name.startswith("directory-") and Path(p).suffix == ".json"
    )
    if len(banks) != 60:
        raise ValueError("all60 actual directory banks required")
    for ordinal, p in enumerate(banks):
        bank = json.loads(Path(p).read_bytes())
        checked = verify_bank(bank)
        if checked["class_index"] != ordinal:
            raise ValueError("source class ordering mismatch")
        sizes = {i["name"]: i["size"] for i in bank["members"]}
        for g in bank["groups"]:
            label, rp = parse_group(g["group_id"])
            if label != ordinal or g["group_id"] in seen or (label, rp) in class_rp:
                raise ValueError("duplicate or substituted class/repetition identity")
            seen.add(g["group_id"])
            class_rp.add((label, rp))
            rows.append(dict(group_id=g["group_id"], class_target=label, repetition=rp))
            features.append(length_features(g, sizes))
    if len(rows) != 2135 or set(r["class_target"] for r in rows) != set(range(60)):
        raise ValueError("complete2135-group/60-class source cohort required")
    split = split_blocks(r["repetition"] for r in rows)
    for row in rows:
        row["split"] = split[row["repetition"]]
    return rows, np.array(features), parent


def calculate(rows, x):
    y = np.array([r["class_target"] for r in rows], dtype=np.int64)
    rp = np.array([r["repetition"] for r in rows], dtype=np.int64)
    train = np.array([r["split"] == "train" for r in rows])
    if set(y[train]) != set(range(60)) or set(y[~train]) != set(range(60)):
        raise ValueError("every class required in both development splits")
    predictions = {
        arm: centroid_logp(x[train][:, cols], y[train], x[:, cols], 60)
        for arm, cols in ARMS.items()
    }
    counts = np.bincount(y[train], minlength=60)
    predictions["train_frequency_prior"] = np.broadcast_to(
        np.log(counts / counts.sum()), (len(y), 60)
    ).copy()
    results = {}
    for arm, logp in predictions.items():
        results[arm] = {}
        for task in TASKS:
            p = task_logp(logp, task)
            targets = task_mapping(task)[y]
            control = task_logp(predictions["train_frequency_prior"], task)
            results[arm][task] = dict(
                train=metrics(p[train], targets[train]),
                validation=metrics(p[~train], targets[~train]),
                paired_validation_vs_prior=paired_block_interval(
                    p[~train].argmax(1) == targets[~train],
                    control[~train].argmax(1) == targets[~train],
                    rp[~train],
                ),
            )
    return predictions, results


def write(path, value):
    blob = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf8")
    expected = hashlib.sha256(blob).hexdigest()
    with path.open("xb") as f:
        f.write(blob)
    return expected


def write_arrays(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    blob = buffer.getvalue()
    expected = hashlib.sha256(blob).hexdigest()
    with path.open("xb") as f:
        f.write(blob)
    return expected


def require_products(output, hashes, config, rows, arrays):
    if any(sha(output / name) != expected for name, expected in hashes.items()):
        raise ValueError("saved artifact hash changed since creation")
    if (
        json.loads((output / "config.json").read_bytes()) != config
        or json.loads((output / "rows.json").read_bytes()) != rows
    ):
        raise ValueError("saved effective config or rows differ from calculated values")
    with np.load(output / "arrays.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("every saved diagnostic array required")
        for key, value in arrays.items():
            a = saved[key]
            if a.shape != value.shape or a.dtype != value.dtype or a.tobytes() != value.tobytes():
                raise ValueError("saved diagnostic array shape/dtype/bytes differ")


def run_body(output, tick):
    if shutil.disk_usage(output).free < 2**30 + 16 * 2**20:
        raise RuntimeError("separate metadata diagnostic1GiB reserve unavailable")
    rows, x, parent = load_rows()
    source = {**parent["validated_hashes"], str(PARENT): PARENT_SHA}
    for p in [
        Path(__file__),
        Path("scripts/robomnist_length_probe.py"),
        Path("scripts/fetch_robomnist_input_smoke.py"),
        Path("scripts/replay_robomnist_directories.py"),
        Path("tests/test_robomnist_length_probe.py"),
        PROTOCOL,
        PLAN,
        Path(np.__file__),
        Path("src/tiny_omni_decision/decision_math.py"),
    ]:
        source[str(p.resolve())] = sha(p)
    config = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(),
        numpy=np.__version__,
        backend="CPU NumPy float64",
        platform=platform.platform(),
        execution_arguments=sys.argv,
        seed=17,
        source_hashes=source,
        feature_order=[
            "camera1_length",
            "camera2_length",
            "camera3_length",
            "mic1_length",
            "mic2_length",
            "mic3_length",
            "csi_length",
        ],
        arms=ARMS,
        tasks=TASKS,
        estimator="train-standardized centroids; -0.5 distance squared; temperature1; no tuning",
        split_rule="Rp01 train;11smallest SHA256 global-Rp blocks validation; all2135 retained",
        nll="stable log posterior, no clipping",
        ece_bins=15,
        bootstrap_draws=2000,
        bootstrap_seed=17,
        max_seconds=120,
        max_cpu_bytes=2 * 2**30,
        max_output_bytes=16 * 2**20,
        disk_reserve_bytes=2**30,
    )
    products = {
        "config.json": write(output / "config.json", config),
        "rows.json": write(output / "rows.json", rows),
    }
    predictions, results = calculate(rows, x)
    arrays = dict(
        diagnostic_log_lengths=x,
        class_targets=np.array([r["class_target"] for r in rows], np.int64),
        **predictions,
    )
    products["arrays.npz"] = write_arrays(output / "arrays.npz", arrays)
    require_products(output, products, config, rows, arrays)
    for p, h in source.items():
        if sha(p) != h:
            raise ValueError("source changed during diagnostic")
    seconds, peak = time.perf_counter() - tick, peak_bytes()
    size = sum(p.stat().st_size for p in output.iterdir())
    if (
        seconds > 120
        or peak > 2 * 2**30
        or size > 15 * 2**20
        or shutil.disk_usage(output).free < 2**30
    ):
        raise RuntimeError("metadata diagnostic resource bound exhausted")
    report = dict(
        status="PASS_FIXED_DIRECTORY_LENGTH_SHORTCUT_DIAGNOSTIC_ONLY",
        source_commit=config["source_commit"],
        config_sha256=products["config.json"],
        source_pins_unchanged=True,
        unique_source_groups=len(rows),
        split_counts=dict(Counter(r["split"] for r in rows)),
        split_repetition_blocks={
            s: sorted({r["repetition"] for r in rows if r["split"] == s})
            for s in ["train", "validation"]
        },
        class_counts={
            s: dict(sorted(Counter(r["class_target"] for r in rows if r["split"] == s).items()))
            for s in ["train", "validation"]
        },
        group_order_sha256=hashlib.sha256(
            "\n".join(r["group_id"] for r in rows).encode()
        ).hexdigest(),
        split_assignment_sha256=hashlib.sha256(
            "\n".join(f"{r['group_id']}:{r['split']}" for r in rows).encode()
        ).hexdigest(),
        results=results,
        elapsed_seconds=seconds,
        peak_cpu_bytes=peak,
        output_bytes_before_terminal=size,
        artifacts=products,
        pretrained_models_loaded=False,
        metadata_estimator_fit=True,
        teacher_training=False,
        gpu_used=False,
        new_payloads_opened=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        true_session_independence="UNVERIFIED",
        common_clock="UNVERIFIED",
        native_modality_quality_measured=False,
        training_corpus_admission="NOT_ADMITTED",
        cloud_cost=0,
    )
    require_products(output, products, config, rows, arrays)
    return report


def execute(output):
    if output.exists():
        raise ValueError("new diagnostic output directory required; existing outputs protected")
    output.mkdir()
    tick = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    try:
        report = run_body(output, tick)
        final_seconds, final_peak = time.perf_counter() - tick, peak_bytes()
        final_size = sum(p.stat().st_size for p in output.iterdir())
        if (
            final_seconds > 120
            or final_peak > 2 * 2**30
            or final_size > 15 * 2**20
            or shutil.disk_usage(output).free < 2**30
        ):
            raise RuntimeError("terminal metadata diagnostic resource bound exhausted")
        report.update(
            elapsed_seconds=final_seconds,
            peak_cpu_bytes=final_peak,
            output_bytes_before_terminal=final_size,
        )
    except Exception as exc:
        report = dict(
            status="FAIL_INCOMPLETE_LENGTH_DIAGNOSTIC",
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - tick,
            peak_cpu_bytes=peak_bytes(),
            native_modality_quality_measured=False,
            training_corpus_admission="NOT_ADMITTED",
        )
    report.update(pid=os.getpid(), started_utc=started, ended_utc=datetime.now(UTC).isoformat())
    write(output / "result.json", report)
    print(
        report["status"],
        report.get("split_counts", {}),
        "seconds",
        report["elapsed_seconds"],
        flush=True,
    )
    return report["status"] == "PASS_FIXED_DIRECTORY_LENGTH_SHORTCUT_DIAGNOSTIC_ONLY"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not execute(args.output):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
