"""Compare the native capacity control to the frozen baseline/LSTM receipts."""

# Resolve this worktree's local package before using the shared analysis helper.
# ruff: noqa: E402

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scripts.native_fit import summarize  # noqa: E402

ROOT = Path(r"C:\CodexArtifacts\pretrained-reuse-r2")
EXPECTED_OLD_RECEIPTS = {
    "baseline": "aaa1fe96a4be85a71ce146981c6b2e8b58e1308d91fc110e1ece6804b4d8048c",
    "lstm": "47dde479c91ed82cdbf260686f0184ebdf95b53981b28248e0ad1ebf2c6b10c6",
}
STEPS = (512, 1024, 1536, 1920, 2048)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_receipt(folder, expected_status, expected_sha=None):
    path = folder / "result.json"
    digest = sha(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"preserved {folder.name} receipt changed")
    receipt = load(path)
    if receipt["status"] != expected_status or receipt["step"] != 2048:
        raise ValueError(f"incomplete run receipt: {folder.name}")
    return receipt, digest


def verify_artifacts(folder, receipt):
    for filename, expected in receipt["artifacts"].items():
        if sha(folder / filename) != expected:
            raise ValueError(f"artifact hash mismatch: {folder.name}/{filename}")
    return len(receipt["artifacts"])


def frozen_data_pins(receipt):
    prefixes = tuple(
        str(ROOT / name).replace("/", "\\").lower() + "\\"
        for name in (
            "native-common-corpus-02",
            "native-common-queries-cache-01",
            "native-conditional-candidate-smoke-02",
        )
    )
    return {
        path: digest
        for path, digest in receipt["validated_hashes"].items()
        if path.replace("/", "\\").lower().startswith(prefixes)
    }


def metric_equal(left, right, tolerance=1e-8):
    for key in ("accuracy", "nll", "brier", "ece"):
        if not math.isclose(float(left[key]), float(right[key]), abs_tol=tolerance, rel_tol=0):
            return False
    return True


def cluster_bootstrap(rows, left, right, task_filter, *, repetitions=5000, seed=17):
    selected = [
        i for i, row in enumerate(rows)
        if row["modality"] == "video" and (task_filter is None or row["task"] == task_filter)
    ]
    groups = sorted({rows[i]["group"] for i in selected})
    if not groups:
        return {"questions": 0, "groups": 0, "status": "NOT_APPLICABLE"}
    indices = [[i for i in selected if rows[i]["group"] == group] for group in groups]
    counts = np.asarray([len(group) for group in indices], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    output = {
        "questions": len(selected),
        "groups": len(groups),
        "seed": seed,
        "repetitions": repetitions,
        "interval": "paired scene-cluster percentile 95%, NumPy linear quantile",
    }
    for metric in ("accuracy", "nll", "brier"):
        delta = right[metric] - left[metric]
        sums = np.asarray([delta[group].sum() for group in indices])
        sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        output[metric] = {
            "right_minus_left": float(delta[selected].mean()),
            "low": float(np.quantile(sampled, 0.025, method="linear")),
            "high": float(np.quantile(sampled, 0.975, method="linear")),
        }
    return output


def per_example(rows, scores):
    scores = np.asarray(scores)
    if scores.shape != (len(rows), 40) or not np.isfinite(scores).all():
        raise ValueError("saved validation score matrix shape/value mismatch")
    accuracy = np.empty(len(rows), dtype=np.float64)
    nll = np.empty(len(rows), dtype=np.float64)
    brier = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        count = len(row["options"])
        logits = scores[index, :count].astype(np.float64)
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        target = row["target_index"]
        accuracy[index] = probabilities.argmax() == target
        nll[index] = -math.log(max(float(probabilities[target]), np.finfo(float).tiny))
        brier[index] = float(np.square(probabilities).sum() - 2 * probabilities[target] + 1)
    return {"accuracy": accuracy, "nll": nll, "brier": brier}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capacity-run", type=Path,
        default=ROOT / "native-common-capacity-control-fit-01",
    )
    parser.add_argument(
        "--capacity-replay", type=Path,
        default=ROOT / "native-common-capacity-control-replay-01",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "native-common-capacity-control-comparison-01",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    folders = {
        "baseline": ROOT / "native-common-baseline-fit-01",
        "lstm": ROOT / "native-common-lstm-fit-01",
        "capacity_mlp": args.capacity_run,
    }
    receipts, receipt_hashes = {}, {}
    for name, folder in folders.items():
        expected_sha = EXPECTED_OLD_RECEIPTS.get(name)
        receipts[name], receipt_hashes[name] = checked_receipt(
            folder, "PASS_NATIVE_FIT_TRAIN", expected_sha
        )
    replay, replay_sha = checked_receipt(args.capacity_replay, "PASS_NATIVE_FIT_REPLAY")
    verified_control_artifacts = verify_artifacts(args.capacity_run, receipts["capacity_mlp"])
    verified_replay_artifacts = verify_artifacts(args.capacity_replay, replay)
    control = receipts["capacity_mlp"]
    if replay["reference_sha256"] != receipt_hashes["capacity_mlp"]:
        raise ValueError("capacity control replay does not match its training receipt")
    old = [receipts[name] for name in ("baseline", "lstm")]
    if any(
        frozen_data_pins(receipt) != frozen_data_pins(control)
        or receipt["actual_accounting"] != control["actual_accounting"]
        for receipt in old
    ):
        raise ValueError("source/sampler accounting changed across arms")
    if any(
        receipts[name]["artifacts"][key] != control["artifacts"][key]
        for name in ("baseline", "lstm")
        for key in ("sampling.json", "validation-ids.json", "train-probe-ids.json")
    ):
        raise ValueError("sample order, validation IDs, or probe IDs differ")

    rows = [
        json.loads(line)
        for line in (ROOT / "native-common-corpus-02" / "validation.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    expected_ids = load(folders["baseline"] / "validation-ids.json")
    if expected_ids != [
        {"id": row["id"], "option_ids": row["option_ids"], "target_index": row["target_index"]}
        for row in rows
    ]:
        raise ValueError("frozen validation ID order differs")

    arrays = {}
    metrics = {}
    curves = {}
    for name, folder in folders.items():
        receipt = receipts[name]
        for filename in ("validation-0000.npy", "validation-2048.npy"):
            if sha(folder / filename) != receipt["artifacts"][filename]:
                raise ValueError(f"saved prediction hash mismatch: {name}/{filename}")
        initial = np.load(folder / "validation-0000.npy", allow_pickle=False)
        final = np.load(folder / "validation-2048.npy", allow_pickle=False)
        arrays[name] = {"initial": per_example(rows, initial), "final": per_example(rows, final)}
        if name != "baseline" and not np.array_equal(initial, np.load(
            folders["baseline"] / "validation-0000.npy", allow_pickle=False
        )):
            raise ValueError(f"initial function differs: {name}")
        computed = summarize(rows, final)
        if not metric_equal(computed["macro"], receipt["final_metrics"]["macro"]):
            raise ValueError(f"saved final metrics differ from receipt: {name}")
        selected_scores = np.load(
            folder / f"validation-{receipt['best_step']:04}.npy", allow_pickle=False
        )
        if sha(folder / f"validation-{receipt['best_step']:04}.npy") != receipt["artifacts"][
            f"validation-{receipt['best_step']:04}.npy"
        ]:
            raise ValueError(f"selected prediction hash mismatch: {name}")
        selected_metrics = summarize(rows, selected_scores)
        if not metric_equal(selected_metrics["macro"], receipt["selected_metrics"]["macro"]):
            raise ValueError(f"selected metrics differ from receipt: {name}")
        metrics[name] = {
            "final": computed,
            "selected": selected_metrics,
            "selected_step": receipt["best_step"],
        }
        learning_curve = load(folder / "learning-curve.json")
        curves[name] = {
            step: next(item for item in learning_curve if item["step"] == step)
            for step in STEPS
        }

    comparisons = {}
    for phase in ("final",):
        for left_name, right_name in (("baseline", "capacity_mlp"), ("lstm", "capacity_mlp")):
            comparisons[f"{right_name}_minus_{left_name}_{phase}"] = {
                subset: cluster_bootstrap(
                    rows,
                    arrays[left_name][phase],
                    arrays[right_name][phase],
                    task_filter=task,
                )
                for subset, task in (
                    ("video_all", None),
                    ("temporal_descriptive", "temporal_descriptive"),
                )
            }

    args.output.mkdir(parents=True)
    report = {
        "status": "PASS_NATIVE_TEMPORAL_CAPACITY_COMPARISON",
        "scope": "same exposed development validation; not independent generalization evidence",
        "runs": {
            name: {
                "receipt_sha256": receipt_hashes[name],
                "config_sha256": receipt["config_sha256"],
                "trainable_parameters": receipt["trainable_parameters"],
                "total_parameters": receipt["total_parameters"],
                "best_step": receipt["best_step"],
                "selected_metrics": metrics[name]["selected"],
                "final_metrics": metrics[name]["final"],
                "actual_accounting": receipt["actual_accounting"],
                "training_seconds": receipt["training_seconds"],
                "evaluation_seconds": receipt["evaluation_seconds"],
                "wall_seconds": receipt["seconds"],
                "peak_vram_bytes": receipt["peak_vram_bytes"],
                "device": receipt["device"],
            }
            for name, receipt in receipts.items()
        },
        "learning_curve": {
            name: {
                str(step): {
                    "macro_accuracy": item["validation"]["macro"]["accuracy"],
                    "macro_nll": item["validation"]["macro"]["nll"],
                    "macro_brier": item["validation"]["macro"]["brier"],
                    "macro_ece": item["validation"]["macro"]["ece"],
                    "minimum_modality_accuracy": item["validation"]["minimum_modality_accuracy"],
                    "video_accuracy": item["validation"]["modality_macro"]["video"]["accuracy"],
                    "video_nll": item["validation"]["modality_macro"]["video"]["nll"],
                    "warnings": item["warnings"],
                }
                for step, item in steps.items()
            }
            for name, steps in curves.items()
        },
        "paired_video_uncertainty": comparisons,
        "replay": {
            "receipt_sha256": replay_sha,
            "reference_sha256": replay["reference_sha256"],
            "max_abs_prediction_delta": replay["replay_max_abs"],
            "exact_reload_verified": replay["replay_max_abs"] == 0,
            "verified_artifact_count": verified_replay_artifacts,
        },
        "capacity_control_verified_artifact_count": verified_control_artifacts,
        "limits": [
            "The development validation and selector were observed in the earlier "
            "baseline/LSTM comparison.",
            "Single seed only; scene bootstrap does not include training-seed variance.",
            "The matched control isolates tokenwise capacity from recurrence, not FLOPs "
            "or all architecture differences.",
            "Train probe rows are a selected subset, not a matched-distribution "
            "generalization gap.",
            "No product model is promoted; the 95-98% task, joint-input, OOD and "
            "deployment gates remain open.",
        ],
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
