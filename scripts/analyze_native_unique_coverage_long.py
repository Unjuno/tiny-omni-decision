"""Audit the 32,768-update unique-coverage arm against its pinned repeat control."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_native_temporal_fit import bootstrap, per_example
from scripts.native_fit import summarize

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
CONTROL = ROOT / "native-long-budget-fit-01"
FRESH = ROOT / "native-unique-coverage-long-fresh-fit-01"
REPLAY = ROOT / "native-unique-coverage-long-fresh-replay-01"
OUT = ROOT / "native-unique-coverage-long-comparison-04"
VALIDATION = ROOT / "native-common-corpus-02/validation.jsonl"
TRAIN = ROOT / "native-common-corpus-02/train.jsonl"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_artifacts(folder, receipt):
    for name, digest in receipt["artifacts"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"run artifact hash mismatch: {folder / name}")


def rename_bootstrap_delta(values):
    return json.loads(json.dumps(values).replace("lstm_minus_baseline", "fresh_minus_repeat"))


def summarize_deltas(control_metrics, fresh_metrics):
    metric_names = ("accuracy", "nll", "brier", "ece")
    return {
        "macro": {
            metric: fresh_metrics["macro"][metric] - control_metrics["macro"][metric]
            for metric in metric_names
        },
        "modalities": {
            name: {
                metric: fresh_metrics["modalities"][name][metric]
                - control_metrics["modalities"][name][metric]
                for metric in metric_names
            }
            for name in control_metrics["modalities"]
        },
        "sources": {
            name: {
                metric: fresh_metrics["sources"][name][metric]
                - control_metrics["sources"][name][metric]
                for metric in metric_names
            }
            for name in control_metrics["sources"]
        },
        "tasks": {
            name: {
                metric: fresh_metrics["tasks"][name][metric]
                - control_metrics["tasks"][name][metric]
                for metric in metric_names
            }
            for name in control_metrics["tasks"]
        },
    }


def unique_row_summary(rows, ids):
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
        raise ValueError("unique TRAIN question IDs required")
    selected = [by_id[row_id] for row_id in ids]

    def counts(key):
        values = {}
        for row in selected:
            value = row[key] if key != "source_task" else f"{row['source']}:{row['task']}"
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    return {
        "total": len(selected),
        "by_modality": counts("modality"),
        "by_source": counts("source"),
        "by_source_task": counts("source_task"),
    }


def run_comparison():
    rows = [json.loads(line) for line in VALIDATION.read_text(encoding="utf-8").splitlines()]
    control = load(CONTROL / "result.json")
    fresh = load(FRESH / "result.json")
    replay = load(REPLAY / "result.json")
    if not (
        control.get("status") == fresh.get("status") == "PASS_NATIVE_LONG_BUDGET_TRAIN"
        and replay.get("status") == "PASS_NATIVE_LONG_BUDGET_REPLAY"
        and control.get("step") == fresh.get("step") == replay.get("step") == 32768
    ):
        raise ValueError("complete matched 32,768-update control, candidate and replay required")
    verify_artifacts(CONTROL, control)
    verify_artifacts(FRESH, fresh)
    verify_artifacts(REPLAY, replay)
    if replay.get("reference_sha256") != sha(FRESH / "result.json"):
        raise ValueError("fresh-process replay does not reference this completed fit")

    validation_ids = [
        {"id": row["id"], "option_ids": row["option_ids"], "target_index": row["target_index"]}
        for row in rows
    ]
    if load(CONTROL / "validation-ids.json") != validation_ids:
        raise ValueError("repeat control validation rows/order differ from the frozen corpus")
    if load(FRESH / "validation-ids.json") != validation_ids:
        raise ValueError("fresh arm validation rows/order differ from the frozen corpus")
    if control["artifacts"]["initial.pt"] != fresh["artifacts"]["initial.pt"]:
        raise ValueError("initial readout differs between matched arms")
    if load(CONTROL / "lr-plan.json") != load(FRESH / "lr-plan.json"):
        raise ValueError("learning-rate schedule differs between matched arms")
    for key in ("modalities", "sources", "tasks"):
        if control["actual_accounting"][key] != fresh["actual_accounting"][key]:
            raise ValueError(f"matched full-horizon {key} occurrence quotas differ")

    train_rows = [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines()]
    control_unique = unique_row_summary(train_rows, load(CONTROL / "train-probe-ids.json"))
    fresh_unique = unique_row_summary(train_rows, load(FRESH / "train-probe-ids.json"))
    if control_unique["total"] != 6656 or fresh_unique["total"] != 12800:
        raise ValueError("fixed repeat/fresh unique TRAIN coverage counts changed")
    control_ids = set(load(CONTROL / "train-probe-ids.json"))
    fresh_ids = set(load(FRESH / "train-probe-ids.json"))
    additional_ids = fresh_ids - control_ids
    if len(control_ids & fresh_ids) != 6656 or len(additional_ids) != 6144:
        raise ValueError("fresh arm must retain the repeat parent and add 6,144 unique rows")
    added_modalities = unique_row_summary(train_rows, sorted(additional_ids))["by_modality"]
    if added_modalities != {"audio": 2048, "image": 2048, "text": 2048}:
        raise ValueError("registered added unique rows must be 2,048 each for Text/Image/Audio")

    selected_control = control["best_step"]
    selected_fresh = fresh["best_step"]
    if selected_control != selected_fresh:
        raise ValueError("the preregistered selected-step pair is not step matched")
    comparisons = {}
    bootstrap_results = {}
    for label, control_step, fresh_step in (
        ("selected", selected_control, selected_fresh),
        ("fixed_endpoint", 32768, 32768),
    ):
        control_scores = np.load(CONTROL / f"validation-{control_step:04}.npy", allow_pickle=False)
        fresh_scores = np.load(FRESH / f"validation-{fresh_step:04}.npy", allow_pickle=False)
        control_metrics = summarize(rows, control_scores)
        fresh_metrics = summarize(rows, fresh_scores)
        expected_control = control["selected_metrics" if label == "selected" else "final_metrics"]
        expected_fresh = fresh["selected_metrics" if label == "selected" else "final_metrics"]
        if control_metrics != expected_control or fresh_metrics != expected_fresh:
            raise ValueError(f"saved validation metrics do not reproduce for {label}")
        deltas = summarize_deltas(control_metrics, fresh_metrics)
        comparisons[label] = {
            "control_step": control_step,
            "fresh_step": fresh_step,
            "control": control_metrics,
            "fresh": fresh_metrics,
            "macro_delta_fresh_minus_repeat": deltas["macro"],
            "modality_delta_fresh_minus_repeat": deltas["modalities"],
            "source_delta_fresh_minus_repeat": deltas["sources"],
            "task_delta_fresh_minus_repeat": deltas["tasks"],
        }
        bootstrap_results[label] = rename_bootstrap_delta(
            bootstrap(
                rows,
                per_example(rows, control_scores),
                per_example(rows, fresh_scores),
            )
        )

    curve = load(FRESH / "learning-curve.json")
    control_curve = load(CONTROL / "learning-curve.json")
    resources = {
        "repeat_control": {
            key: control[key]
            for key in (
                "seconds",
                "training_seconds",
                "evaluation_seconds",
                "peak_vram_bytes",
                "peak_vram_reserved_bytes",
                "peak_working_set_bytes",
                "started_utc",
                "finished_utc",
            )
        },
        "fresh_arm": {
            key: fresh[key]
            for key in (
                "seconds",
                "training_seconds",
                "evaluation_seconds",
                "peak_vram_bytes",
                "peak_vram_reserved_bytes",
                "peak_working_set_bytes",
                "started_utc",
                "finished_utc",
            )
        },
        "fresh_replay": {
            key: replay[key]
            for key in (
                "seconds",
                "training_seconds",
                "evaluation_seconds",
                "peak_vram_bytes",
                "peak_vram_reserved_bytes",
                "peak_working_set_bytes",
            )
        },
    }
    report = {
        "status": "PASS_NATIVE_UNIQUE_COVERAGE_LONG_COMPARISON",
        "analysis_sha256": sha(__file__),
        "blind_evaluation": False,
        "sealed_audit_loaded": False,
        "product_promoted": False,
        "validation": {
            "path": str(VALIDATION),
            "count": len(rows),
            "ids_sha256": sha(FRESH / "validation-ids.json"),
            "same_ids_and_order": True,
        },
        "runs": {
            "repeat_control": {
                "path": str(CONTROL),
                "result_sha256": sha(CONTROL / "result.json"),
                "source_commit": control["source_commit"],
                "step": control["step"],
                "best_step": control["best_step"],
                "initial_sha256": control["artifacts"]["initial.pt"],
                "config_sha256": control["config_sha256"],
                "actual_accounting": control["actual_accounting"],
                "selected_accounting": control["selected_accounting"],
            },
            "fresh_arm": {
                "path": str(FRESH),
                "result_sha256": sha(FRESH / "result.json"),
                "source_commit": fresh["source_commit"],
                "step": fresh["step"],
                "best_step": fresh["best_step"],
                "initial_sha256": fresh["artifacts"]["initial.pt"],
                "config_sha256": fresh["config_sha256"],
                "actual_accounting": fresh["actual_accounting"],
                "selected_accounting": fresh["selected_accounting"],
            },
            "fresh_replay": {
                "path": str(REPLAY),
                "result_sha256": sha(REPLAY / "result.json"),
                "reference_sha256": replay["reference_sha256"],
            },
        },
        "model": {
            "total_parameters": fresh["total_parameters"],
            "trainable_parameters": fresh["trainable_parameters"],
            "python": fresh["python"],
            "versions": fresh["versions"],
            "cuda": fresh["cuda"],
            "device": fresh["device"],
            "dtype": fresh["dtype"],
            "cloud_cost": fresh["cloud_cost"],
        },
        "unique_train_question_counts": {
            "repeat_control": control_unique,
            "fresh_arm": fresh_unique,
            "added_fresh_rows": {
                "total": len(additional_ids),
                "by_modality": added_modalities,
            },
        },
        "resources": resources,
        "learning_curve": curve,
        "repeat_control_learning_curve": control_curve,
        "paired_comparison": comparisons,
        "video_scene_bootstrap": {
            "method": "paired underlying scene clusters; percentile 95%; NumPy linear quantile",
            "seed": 17,
            "resamples": 5000,
            "results": bootstrap_results,
            "selection_caveat": "The selected step was chosen on this same validation set.",
        },
        "interpretation": {
            "selected_checkpoint": (
                "Fresh coverage did not improve the macro-NLL-selected checkpoint over the "
                "repeat control; both selected step 8,192."
            ),
            "fixed_endpoint": (
                "The fresh arm improves endpoint macro Accuracy and NLL versus the repeat "
                "endpoint, but both are substantially worse calibrated than their selected "
                "checkpoints. This is an overfit endpoint comparison, not evidence to select "
                "the final checkpoint."
            ),
            "registered_hypothesis": (
                "Mixed: endpoint gains favor the fresh coverage arm, while the fixed selector "
                "does not select a better checkpoint. One seed and repeatedly observed "
                "development validation cannot isolate a general coverage effect."
            ),
        },
        "limitations": [
            "Single seed 17; no seed variance is estimated.",
            "Repeatedly observed development validation; not a blind or sealed evaluation.",
            (
                "The two arms share source rows, initialization, schedule, occurrence quotas "
                "and validation, but differ in training question coverage by design."
            ),
            (
                "Video TRAIN contains 512 unique question rows grouped into 256 scenes "
                "in both arms; no new video question or scene was added."
            ),
            (
                "Rolling training metrics are not a fixed generalization-gap estimate; "
                "endpoint train probes are descriptive and differ in unique-row count."
            ),
            (
                "No product promotion, new seed, budget extension, or sealed-audit "
                "evaluation was performed."
            ),
        ],
    }
    return report


def main():
    report = run_comparison()
    OUT.mkdir(exist_ok=False)
    (OUT / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(OUT / "comparison.json"),
                "selected_macro_delta": report["paired_comparison"]["selected"][
                    "macro_delta_fresh_minus_repeat"
                ],
                "endpoint_macro_delta": report["paired_comparison"]["fixed_endpoint"][
                    "macro_delta_fresh_minus_repeat"
                ],
                "video_scene_bootstrap": report["video_scene_bootstrap"]["results"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
