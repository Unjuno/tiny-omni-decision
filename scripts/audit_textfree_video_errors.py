"""Read-only task-level attribution for saved text-free CLEVRER predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_RUN_REPORT_SHA256 = "ebd35fb94cd655c982773e370200f987055a97a312536e9b3e0416a19d673365"
ARMS = ("mean_mlp", "bilstm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    return rows


def validate_rows(rows: list[dict[str, Any]], *, expected_count: int | None = None) -> None:
    ids: set[str] = set()
    for row in rows:
        required = {
            "id",
            "scene_group_id",
            "question_type",
            "taxonomy",
            "target",
            "prediction",
            "options",
        }
        if not required.issubset(row):
            raise ValueError("prediction row is missing required attribution fields")
        if row["id"] in ids:
            raise ValueError("duplicate prediction ID")
        ids.add(row["id"])
        if not isinstance(row["options"], list) or len(row["options"]) < 2:
            raise ValueError("at least two supplied options are required")
        option_ids = [option.get("option_id") for option in row["options"]]
        probabilities = [option.get("probability") for option in row["options"]]
        if (
            len(set(option_ids)) != len(option_ids)
            or row["target"] not in option_ids
            or row["prediction"] not in option_ids
            or any(
                not isinstance(p, (float, int)) or not math.isfinite(p) or p < 0
                for p in probabilities
            )
            or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5)
        ):
            raise ValueError("invalid target, prediction, or option probabilities")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError("prediction count differs from frozen run")


def _cell_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_counts = Counter(row["target"] for row in rows)
    predicted_counts = Counter(row["prediction"] for row in rows)
    correct_counts = Counter(
        row["target"] for row in rows if row["target"] == row["prediction"]
    )
    recalls = {label: correct_counts[label] / count for label, count in target_counts.items()}
    n = len(rows)
    correct = sum(row["target"] == row["prediction"] for row in rows)
    nll = 0.0
    brier = 0.0
    for row in rows:
        target_probability = next(
            option["probability"]
            for option in row["options"]
            if option["option_id"] == row["target"]
        )
        nll -= math.log(max(float(target_probability), 1e-15))
        brier += sum(
            (float(option["probability"]) - float(option["option_id"] == row["target"])) ** 2
            for option in row["options"]
        )
    return {
        "examples": n,
        "supported_target_classes": len(target_counts),
        "accuracy": correct / n,
        "nll": nll / n,
        "brier": brier / n,
        "balanced_accuracy_supported_classes": sum(recalls.values()) / len(recalls),
        "empirical_majority_class_accuracy": max(target_counts.values()) / n,
        "target_counts": dict(sorted(target_counts.items())),
        "predicted_counts": dict(sorted(predicted_counts.items())),
        "per_class_recall": dict(sorted(recalls.items())),
        "errors_by_target_to_prediction": dict(
            sorted(
                Counter(
                    f"{row['target']}->{row['prediction']}"
                    for row in rows
                    if row["target"] != row["prediction"]
                ).items()
            )
        ),
    }


def analyze_program_signatures(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure exact typed-program reuse without exposing labels to the model."""
    result = {}
    question_types = sorted(
        {row["question_type"] for row in train_rows + validation_rows}
    )
    for question_type in question_types:
        train = [row for row in train_rows if row["question_type"] == question_type]
        validation = [row for row in validation_rows if row["question_type"] == question_type]
        train_signatures = Counter(tuple(row["program"]) for row in train)
        validation_signatures = Counter(tuple(row["program"]) for row in validation)
        unseen = {
            signature: count
            for signature, count in validation_signatures.items()
            if signature not in train_signatures
        }
        train_ops = {operator for row in train for operator in row["program"]}
        validation_ops = {operator for row in validation for operator in row["program"]}
        if not train or not validation:
            raise ValueError("each frozen question type needs train and validation rows")
        result[question_type] = {
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_unique_exact_programs": len(train_signatures),
            "validation_unique_exact_programs": len(validation_signatures),
            "validation_programs_absent_from_train": len(unseen),
            "validation_examples_with_program_absent_from_train": sum(unseen.values()),
            "validation_example_fraction_with_program_absent_from_train": (
                sum(unseen.values()) / len(validation)
            ),
            "validation_singleton_program_examples": sum(
                count == 1 for count in validation_signatures.values()
            ),
            "training_only_program_operators": sorted(train_ops - validation_ops),
            "validation_only_program_operators": sorted(validation_ops - train_ops),
            "shared_program_operator_count": len(train_ops & validation_ops),
        }
    return result


def attribute_predictions(
    predictions: dict[str, list[dict[str, Any]]], *, expected_count: int | None = None
) -> dict[str, Any]:
    if set(predictions) != set(ARMS):
        raise ValueError("exactly the frozen mean_mlp and bilstm arms are required")
    for rows in predictions.values():
        validate_rows(rows, expected_count=expected_count)
    keyed = {arm: {row["id"]: row for row in rows} for arm, rows in predictions.items()}
    if set(keyed["mean_mlp"]) != set(keyed["bilstm"]):
        raise ValueError("the paired arms have different prediction IDs")
    for sample_id in keyed["mean_mlp"]:
        a, b = keyed["mean_mlp"][sample_id], keyed["bilstm"][sample_id]
        if (
            a["scene_group_id"] != b["scene_group_id"]
            or a["taxonomy"] != b["taxonomy"]
            or a["question_type"] != b["question_type"]
            or a["target"] != b["target"]
        ):
            raise ValueError("paired prediction metadata or target differs")

    cells: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample_id, row in keyed["mean_mlp"].items():
        grouped[(row["taxonomy"], row["question_type"])].append(sample_id)
    for (taxonomy, question_type), ids in sorted(grouped.items()):
        group_key = f"{taxonomy}/{question_type}"
        per_arm = {
            arm: _cell_metrics([keyed[arm][sample_id] for sample_id in ids]) for arm in ARMS
        }
        delta = [
            int(keyed["bilstm"][sample_id]["prediction"] == keyed["bilstm"][sample_id]["target"])
            - int(
                keyed["mean_mlp"][sample_id]["prediction"]
                == keyed["mean_mlp"][sample_id]["target"]
            )
            for sample_id in ids
        ]
        cells[group_key] = {
            "taxonomy": taxonomy,
            "question_type": question_type,
            "mean_mlp": per_arm["mean_mlp"],
            "bilstm": per_arm["bilstm"],
            "paired_accuracy_outcomes": {
                "bilstm_win": sum(value == 1 for value in delta),
                "bilstm_loss": sum(value == -1 for value in delta),
                "tie": sum(value == 0 for value in delta),
                "net_correct_examples": sum(delta),
            },
        }
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in keyed["mean_mlp"].values():
        scene_rows[row["scene_group_id"]].append(row)
    scene_accuracy = {
        arm: {
            scene_id: sum(
                keyed[arm][row["id"]]["prediction"] == keyed[arm][row["id"]]["target"]
                for row in rows
            )
            / len(rows)
            for scene_id, rows in scene_rows.items()
        }
        for arm in ARMS
    }
    return {
        "analysis_scope": (
            "saved selected-checkpoint predictions only; no model inference or tuning"
        ),
        "paired_examples": len(keyed["mean_mlp"]),
        "unique_scenes": len(scene_rows),
        "question_taxonomy_cells": cells,
        "scene_accuracy_summary": {
            arm: {
                "zero_accuracy_scenes": sum(value == 0 for value in values.values()),
                "perfect_accuracy_scenes": sum(value == 1 for value in values.values()),
                "median_scene_accuracy": statistics.median(values.values()),
            }
            for arm, values in scene_accuracy.items()
        },
    }


def build_report(run_dir: Path, expected_report_sha256: str) -> dict[str, Any]:
    run_report_path = run_dir / "run-report.json"
    if sha256(run_report_path) != expected_report_sha256:
        raise ValueError("frozen run report hash mismatch")
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    if (
        run_report["status"] != "PASS_TEXTFREE_VIDEO_PAIRED_FIT"
        or run_report["sealed_audit_loaded"]
        or not run_report["validation_used_for_checkpoint_selection"]
    ):
        raise ValueError("expected completed development-only run with selector use")
    preflight_path = run_dir / "input-preflight.json"
    preflight_hash = sha256(preflight_path)
    if preflight_hash != run_report["input_preflight_sha256"]:
        raise ValueError("frozen input preflight hash mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    sample_dir = Path(preflight["sample_dir"])
    manifests = {}
    for split in ("train", "validation"):
        path = sample_dir / f"{split}.jsonl"
        expected = preflight[f"{split}_jsonl_sha256"]
        if sha256(path) != expected:
            raise ValueError(f"frozen {split} task manifest hash mismatch")
        manifests[split] = read_jsonl(path)
        if len(manifests[split]) != preflight[f"{split}_examples"]:
            raise ValueError(f"frozen {split} row count mismatch")
        if any(not isinstance(row.get("program"), list) for row in manifests[split]):
            raise ValueError(f"frozen {split} manifest lacks typed programs")
    predictions = {}
    input_hashes = {
        str(run_report_path): expected_report_sha256,
        str(preflight_path): preflight_hash,
        str(sample_dir / "train.jsonl"): preflight["train_jsonl_sha256"],
        str(sample_dir / "validation.jsonl"): preflight["validation_jsonl_sha256"],
    }
    for arm in ARMS:
        path = run_dir / f"selected-validation-{arm}.jsonl"
        expected = run_report["runs"][arm]["selected_prediction_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"selected prediction hash mismatch for {arm}")
        predictions[arm] = read_jsonl(path)
        input_hashes[str(path)] = actual
    analysis = attribute_predictions(predictions, expected_count=run_report["validation_examples"])
    validation_by_id = {row["id"]: row for row in manifests["validation"]}
    mean_by_id = {row["id"]: row for row in predictions["mean_mlp"]}
    if set(validation_by_id) != set(mean_by_id):
        raise ValueError("frozen validation manifest IDs differ from saved predictions")
    for sample_id, row in mean_by_id.items():
        source = validation_by_id[sample_id]
        if any(
            row[key] != source[source_key]
            for key, source_key in (
                ("scene_group_id", "scene_group_id"),
                ("taxonomy", "taxonomy"),
                ("question_type", "question_type"),
                ("target", "target"),
            )
        ) or [option["option_id"] for option in row["options"]] != source["options"]:
            raise ValueError("saved prediction task metadata differs from frozen validation row")
    analysis["typed_program_coverage"] = analyze_program_signatures(
        manifests["train"], manifests["validation"]
    )
    analysis.update(
        {
            "experiment_id": run_report["experiment_id"],
            "source_commit": run_report["source_commit"],
            "effective_config_sha256": run_report["effective_config_sha256"],
            "validation_used_for_checkpoint_selection": True,
            "validation_examples": run_report["validation_examples"],
            "validation_unique_scenes": run_report["validation_unique_scenes"],
            "prior_selection_uncertainty": (
                "descriptive attribution only; selected on this same development validation"
            ),
            "input_sha256": input_hashes,
            "analysis_script_sha256": sha256(Path(__file__)),
            "expectation_class_count_is_not_uniform": True,
        }
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-run-report-sha256", default=EXPECTED_RUN_REPORT_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.run_dir, args.expected_run_report_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"PASS_SAVED_PREDICTION_ATTRIBUTION {args.output}")


if __name__ == "__main__":
    main()
