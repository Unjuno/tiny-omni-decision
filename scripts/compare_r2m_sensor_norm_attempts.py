"""Compare the original and train-zscored R2-M validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tiny_omni_decision.r2m_attempt_comparison import (
    binary_metric_summary,
    paired_choice_rows,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def prediction_key(row: dict[str, object]) -> tuple[str, int, str]:
    return (str(row.get("pair_id", "")), int(row.get("target", -1)),
            str(row.get("sensor_sample_id", "")))


def validate_config_pair(
    baseline: dict[str, object], normalized: dict[str, object],
) -> None:
    fixed_fields = (
        "seed", "epochs", "batch_size", "learning_rate", "optimizer", "scheduler",
        "selection", "loss", "modality_dimensions", "query_codes", "fusion_width",
        "fusion_heads", "use_time", "use_gates", "trainable_parameters",
        "pair_index_sha256", "feature_cache_index_sha256", "feature_samples",
    )
    for key in fixed_fields:
        if baseline.get(key) != normalized.get(key):
            raise ValueError(f"attempts differ in fixed config field {key!r}")
    old_normalization = baseline.get("sensor_normalization") or {"method": "none"}
    if old_normalization.get("method") != "none":
        raise ValueError("baseline attempt must use no sensor normalization")
    normalization = normalized.get("sensor_normalization")
    if not isinstance(normalization, dict) or normalization.get("method") != (
        "per-axis z-score using every token from each unique train feature once"
    ):
        raise ValueError("second attempt does not declare the expected train-only transform")
    if normalization.get("sealed_audit_loaded") is not False:
        raise ValueError("normalization metadata does not establish audit isolation")


def compare_attempts(
    baseline_dir: Path,
    normalized_dir: Path,
    pair_index_path: Path,
) -> dict[str, object]:
    baseline_metrics_path = baseline_dir / "metrics.json"
    baseline_config_path = baseline_dir / "effective-config.json"
    baseline_predictions_path = baseline_dir / "validation-predictions.jsonl"
    normalized_metrics_path = normalized_dir / "metrics.json"
    normalized_config_path = normalized_dir / "effective-config.json"
    normalized_predictions_path = normalized_dir / "validation-predictions.jsonl"
    required = (
        baseline_metrics_path, baseline_config_path, baseline_predictions_path,
        normalized_metrics_path, normalized_config_path, normalized_predictions_path,
        pair_index_path,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("both attempts, the validation outputs and pair index are required")
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    normalized_metrics = json.loads(normalized_metrics_path.read_text(encoding="utf-8"))
    baseline_config = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    normalized_config = json.loads(normalized_config_path.read_text(encoding="utf-8"))
    validate_config_pair(baseline_config, normalized_config)
    if any(metrics.get("sealed_audit_accessed") is not False
           for metrics in (baseline_metrics, normalized_metrics)):
        raise ValueError("an attempt does not establish sealed-audit isolation")
    if any(metrics.get("status") != "COMPLETED"
           for metrics in (baseline_metrics, normalized_metrics)):
        raise ValueError("both fixed-budget runs must be complete")
    if (baseline_metrics.get("pair_index_sha256")
            != normalized_metrics.get("pair_index_sha256")
            or baseline_metrics.get("feature_cache_index_sha256")
            != normalized_metrics.get("feature_cache_index_sha256")):
        raise ValueError("attempts use different pair-index or feature-cache revisions")

    pair_index = json.loads(pair_index_path.read_text(encoding="utf-8"))
    if pair_index.get("manifest_sha256") != baseline_metrics.get("pair_index_sha256"):
        raise ValueError("provided pair index does not match recorded run hashes")
    if pair_index.get("sealed_audit_accessed") is not False:
        raise ValueError("pair index records sealed-audit access")
    validation_pairs = {
        str(pair["pair_id"]): pair for pair in pair_index["pairs"]
        if pair.get("split") == "validation"
    }
    expected_pair_ids = set(validation_pairs)

    baseline_rows = read_jsonl(baseline_predictions_path)
    normalized_rows = read_jsonl(normalized_predictions_path)
    baseline_by_key = {prediction_key(row): row for row in baseline_rows}
    normalized_by_key = {prediction_key(row): row for row in normalized_rows}
    if len(baseline_by_key) != len(baseline_rows) or len(normalized_by_key) != len(normalized_rows):
        raise ValueError("duplicate prediction rows detected")
    if baseline_by_key.keys() != normalized_by_key.keys():
        raise ValueError("attempt prediction IDs or candidate rows differ")
    if len(baseline_rows) != 2 * len(expected_pair_ids):
        raise ValueError("validation prediction count does not match frozen pair index")

    for key in baseline_by_key:
        old, new = baseline_by_key[key], normalized_by_key[key]
        pair_id = key[0]
        pair = validation_pairs.get(pair_id)
        if pair is None:
            raise ValueError("prediction references a non-validation pair")
        expected_sensor_id = str(
            pair["positive_sensor_sample_id"] if key[1] == 1
            else pair["negative_sensor_sample_id"]
        )
        expected_anchor = str(pair["anchor_sample_id"])
        if (str(old.get("sensor_sample_id")) != expected_sensor_id
                or str(new.get("sensor_sample_id")) != expected_sensor_id
                or str(old.get("anchor_sample_id")) != expected_anchor
                or str(new.get("anchor_sample_id")) != expected_anchor
                or old.get("negative_relation") != pair.get("negative_participant_relation")
                or new.get("negative_relation") != pair.get("negative_participant_relation")):
            raise ValueError("prediction identities do not match frozen validation pair index")

    results = {}
    for name, rows in (("attempt_01_raw_counts", baseline_rows),
                       ("attempt_02_train_axis_zscore", normalized_rows)):
        probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
        targets = np.asarray([row["target"] for row in rows], dtype=np.int64)
        candidate_metrics = binary_metric_summary(probabilities, targets)
        paired_rows = paired_choice_rows(rows)
        paired_probabilities = np.asarray(
            [row["probabilities"] for row in paired_rows], dtype=np.float64
        )
        paired_targets = np.ones(len(paired_rows), dtype=np.int64)
        pair_metrics = binary_metric_summary(paired_probabilities, paired_targets)
        pair_metrics["by_negative_relation"] = {}
        for relation in ("cross_participant", "within_participant"):
            indexes = [index for index, row in enumerate(paired_rows)
                       if row.get("negative_relation") == relation]
            if indexes:
                pair_metrics["by_negative_relation"][relation] = binary_metric_summary(
                    paired_probabilities[indexes], paired_targets[indexes]
                )
        pair_metrics["paired_questions"] = len(paired_rows)
        by_participant = {}
        def participant_for(pair_id: str) -> int:
            sample_id = str(validation_pairs[pair_id]["anchor_sample_id"])
            return int(sample_id.split("|participant=")[1].split("|")[0])

        for participant in sorted({
            participant_for(str(row["pair_id"])) for row in paired_rows
        }):
            selected = [index for index, row in enumerate(paired_rows)
                        if participant_for(str(row["pair_id"])) == participant]
            by_participant[str(participant)] = binary_metric_summary(
                paired_probabilities[selected], paired_targets[selected]
            )
        pair_metrics["by_anchor_participant"] = by_participant
        run_metrics = (
            baseline_metrics if name == "attempt_01_raw_counts" else normalized_metrics
        )
        results[name] = {
            "best_epoch": int(run_metrics["selected_checkpoint_epoch_by_calibration_nll"]),
            "candidate_level": candidate_metrics,
            "paired_choice": pair_metrics,
        }

    metric_names = ("accuracy", "nll", "brier", "ece_15bin", "mean_confidence")
    deltas = {}
    for level in ("candidate_level", "paired_choice"):
        deltas[level] = {
            metric: float(
                results["attempt_02_train_axis_zscore"][level][metric]
                - results["attempt_01_raw_counts"][level][metric]
            ) for metric in metric_names
        }
    participant_mapping = {
        pair_id: participant_for(pair_id) for pair_id in validation_pairs
    }
    output: dict[str, object] = {
        "schema": "r2m_sensor_normalization_attempt_comparison_v1",
        "pair_index_sha256": pair_index["manifest_sha256"],
        "baseline_metrics_sha256": sha256_file(baseline_metrics_path),
        "baseline_config_sha256": sha256_file(baseline_config_path),
        "baseline_predictions_sha256": sha256_file(baseline_predictions_path),
        "normalized_metrics_sha256": sha256_file(normalized_metrics_path),
        "normalized_config_sha256": sha256_file(normalized_config_path),
        "normalized_predictions_sha256": sha256_file(normalized_predictions_path),
        "pair_index_file_sha256": sha256_file(pair_index_path),
        "validation_pair_count": len(expected_pair_ids),
        "validation_prediction_count_per_attempt": len(baseline_rows),
        "anchor_participant_counts": {
            str(participant): sum(value == participant for value in participant_mapping.values())
            for participant in sorted(set(participant_mapping.values()))
        },
        "results": results,
        "attempt_02_minus_attempt_01": deltas,
        "uncertainty": (
            "No population interval computed: only two held-out participants; "
            "validation already exposed for development."
        ),
        "sealed_audit_loaded": False,
        "validation_used_for_checkpoint_selection": False,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing comparison artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison = compare_attempts(args.baseline_dir, args.normalized_dir,
                                  args.pair_index)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison["attempt_02_minus_attempt_01"], sort_keys=True))


if __name__ == "__main__":
    main()
