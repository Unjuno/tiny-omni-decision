"""Compare two frozen IntPhys dev prediction receipts by paired scene bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.intphys_metrics import grouped_paired_bootstrap, threshold_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(
    evaluation_dir: Path, training_metadata_path: Path, corpus_report_path: Path
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    report_path = evaluation_dir / "evaluation-report.json"
    predictions_path = evaluation_dir / "dev-predictions.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = json.loads(training_metadata_path.read_text(encoding="utf-8"))
    corpus_report = json.loads(corpus_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_single_fixed_intphys_dev_diagnostic":
        raise ValueError(f"incomplete evaluation receipt: {report_path}")
    if report.get("sealed_test_accessed") is not False:
        raise ValueError("comparison inputs must explicitly confirm sealed test was not accessed")
    if report.get("dev_used_for_fit_or_calibration") is not False:
        raise ValueError("dev labels must not have been used for fit or calibration")
    if report.get("training_metadata_sha256") != sha256_file(training_metadata_path):
        raise ValueError("evaluation report does not bind to the supplied training metadata")
    if report.get("training_corpus_report_sha256") != sha256_file(corpus_report_path):
        raise ValueError("evaluation report does not bind to the supplied corpus report")
    if report.get("checkpoint_sha256") != metadata.get("best_checkpoint_sha256"):
        raise ValueError("evaluation report and training metadata identify different checkpoints")
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != report.get("dev_video_count"):
        raise ValueError("prediction row count does not match evaluation receipt")
    feature_shape = corpus_report.get("feature_shape", [])
    frame_count = corpus_report.get("frame_count", feature_shape[0] if feature_shape else None)
    if not isinstance(frame_count, int) or frame_count < 2:
        raise ValueError("corpus receipt lacks its sampled frame count")
    if metadata.get("sequence_length", frame_count) != frame_count:
        raise ValueError("training and corpus receipts disagree on sampled frame count")
    metadata.setdefault("sequence_length", frame_count)
    return report, metadata, corpus_report, rows


def compare_runs(
    *,
    reference_evaluation_dir: Path,
    candidate_evaluation_dir: Path,
    reference_training_metadata: Path,
    candidate_training_metadata: Path,
    reference_corpus_report: Path,
    candidate_corpus_report: Path,
    replicates: int = 2_000,
    seed: int = 17,
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("at least 100 paired bootstrap replicates are required")
    reference_report, reference_metadata, reference_corpus, reference_rows = _load_run(
        reference_evaluation_dir, reference_training_metadata, reference_corpus_report
    )
    candidate_report, candidate_metadata, candidate_corpus, candidate_rows = _load_run(
        candidate_evaluation_dir, candidate_training_metadata, candidate_corpus_report
    )
    for key in (
        "archive_sha256",
        "checkpoint_sha256",
        "model_revision",
        "videoprism_source_commit",
    ):
        if reference_corpus.get(key) != candidate_corpus.get(key):
            raise ValueError(f"the two runs use different pinned corpus or encoder: {key}")
    for key in (
        "seed",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "early_stopping_patience",
        "min_delta",
        "optimizer",
    ):
        if reference_metadata.get(key) != candidate_metadata.get(key):
            raise ValueError(f"the two runs differ on a non-frame training setting: {key}")
    reference_parameter_count = reference_metadata.get(
        "trainable_parameter_count", reference_metadata.get("parameter_count")
    )
    candidate_parameter_count = candidate_metadata.get(
        "trainable_parameter_count", candidate_metadata.get("parameter_count")
    )
    if reference_parameter_count != candidate_parameter_count:
        raise ValueError("the two runs have different predictor parameter counts")
    reference_by_id = {row["sample_id"]: row for row in reference_rows}
    candidate_by_id = {row["sample_id"]: row for row in candidate_rows}
    if (
        len(reference_by_id) != len(reference_rows)
        or len(candidate_by_id) != len(candidate_rows)
        or reference_by_id.keys() != candidate_by_id.keys()
    ):
        raise ValueError("candidate prediction sample IDs differ or contain duplicates")

    sample_ids = sorted(reference_by_id)
    labels: list[int] = []
    groups: list[str] = []
    blocks: list[str] = []
    reference_scores: list[float] = []
    candidate_scores: list[float] = []
    for sample_id in sample_ids:
        reference = reference_by_id[sample_id]
        candidate = candidate_by_id[sample_id]
        identity = ("label_impossible", "scene_group_id", "block")
        if any(reference[key] != candidate[key] for key in identity):
            raise ValueError(f"label, scene, or block mismatch for {sample_id}")
        labels.append(int(reference["label_impossible"]))
        groups.append(str(reference["scene_group_id"]))
        blocks.append(str(reference["block"]))
        reference_scores.append(float(reference["learned_lstm_anomaly_score"]))
        candidate_scores.append(float(candidate["learned_lstm_anomaly_score"]))

    reference_threshold = float(reference_report["learned_lstm"]["threshold"])
    candidate_threshold = float(candidate_report["learned_lstm"]["threshold"])
    reference_metrics = threshold_metrics(labels, reference_scores, reference_threshold)
    candidate_metrics = threshold_metrics(labels, candidate_scores, candidate_threshold)
    paired = grouped_paired_bootstrap(
        labels,
        candidate_scores,
        reference_scores,
        groups,
        first_threshold=candidate_threshold,
        second_threshold=reference_threshold,
        replicates=replicates,
        seed=seed,
    )
    by_block: dict[str, dict[str, Any]] = {}
    label_array = np.asarray(labels, dtype=np.int8)
    reference_array = np.asarray(reference_scores, dtype=np.float64)
    candidate_array = np.asarray(candidate_scores, dtype=np.float64)
    block_array = np.asarray(blocks, dtype=str)
    for block in sorted(set(blocks)):
        indices = np.flatnonzero(block_array == block)
        by_block[block] = {
            "reference": threshold_metrics(
                label_array[indices], reference_array[indices], reference_threshold
            ),
            "candidate": threshold_metrics(
                label_array[indices], candidate_array[indices], candidate_threshold
            ),
        }

    reference_prediction_path = reference_evaluation_dir / "dev-predictions.jsonl"
    candidate_prediction_path = candidate_evaluation_dir / "dev-predictions.jsonl"
    reference_report_path = reference_evaluation_dir / "evaluation-report.json"
    candidate_report_path = candidate_evaluation_dir / "evaluation-report.json"
    return {
        "status": "complete_paired_development_comparison",
        "not_blind": True,
        "interpretation": (
            "Repeated fixed IntPhys development diagnostic; not a sealed evaluation or "
            "product result."
        ),
        "reference_frame_count": reference_metadata["sequence_length"],
        "candidate_frame_count": candidate_metadata["sequence_length"],
        "reference_selected_epoch": reference_metadata["selected_epoch"],
        "candidate_selected_epoch": candidate_metadata["selected_epoch"],
        "n_videos": len(sample_ids),
        "scene_groups": len(set(groups)),
        "sample_id_set_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "reference": {
            "threshold": reference_threshold,
            "metrics": reference_metrics,
            "training_metadata_sha256": sha256_file(reference_training_metadata),
            "corpus_report_sha256": sha256_file(reference_corpus_report),
            "prediction_sha256": sha256_file(reference_prediction_path),
            "evaluation_report_sha256": sha256_file(reference_report_path),
        },
        "candidate": {
            "threshold": candidate_threshold,
            "metrics": candidate_metrics,
            "training_metadata_sha256": sha256_file(candidate_training_metadata),
            "corpus_report_sha256": sha256_file(candidate_corpus_report),
            "prediction_sha256": sha256_file(candidate_prediction_path),
            "evaluation_report_sha256": sha256_file(candidate_report_path),
        },
        "candidate_minus_reference": {
            "accuracy": candidate_metrics["accuracy"] - reference_metrics["accuracy"],
            "balanced_accuracy": (
                candidate_metrics["balanced_accuracy"] - reference_metrics["balanced_accuracy"]
            ),
            "roc_auc": candidate_metrics["roc_auc"] - reference_metrics["roc_auc"],
        },
        "paired_scene_group_bootstrap_candidate_minus_reference": paired,
        "bootstrap_note": (
            f"{replicates:,} paired percentile resamples of {len(set(groups))} scene groups "
            f"with replacement; seed {seed}. The development set was previously inspected."
        ),
        "by_block": by_block,
        "sealed_test_accessed": False,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-evaluation-dir", type=Path, required=True)
    parser.add_argument("--candidate-evaluation-dir", type=Path, required=True)
    parser.add_argument("--reference-training-metadata", type=Path, required=True)
    parser.add_argument("--candidate-training-metadata", type=Path, required=True)
    parser.add_argument("--reference-corpus-report", type=Path, required=True)
    parser.add_argument("--candidate-corpus-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite comparison receipt: {output}")
    result = compare_runs(
        reference_evaluation_dir=args.reference_evaluation_dir.resolve(strict=True),
        candidate_evaluation_dir=args.candidate_evaluation_dir.resolve(strict=True),
        reference_training_metadata=args.reference_training_metadata.resolve(strict=True),
        candidate_training_metadata=args.candidate_training_metadata.resolve(strict=True),
        reference_corpus_report=args.reference_corpus_report.resolve(strict=True),
        candidate_corpus_report=args.candidate_corpus_report.resolve(strict=True),
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
