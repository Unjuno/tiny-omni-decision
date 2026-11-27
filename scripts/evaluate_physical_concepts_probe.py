"""Run the frozen one-time Physical Concepts anomaly-transfer evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"

EXPECTED_CATEGORIES = (
    "continuity",
    "solidity",
    "object_persistence",
    "unchangeableness",
    "directional_inertia",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_validation_threshold(scores: np.ndarray) -> tuple[float, float]:
    """Freeze q95 and robust scale using only positive freeform validation clips."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("validation scores must be finite one-dimensional data with n>=20")
    threshold = float(np.quantile(values, 0.95, method="linear"))
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if scale == 0:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("validation score distribution has no positive robust/std scale")
    return threshold, scale


def probe_receipts_match_frozen_protocol(
    extraction_report: dict[str, Any],
    selection_receipt: dict[str, Any],
    *,
    manifest_sha256: str,
    selection_sha256: str,
    feature_sha256: str,
    threshold_sha256: str,
) -> bool:
    """Verify cross-linked receipts before loading any probe feature or label."""
    categories = (
        "continuity",
        "solidity",
        "object_persistence",
        "unchangeableness",
        "directional_inertia",
    )
    report_threshold_sha = extraction_report.get("pre_probe_validation_threshold_sha256")
    return (
        extraction_report.get("status")
        == "complete_frozen_physical_concepts_probe_feature_extraction"
        and extraction_report.get("sample_selection_sha256") == selection_sha256
        and extraction_report.get("source_manifest_sha256") == manifest_sha256
        and extraction_report.get("feature_file_sha256") == feature_sha256
        and (report_threshold_sha is None or report_threshold_sha == threshold_sha256)
        and selection_receipt.get("source_manifest_sha256") == manifest_sha256
        and selection_receipt.get("pre_probe_validation_threshold_sha256")
        == threshold_sha256
        and selection_receipt.get("sample_quota_per_category")
        == {category: 512 for category in categories}
        and len(selection_receipt.get("sample_rows", [])) == 2560
        and selection_receipt.get("probe_data_opened_after_model_selection") is True
    )


def anomaly_probability(scores: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    if not math.isfinite(threshold) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("threshold must be finite and scale positive")
    logits = np.clip((np.asarray(scores, dtype=np.float64) - threshold) / scale, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if bins < 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("ECE requires matching nonempty arrays and at least one bin")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid binary labels or probabilities")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        selected = (p >= edges[index]) & (
            (p < edges[index + 1]) if index < bins - 1 else (p <= edges[index + 1])
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(y[selected].mean()) - float(p[selected].mean())
            )
    return ece


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y) == 0 or len(y) != len(s) or len(y) != len(p):
        raise ValueError("binary metric inputs must be nonempty and have equal lengths")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("binary labels/scores are invalid")
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    predicted = p >= 0.5
    positive = y == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("both binary classes are required")
    return {
        "count": int(len(y)),
        "accuracy": float(np.mean(predicted == y)),
        "sensitivity": float(np.mean(predicted[positive])),
        "specificity": float(np.mean(~predicted[negative])),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "brier": float(np.mean(np.square(p - y))),
        "ece_10_bins": expected_calibration_error(y, p, 10),
        "auc": _binary_auc(y, s),
    }


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if positive == 0 or negative == 0 or len(y) != len(s):
        raise ValueError("AUC requires matching scores and both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and s[order[end]] == s[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2
        cursor = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def group_ranking_accuracy(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("group scores must be finite [quadruplets,4] values")
    gaps = values[:, 2:].mean(axis=1) - values[:, :2].mean(axis=1)
    return np.where(gaps > 0, 1.0, np.where(gaps == 0, 0.5, 0.0))


def bootstrap_group_accuracy(
    scores_by_category: dict[str, np.ndarray], *, seed: int = 17, iterations: int = 10_000
) -> dict[str, float]:
    if set(scores_by_category) != set(EXPECTED_CATEGORIES) or iterations < 100:
        raise ValueError("bootstrap needs all five categories and at least 100 resamples")
    accuracies = {
        name: group_ranking_accuracy(scores) for name, scores in scores_by_category.items()
    }
    observed = float(np.mean([values.mean() for values in accuracies.values()]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        estimates[iteration] = np.mean(
            [
                values[rng.integers(0, len(values), size=len(values))].mean()
                for values in accuracies.values()
            ]
        )
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return {
        "macro_group_accuracy": observed,
        "paired_quadruplet_bootstrap_seed": seed,
        "paired_quadruplet_bootstrap_iterations": iterations,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
    }


def _sequence_scores(model: Any, features: np.ndarray, batch_size: int = 64) -> np.ndarray:
    import torch

    from tiny_omni_decision.temporal_prediction import per_sequence_cosine_error

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    values = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            scores = per_sequence_cosine_error(model(batch), batch)
            outputs.append(scores.cpu().numpy())
    result = np.concatenate(outputs)
    if len(result) != len(features) or not np.isfinite(result).all():
        raise FloatingPointError("sequence scoring produced missing or non-finite values")
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-run-metadata", type=Path, required=True)
    parser.add_argument("--sample-selection", type=Path, required=True)
    parser.add_argument("--feature-extraction-report", type=Path, required=True)
    parser.add_argument("--pre-probe-validation-threshold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    import torch

    if str(SOURCE_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))
    from scripts.train_intphys_next_feature_predictor import load_feature_split
    from tiny_omni_decision.temporal_prediction import NextFrameFeaturePredictor

    args = _args()
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    manifest_path = args.manifest.resolve(strict=True)
    validation_path = args.validation_features.resolve(strict=True)
    checkpoint_path = args.predictor_checkpoint.resolve(strict=True)
    predictor_metadata_path = args.predictor_run_metadata.resolve(strict=True)
    probe_path = args.probe_features.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation output: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_PROBE_PAYLOAD_READ":
        raise ValueError("probe manifest status is not frozen")
    candidate = manifest["candidate"]
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != candidate["predictor_checkpoint_sha256"]:
        raise ValueError("predictor checkpoint differs from the frozen manifest")
    predictor_metadata = json.loads(predictor_metadata_path.read_text(encoding="utf-8"))
    if (
        predictor_metadata.get("status")
        != "complete_physical_concepts_frozen_feature_next_step_fit"
    ):
        raise ValueError("predictor run metadata is incomplete")
    if sha256_file(predictor_metadata_path) != candidate["predictor_run_metadata_sha256"]:
        raise ValueError("predictor metadata differs from frozen candidate receipt")
    if predictor_metadata.get("best_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("predictor checkpoint does not match its saved run metadata")

    validation, validation_ids, _ = load_feature_split(validation_path)
    if sha256_file(validation_path) != predictor_metadata["validation_features_sha256"]:
        raise ValueError("freeform validation features differ from the selected predictor run")
    if sha256_file(validation_path) != candidate["validation_feature_archive_sha256"]:
        raise ValueError("freeform validation features differ from frozen candidate receipt")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = NextFrameFeaturePredictor(feature_width=int(state["feature_width"]), hidden_width=64)
    model.load_state_dict(state["state_dict"], strict=True)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    validation_scores = _sequence_scores(model, validation)
    validation_mean_loss = float(validation_scores.mean())
    if abs(validation_mean_loss - predictor_metadata["reloaded_validation_cosine_loss"]) > 1e-7:
        raise ValueError("checkpoint validation replay differs from selected run metadata")
    threshold, scale = fixed_validation_threshold(validation_scores)
    threshold_receipt_path = args.pre_probe_validation_threshold.resolve(strict=True)
    threshold_receipt = json.loads(threshold_receipt_path.read_text(encoding="utf-8"))
    if (
        threshold_receipt.get("status") != "frozen_before_probe_payload_download"
        or threshold_receipt.get("predictor_checkpoint_sha256") != checkpoint_sha
        or threshold_receipt.get("validation_feature_archive_sha256")
        != sha256_file(validation_path)
        or threshold_receipt.get("validation_sample_count") != len(validation_ids)
        or threshold_receipt.get("validation_q95_anomaly_threshold") != threshold
        or threshold_receipt.get("validation_mad_scale") != scale
        or threshold_receipt.get("validation_mean_cosine_loss") != validation_mean_loss
    ):
        raise ValueError("pre-probe validation threshold receipt differs from replay")

    selection_path = args.sample_selection.resolve(strict=True)
    extraction_report_path = args.feature_extraction_report.resolve(strict=True)
    selection_receipt = json.loads(selection_path.read_text(encoding="utf-8"))
    extraction_report = json.loads(extraction_report_path.read_text(encoding="utf-8"))
    if not probe_receipts_match_frozen_protocol(
        extraction_report,
        selection_receipt,
        manifest_sha256=sha256_file(manifest_path),
        selection_sha256=sha256_file(selection_path),
        feature_sha256=sha256_file(probe_path),
        threshold_sha256=sha256_file(threshold_receipt_path),
    ):
        raise ValueError("probe feature/selection receipts do not match the frozen protocol")
    expected_ids = [row["sample_id"] for row in selection_receipt["sample_rows"]]
    expected_categories = [row["category"] for row in selection_receipt["sample_rows"]]
    expected_source_objects = [row["object_name"] for row in selection_receipt["sample_rows"]]
    expected_source_generations = [
        row["object_generation"] for row in selection_receipt["sample_rows"]
    ]
    if (
        len(set(expected_ids)) != 2560
        or selection_receipt.get("sample_ids_order_sha256")
        != hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    ):
        raise ValueError("frozen sample selection order/hash is invalid")

    # Only after the candidate, validation replay, and anomaly threshold are frozen
    # does this function load the one-time probe feature/label cache.
    with np.load(probe_path, allow_pickle=False) as payload:
        probe_features = np.asarray(payload["features"], dtype=np.float32)
        sample_ids = payload["sample_ids"].astype(str)
        categories = payload["categories"].astype(str)
        source_objects = payload["source_objects"].astype(str)
        labels = payload["clip_labels"].astype(np.int8).reshape(-1, 4)
    if (
        probe_features.shape != (2560, 4, 8, int(state["feature_width"]))
        or len(sample_ids) != 2560
        or len(set(sample_ids.tolist())) != 2560
        or labels.shape != (2560, 4)
        or len(categories) != 2560
        or len(source_objects) != 2560
        or sample_ids.tolist() != expected_ids
        or categories.tolist() != expected_categories
        or source_objects.tolist() != expected_source_objects
        or not np.isfinite(probe_features).all()
    ):
        raise ValueError("probe feature cache differs from the frozen sample protocol")
    for category in EXPECTED_CATEGORIES:
        if np.sum(categories == category) != 512:
            raise ValueError(f"probe sample count does not match frozen quota for {category}")
    expected_labels = np.tile(np.asarray([0, 0, 1, 1], dtype=np.int8), (len(sample_ids), 1))
    if not np.array_equal(labels, expected_labels):
        raise ValueError("probe possible/impossible label ordering is invalid")

    flat_features = probe_features.reshape(-1, probe_features.shape[2], probe_features.shape[3])
    flat_scores = _sequence_scores(model, flat_features)
    clip_scores = flat_scores.reshape(-1, 4)
    probabilities = anomaly_probability(clip_scores, threshold, scale)
    clip_labels = labels.reshape(-1)
    clip_metrics = binary_metrics(clip_labels, clip_scores.reshape(-1), probabilities.reshape(-1))
    per_category: dict[str, dict[str, Any]] = {}
    score_groups: dict[str, np.ndarray] = {}
    group_lines: list[str] = []
    clip_lines: list[str] = []
    for category in EXPECTED_CATEGORIES:
        indices = np.flatnonzero(categories == category)
        scores = clip_scores[indices]
        probs = probabilities[indices]
        truth = labels[indices]
        category_flat_metrics = binary_metrics(
            truth.reshape(-1), scores.reshape(-1), probs.reshape(-1)
        )
        group_accuracy_values = group_ranking_accuracy(scores)
        gaps = scores[:, 2:].mean(axis=1) - scores[:, :2].mean(axis=1)
        score_groups[category] = scores
        per_category[category] = {
            "quadruplet_count": int(len(indices)),
            "clip_metrics": category_flat_metrics,
            "group_ranking_accuracy": float(group_accuracy_values.mean()),
            "group_tie_count": int(np.sum(gaps == 0)),
            "mean_impossible_minus_possible_surprise": float(gaps.mean()),
        }
        for local_index, global_index in enumerate(indices):
            gap = float(gaps[local_index])
            group_lines.append(
                json.dumps(
                    {
                        "category": category,
                        "sample_id": sample_ids[global_index],
                        "source_object": source_objects[global_index],
                        "source_generation": expected_source_generations[global_index],
                        "mean_possible_surprise": float(scores[local_index, :2].mean()),
                        "mean_impossible_surprise": float(scores[local_index, 2:].mean()),
                        "impossible_minus_possible_gap": gap,
                        "group_prediction_correct": bool(gap > 0) if gap != 0 else None,
                        "group_accuracy_credit": float(group_accuracy_values[local_index]),
                    },
                    sort_keys=True,
                )
            )
            for clip_index in range(4):
                clip_lines.append(
                    json.dumps(
                        {
                            "category": category,
                            "sample_id": sample_ids[global_index],
                            "source_object": source_objects[global_index],
                            "source_generation": expected_source_generations[global_index],
                            "clip_index": clip_index,
                            "label_impossible": int(truth[local_index, clip_index]),
                            "surprise_score": float(scores[local_index, clip_index]),
                            "probability_impossible": float(probs[local_index, clip_index]),
                            "prediction_impossible": bool(probs[local_index, clip_index] >= 0.5),
                        },
                        sort_keys=True,
                    )
                )
    group_path = output_dir / "quadruplet-predictions.jsonl"
    clips_path = output_dir / "clip-predictions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=False)
    group_path.write_text("\n".join(group_lines) + "\n", encoding="utf-8")
    clips_path.write_text("\n".join(clip_lines) + "\n", encoding="utf-8")
    report = {
        "status": "complete_one_time_physical_concepts_probe_diagnostic",
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "device": "cpu",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "source_manifest_sha256": sha256_file(manifest_path),
        "probe_feature_file_sha256": sha256_file(probe_path),
        "validation_features_sha256": sha256_file(validation_path),
        "predictor_checkpoint_sha256": checkpoint_sha,
        "predictor_metadata_sha256": sha256_file(predictor_metadata_path),
        "pre_probe_validation_threshold_sha256": sha256_file(threshold_receipt_path),
        "validation_sample_ids_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode()
        ).hexdigest(),
        "validation_sequence_count": len(validation_scores),
        "validation_mean_cosine_loss_replay": validation_mean_loss,
        "validation_q95_anomaly_threshold": threshold,
        "validation_mad_scale": scale,
        "calibration_note": (
            "probabilities use a one-class validation anomaly threshold; "
            "NLL/Brier/ECE are diagnostics, not fitted or claimed calibrated"
        ),
        "clip_metrics_all_categories_sample_weighted": clip_metrics,
        "per_category": per_category,
        "macro_clip_metrics": {
            key: float(
                np.mean([per_category[name]["clip_metrics"][key] for name in EXPECTED_CATEGORIES])
            )
            for key in (
                "accuracy",
                "sensitivity",
                "specificity",
                "nll",
                "brier",
                "ece_10_bins",
                "auc",
            )
        },
        "paired_quadruplet_bootstrap": bootstrap_group_accuracy(
            score_groups, seed=args.seed, iterations=args.bootstrap_iterations
        ),
        "predictions": {
            "quadruplet_file": group_path.name,
            "quadruplet_sha256": sha256_file(group_path),
            "clip_file": clips_path.name,
            "clip_sha256": sha256_file(clips_path),
            "clip_count": int(clip_scores.size),
            "quadruplet_count": int(len(sample_ids)),
        },
        "tuning_after_probe": False,
        "quality_scope": (
            "task-specific synthetic intuitive-physics probe transfer only; "
            "not broad video or multimodal Decision quality"
        ),
    }
    report_path = output_dir / "evaluation-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
