"""Compare two frozen-feature BSD10k readouts on identical validation examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_bsd10k_audio_readout import LABELS  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _metrics(rows: list[dict], probability_key: str) -> dict:
    targets = np.asarray([LABELS.index(row["target_class_id"]) for row in rows])
    probabilities = np.asarray([row[probability_key] for row in rows], dtype=np.float64)
    if probabilities.shape != (len(rows), len(LABELS)) or not np.isfinite(probabilities).all():
        raise ValueError("invalid saved validation probabilities")
    correct = probabilities.argmax(axis=1) == targets
    per_class = {}
    for class_id, class_index in zip(LABELS, range(len(LABELS)), strict=True):
        mask = targets == class_index
        selected = probabilities[mask]
        selected_correct = correct[mask]
        confidence = selected.max(axis=1)
        ece = 0.0
        for bin_index in range(15):
            lower, upper = bin_index / 15, (bin_index + 1) / 15
            bin_mask = (confidence >= lower) & (
                confidence < upper if bin_index < 14 else confidence <= upper
            )
            if bin_mask.any():
                ece += bin_mask.mean() * abs(
                    selected_correct[bin_mask].mean() - confidence[bin_mask].mean()
                )
        per_class[class_id] = {
            "count": int(mask.sum()),
            "accuracy": float(selected_correct.mean()),
            "nll": float(
                -np.log(np.clip(selected[np.arange(mask.sum()), class_index], 1e-15, 1)).mean()
            ),
            "brier": float(((selected - np.eye(len(LABELS))[class_index]) ** 2).sum(axis=1).mean()),
            "ece_15_bins": float(ece),
        }
    macro = {
        key: float(np.mean([item[key] for item in per_class.values()]))
        for key in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    return {"sample_accuracy": float(correct.mean()), "macro": macro, "per_class": per_class}


def _paired_group_bootstrap(
    baseline: list[dict],
    candidate: list[dict],
    group_by_id: dict[str, str],
    *,
    replicates: int = 2_000,
    seed: int = 17,
) -> dict:
    ids = [row["sound_id"] for row in baseline]
    groups = sorted({group_by_id[sound_id] for sound_id in ids})
    group_index = {group: index for index, group in enumerate(groups)}
    row_groups = np.asarray([group_index[group_by_id[sound_id]] for sound_id in ids])
    targets = np.asarray([LABELS.index(row["target_class_id"]) for row in baseline])
    base_probs = np.asarray([row["probabilities_uncalibrated"] for row in baseline])
    candidate_probs = np.asarray([row["probabilities_uncalibrated"] for row in candidate])
    base_correct = base_probs.argmax(axis=1) == targets
    candidate_correct = candidate_probs.argmax(axis=1) == targets
    correctness_delta = candidate_correct.astype(np.float64) - base_correct.astype(np.float64)
    base_nll = -np.log(np.clip(base_probs[np.arange(len(ids)), targets], 1e-15, 1))
    candidate_nll = -np.log(np.clip(candidate_probs[np.arange(len(ids)), targets], 1e-15, 1))
    rng = np.random.default_rng(seed)
    accuracy_diffs = np.empty(replicates)
    nll_diffs = np.empty(replicates)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        multiplicity = np.bincount(sampled, minlength=len(groups))
        weights = multiplicity[row_groups]
        class_accuracy_diffs = []
        class_nll_diffs = []
        for class_index in range(len(LABELS)):
            mask = targets == class_index
            denominator = np.sum(weights[mask])
            if denominator:
                class_accuracy_diffs.append(
                    np.sum(weights[mask] * correctness_delta[mask]) / denominator
                )
                class_nll_diffs.append(
                    np.sum(weights[mask] * (candidate_nll[mask] - base_nll[mask])) / denominator
                )
        accuracy_diffs[replicate] = np.mean(class_accuracy_diffs)
        nll_diffs[replicate] = np.mean(class_nll_diffs)

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "cluster": "frozen validation uploader group",
        "independent_groups": len(groups),
        "replicates": replicates,
        "seed": seed,
        "interval_method": "paired percentile bootstrap sampling uploader groups with replacement",
        "macro_accuracy_difference_candidate_minus_baseline_95ci": interval(accuracy_diffs),
        "macro_nll_difference_candidate_minus_baseline_95ci": interval(nll_diffs),
    }


def compare(
    baseline_report_path: Path,
    baseline_predictions_path: Path,
    candidate_report_path: Path,
    candidate_predictions_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    *,
    replicates: int = 2_000,
    seed: int = 17,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite comparison report: {output_path}")
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    baseline = _load_predictions(baseline_predictions_path)
    candidate = _load_predictions(candidate_predictions_path)
    base_ids = [row["sound_id"] for row in baseline]
    candidate_ids = [row["sound_id"] for row in candidate]
    if base_ids != candidate_ids or any(
        left["target_class_id"] != right["target_class_id"]
        for left, right in zip(baseline, candidate, strict=True)
    ):
        raise ValueError("validation IDs, order or target labels differ between runs")
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    group_by_id = {
        str(row["sound_id"]): str(row["uploader_group_sha256"])
        for row in manifest["examples"]
        if row["split"] == "validation"
    }
    if set(base_ids) != set(group_by_id):
        raise ValueError("saved predictions do not cover the frozen validation set")
    baseline_metrics = {
        "uncalibrated": _metrics(baseline, "probabilities_uncalibrated"),
        "temperature_scaled": _metrics(baseline, "probabilities_temperature_scaled"),
    }
    candidate_metrics = {
        "uncalibrated": _metrics(candidate, "probabilities_uncalibrated"),
        "temperature_scaled": _metrics(candidate, "probabilities_temperature_scaled"),
    }
    base_steps = baseline_report.get("optimizer_steps")
    if base_steps is None:
        fit_examples = int(baseline_report["fit_examples"])
        batch_size = int(baseline_report["config"]["batch_size"])
        base_steps = ((fit_examples + batch_size - 1) // batch_size) * int(
            baseline_report["epochs_run"]
        )
    base_examples_seen = baseline_report.get("training_examples_seen")
    if base_examples_seen is None:
        base_examples_seen = int(baseline_report["fit_examples"]) * int(
            baseline_report["epochs_run"]
        )
    candidate_steps = candidate_report.get("optimizer_steps")
    if candidate_steps is None:
        candidate_fit_examples = int(candidate_report["fit_examples"])
        batch_size = int(candidate_report["config"]["batch_size"])
        candidate_steps = ((candidate_fit_examples + batch_size - 1) // batch_size) * int(
            candidate_report["epochs_run"]
        )
    candidate_examples_seen = candidate_report.get("training_examples_seen")
    if candidate_examples_seen is None:
        candidate_examples_seen = int(candidate_report["fit_examples"]) * int(
            candidate_report["epochs_run"]
        )
    report = {
        "schema": "tiny-omni-decision/bsd10k-audio-readout-comparison-v1",
        "status": "PASS_IDENTICAL_VALIDATION_IDS",
        "baseline": {
            "run_report_sha256": sha256_file(baseline_report_path),
            "predictions_sha256": sha256_file(baseline_predictions_path),
            "train_examples": baseline_report["train_examples"],
            "unique_train_groups": baseline_report["train_unique_uploader_groups"],
            "fit_examples": baseline_report["fit_examples"],
            "optimizer_steps": base_steps,
            "training_examples_seen": base_examples_seen,
            "epochs_run": baseline_report["epochs_run"],
            "selected_epoch": baseline_report["selected_epoch"],
            "metrics": baseline_metrics,
        },
        "candidate": {
            "run_report_sha256": sha256_file(candidate_report_path),
            "predictions_sha256": sha256_file(candidate_predictions_path),
            "train_examples": candidate_report["train_examples"],
            "unique_train_groups": candidate_report["train_unique_uploader_groups"],
            "fit_examples": candidate_report["fit_examples"],
            "optimizer_steps": candidate_steps,
            "training_examples_seen": candidate_examples_seen,
            "epochs_run": candidate_report["epochs_run"],
            "selected_epoch": candidate_report["selected_epoch"],
            "metrics": candidate_metrics,
        },
        "validation": {
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "example_count": len(base_ids),
            "same_ids_same_order_and_same_targets": True,
            "grouped_bootstrap": _paired_group_bootstrap(
                baseline, candidate, group_by_id, replicates=replicates, seed=seed
            ),
            "development_only_checkpoint_selection_used_validation": True,
            "test_split_loaded": False,
            "sealed_audit_loaded": False,
        },
        "single_seed": seed,
        "interpretation_limit": (
            "Paired development-validation comparison for this 23-class BSD10k task; "
            "not evidence for other modalities or tasks."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2_000)
    args = parser.parse_args()
    result = compare(
        args.baseline_report.resolve(strict=True),
        args.baseline_predictions.resolve(strict=True),
        args.candidate_report.resolve(strict=True),
        args.candidate_predictions.resolve(strict=True),
        args.split_manifest.resolve(strict=True),
        args.output.resolve(),
        replicates=args.replicates,
    )
    print(json.dumps({"status": result["status"], "validation": result["validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
