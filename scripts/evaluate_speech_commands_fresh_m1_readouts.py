"""One-time paired evaluation of the frozen and fresh-speaker audio readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SOURCE = "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da"
FROZEN_READOUT_SHA256 = "171f6a9303b2fd9ad63efe2bee6af119549fb4f26ea663e9c1297f51a678e1c3"
FROZEN_TEMPERATURE = 0.42702477413437295


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(targets)
    one_hot = F.one_hot(targets, num_classes=len(LABELS)).float()
    ece = 0.0
    for bin_index in range(15):
        mask = (confidence >= bin_index / 15) & (
            confidence < (bin_index + 1) / 15 if bin_index < 14 else confidence <= 1
        )
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].float().mean().item() - confidence[mask].mean().item()
            )
    return {
        "count": len(targets),
        "accuracy": correct.float().mean().item(),
        "nll": F.cross_entropy(logits.float(), targets).item(),
        "brier": (probabilities - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15_bins": ece,
        "mean_confidence": confidence.mean().item(),
    }


def _per_class(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, dict[str, float | int]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    rows: dict[str, dict[str, float | int]] = {}
    for class_id, label in enumerate(LABELS):
        mask = targets.eq(class_id)
        selected = probabilities[mask]
        selected_targets = targets[mask]
        one_hot = F.one_hot(selected_targets, num_classes=len(LABELS)).float()
        correct = predictions[mask].eq(class_id)
        rows[label] = {
            "support": int(mask.sum()),
            "recall": correct.float().mean().item(),
            "nll": -selected[:, class_id].clamp_min(1e-12).log().mean().item(),
            "brier": (selected - one_hot).square().sum(dim=-1).mean().item(),
            "one_vs_rest_ece_15_bins": _binary_ece(
                probabilities[:, class_id], targets.eq(class_id).float()
            ),
            "mean_target_probability": selected[:, class_id].mean().item(),
        }
    return rows


def _binary_ece(confidence: torch.Tensor, correct: torch.Tensor) -> float:
    value = 0.0
    for index in range(15):
        mask = (confidence >= index / 15) & (
            confidence < (index + 1) / 15 if index < 14 else confidence <= 1
        )
        if mask.any():
            value += mask.float().mean().item() * abs(
                correct[mask].mean().item() - confidence[mask].mean().item()
            )
    return value


def _group_stats(
    probabilities: torch.Tensor, targets: torch.Tensor, groups: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = sorted(set(groups))
    group_index = {group: index for index, group in enumerate(unique_groups)}
    group_rows = np.array([group_index[group] for group in groups], dtype=np.int64)
    p = probabilities.cpu().numpy().astype(np.float64)
    y = targets.cpu().numpy().astype(np.int64)
    confidence = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
    one_hot = np.eye(len(LABELS), dtype=np.float64)[y]
    brier = ((p - one_hot) ** 2).sum(axis=1)
    bins = np.minimum((confidence * 15).astype(np.int64), 14)
    counts = np.zeros((len(unique_groups), 15), dtype=np.float64)
    bin_correct = np.zeros_like(counts)
    bin_confidence = np.zeros_like(counts)
    totals = np.zeros((len(unique_groups), 3), dtype=np.float64)
    np.add.at(counts, (group_rows, bins), 1.0)
    np.add.at(bin_correct, (group_rows, bins), correct)
    np.add.at(bin_confidence, (group_rows, bins), confidence)
    np.add.at(totals, group_rows, np.stack((correct, nll, brier), axis=1))
    group_sizes = np.bincount(group_rows, minlength=len(unique_groups)).astype(np.float64)
    return unique_groups, counts, bin_correct, bin_confidence, totals, group_sizes


def _bootstrap(
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    seed: int = 17,
    repeats: int = 10_000,
) -> dict[str, Any]:
    base_counts, base_correct, base_conf, base_totals, base_sizes = baseline
    cand_counts, cand_correct, cand_conf, cand_totals, cand_sizes = candidate
    if not (len(base_sizes) == len(cand_sizes)):
        raise ValueError("paired bootstrap requires the same speaker groups")
    rng = np.random.default_rng(seed)
    group_count = len(base_sizes)
    sampled = rng.integers(0, group_count, size=(repeats, group_count), dtype=np.int32)

    def summarize(parts: tuple[np.ndarray, ...], indexes: np.ndarray) -> np.ndarray:
        counts, correct_bins, confidence_bins, totals, sizes = parts
        result = np.empty((len(indexes), 4), dtype=np.float64)
        for start in range(0, len(indexes), 128):
            block = indexes[start : start + 128]
            denoms = sizes[block].sum(axis=1)
            sums = totals[block].sum(axis=1)
            bin_counts = counts[block].sum(axis=1)
            bin_correct_sums = correct_bins[block].sum(axis=1)
            bin_conf_sums = confidence_bins[block].sum(axis=1)
            bin_acc = np.divide(
                bin_correct_sums,
                bin_counts,
                out=np.zeros_like(bin_correct_sums),
                where=bin_counts > 0,
            )
            bin_conf = np.divide(
                bin_conf_sums,
                bin_counts,
                out=np.zeros_like(bin_conf_sums),
                where=bin_counts > 0,
            )
            ece = ((bin_counts / denoms[:, None]) * np.abs(bin_acc - bin_conf)).sum(axis=1)
            result[start : start + len(block), 0] = sums[:, 0] / denoms
            result[start : start + len(block), 1] = sums[:, 1] / denoms
            result[start : start + len(block), 2] = sums[:, 2] / denoms
            result[start : start + len(block), 3] = ece
        return result

    baseline_parts = (base_counts, base_correct, base_conf, base_totals, base_sizes)
    candidate_parts = (cand_counts, cand_correct, cand_conf, cand_totals, cand_sizes)
    base_values = summarize(baseline_parts, sampled)
    candidate_values = summarize(candidate_parts, sampled)
    names = ["accuracy", "nll", "brier", "ece_15_bins"]
    report: dict[str, Any] = {
        "method": (
            "paired speaker-cluster bootstrap; sample 200 speaker groups with replacement; "
            "retain all rows per sampled group"
        ),
        "repeats": repeats,
        "seed": seed,
        "confidence_level": 0.95,
        "groups": group_count,
    }
    report["baseline_ci95"] = {
        name: np.quantile(base_values[:, index], [0.025, 0.975]).tolist()
        for index, name in enumerate(names)
    }
    report["candidate_ci95"] = {
        name: np.quantile(candidate_values[:, index], [0.025, 0.975]).tolist()
        for index, name in enumerate(names)
    }
    report["candidate_minus_baseline_ci95"] = {
        name: np.quantile(
            candidate_values[:, index] - base_values[:, index], [0.025, 0.975]
        ).tolist()
        for index, name in enumerate(names)
    }
    return report


def run(
    feature_cache_path: Path,
    sample_dir: Path,
    baseline_weights: Path,
    baseline_report_path: Path,
    candidate_dir: Path,
    candidate_report_path: Path,
    output_dir: Path,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    paths = [
        p.resolve()
        for p in (
            feature_cache_path,
            sample_dir,
            baseline_weights,
            baseline_report_path,
            candidate_dir,
            candidate_report_path,
            output_dir,
        )
    ]
    (
        feature_cache_path,
        sample_dir,
        baseline_weights,
        baseline_report_path,
        candidate_dir,
        candidate_report_path,
        output_dir,
    ) = paths
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output path: {output_dir}")
    train_rows = [
        json.loads(line)
        for line in (sample_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    validation_rows = [
        json.loads(line)
        for line in (sample_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(row.get("source") != SOURCE for row in train_rows + validation_rows):
        raise ValueError("sample manifests have unexpected dataset revisions")
    if {row["speaker_group_sha256"] for row in train_rows} & {
        row["speaker_group_sha256"] for row in validation_rows
    }:
        raise ValueError("train and validation speaker groups overlap")
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=True)
    if (
        cache.get("source_revision") != SOURCE
        or cache.get("encoder_revision") != "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
    ):
        raise ValueError("feature cache source or encoder revision mismatch")
    if cache.get("train_ids") != [row["id"] for row in train_rows] or cache.get(
        "validation_ids"
    ) != [row["id"] for row in validation_rows]:
        raise ValueError("feature cache IDs/order differ from frozen manifests")
    train_targets = torch.tensor([LABELS.index(row["target"]) for row in train_rows])
    validation_targets = torch.tensor([LABELS.index(row["target"]) for row in validation_rows])
    if (
        cache["train_targets"] != train_targets.tolist()
        or cache["validation_targets"] != validation_targets.tolist()
    ):
        raise ValueError("feature cache targets differ from frozen manifests")
    train_features = cache["train_features"].float()
    validation_features = cache["validation_features"].float()
    if train_features.shape != (len(train_rows), 384) or validation_features.shape != (
        len(validation_rows),
        384,
    ):
        raise ValueError("unexpected feature matrix shape")
    if not torch.isfinite(train_features).all() or not torch.isfinite(validation_features).all():
        raise ValueError("feature cache contains nonfinite values")

    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if baseline_report.get("weights_sha256") != FROZEN_READOUT_SHA256:
        raise ValueError("baseline report does not identify the frozen readout checkpoint")
    baseline_weight_hash = sha256_file(baseline_weights)
    if baseline_weight_hash != FROZEN_READOUT_SHA256:
        raise ValueError(f"baseline readout hash mismatch: {baseline_weight_hash}")
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if candidate_report.get("status") != "complete_text_free_audio_calibration_comparison":
        raise ValueError("candidate training report is incomplete")
    if candidate_report.get("feature_cache_sha256") != sha256_file(feature_cache_path):
        raise ValueError("candidate training did not use this feature cache")
    if candidate_report.get("validation_ids_sha256") != _sha256_text(
        "\n".join(cache["validation_ids"])
    ):
        raise ValueError("candidate training report validation IDs differ")
    candidate_weights = candidate_dir / "last-readout.pt"
    candidate_weight_hash = sha256_file(candidate_weights)
    if candidate_report.get("weights_sha256") != candidate_weight_hash:
        raise ValueError("candidate readout hash differs from its training report")
    candidate_temperature = float(candidate_report["calibration_temperature"])
    if not math.isfinite(candidate_temperature) or candidate_temperature <= 0:
        raise ValueError("invalid candidate calibration temperature")

    baseline = TextFreeAudioReadout(384, LABELS).eval()
    baseline.load_state_dict(torch.load(baseline_weights, map_location="cpu", weights_only=True))
    candidate = TextFreeAudioReadout(384, LABELS).eval()
    candidate.load_state_dict(torch.load(candidate_weights, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        baseline_logits = baseline(validation_features)
        candidate_logits = candidate(validation_features)
        candidate_train_logits = candidate(train_features)
    baseline_logits_scaled = baseline_logits / FROZEN_TEMPERATURE
    candidate_logits_scaled = candidate_logits / candidate_temperature
    baseline_probs = torch.softmax(baseline_logits_scaled, dim=-1)
    candidate_probs = torch.softmax(candidate_logits_scaled, dim=-1)
    groups = [row["speaker_group_sha256"] for row in validation_rows]
    base_group_stats = _group_stats(baseline_probs, validation_targets, groups)
    cand_group_stats = _group_stats(candidate_probs, validation_targets, groups)
    if base_group_stats[0] != cand_group_stats[0]:
        raise ValueError("paired bootstrap group order mismatch")
    base_boot = base_group_stats[1:]
    cand_boot = cand_group_stats[1:]
    bootstrap = _bootstrap(base_boot, cand_boot, repeats=bootstrap_repeats)

    train_class_counts = np.bincount(train_targets.numpy(), minlength=len(LABELS))
    majority_class_id = int(np.flatnonzero(train_class_counts == train_class_counts.max())[0])
    uniform_logits = torch.zeros((len(validation_rows), len(LABELS)))
    majority_logits = torch.full((len(validation_rows), len(LABELS)), -50.0)
    majority_logits[:, majority_class_id] = 0.0
    report = {
        "schema_version": 1,
        "status": "FRESH_M1_PAIRED_READOUT_EVALUATION_COMPLETE",
        "claim_scope": (
            "speaker-disjoint English closed-set Speech Commands keyword classification only"
        ),
        "validation_ids_sha256": _sha256_text("\n".join(cache["validation_ids"])),
        "validation_examples": len(validation_rows),
        "validation_unique_speakers": len(set(groups)),
        "source_revision": SOURCE,
        "feature_cache_sha256": sha256_file(feature_cache_path),
        "train_manifest_sha256": sha256_file(sample_dir / "train.jsonl"),
        "validation_manifest_sha256": sha256_file(sample_dir / "validation.jsonl"),
        "baseline": {
            "name": "frozen prior 3,850-parameter readout; no refit on fresh development data",
            "weights_sha256": baseline_weight_hash,
            "temperature": FROZEN_TEMPERATURE,
            "metrics": _metrics(baseline_logits_scaled, validation_targets),
            "per_class": _per_class(baseline_logits_scaled, validation_targets),
        },
        "candidate": {
            "name": "seed-17 fresh-speaker readout; train-only temperature calibration",
            "weights_sha256": candidate_weight_hash,
            "temperature": candidate_temperature,
            "training_report_sha256": sha256_file(candidate_report_path),
            "metrics": _metrics(candidate_logits_scaled, validation_targets),
            "per_class": _per_class(candidate_logits_scaled, validation_targets),
            "train_metrics": _metrics(candidate_train_logits, train_targets),
        },
        "baselines": {
            "uniform": _metrics(uniform_logits, validation_targets),
            "training_majority": {
                "class": LABELS[majority_class_id],
                "train_class_counts": dict(zip(LABELS, train_class_counts.tolist(), strict=True)),
                "metrics": _metrics(majority_logits, validation_targets),
            },
        },
        "candidate_minus_baseline": {
            name: report_candidate - report_baseline
            for name, report_candidate, report_baseline in (
                (
                    "accuracy",
                    _metrics(candidate_logits_scaled, validation_targets)["accuracy"],
                    _metrics(baseline_logits_scaled, validation_targets)["accuracy"],
                ),
                (
                    "nll",
                    _metrics(candidate_logits_scaled, validation_targets)["nll"],
                    _metrics(baseline_logits_scaled, validation_targets)["nll"],
                ),
                (
                    "brier",
                    _metrics(candidate_logits_scaled, validation_targets)["brier"],
                    _metrics(baseline_logits_scaled, validation_targets)["brier"],
                ),
                (
                    "ece_15_bins",
                    _metrics(candidate_logits_scaled, validation_targets)["ece_15_bins"],
                    _metrics(baseline_logits_scaled, validation_targets)["ece_15_bins"],
                ),
            )
        },
        "speaker_cluster_bootstrap": bootstrap,
        "selection_integrity": {
            "candidate_epochs": candidate_report.get("config", {}).get("epochs"),
            "checkpoint_rule": "fixed last epoch; no validation checkpoint selection",
            "temperature_fit_split": "speaker-disjoint calibration subset from train only",
            "validation_used_for_training_or_temperature_fit": False,
            "sealed_audit_loaded": False,
        },
        "prediction_write_is_one_time_development_readout": True,
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "paired-validation-predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for index, (row, target) in enumerate(
            zip(validation_rows, validation_targets.tolist(), strict=True)
        ):
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "speaker_group_sha256": row["speaker_group_sha256"],
                        "target": LABELS[target],
                        "baseline_probabilities": baseline_probs[index].tolist(),
                        "candidate_probabilities": candidate_probs[index].tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    report["predictions_sha256"] = sha256_file(output_dir / "paired-validation-predictions.jsonl")
    (output_dir / "comparison-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report["comparison_report_sha256"] = sha256_file(output_dir / "comparison-report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 1000:
        raise ValueError("speaker bootstrap must use at least 1,000 repeats")
    print(
        json.dumps(
            run(
                args.feature_cache,
                args.sample_dir,
                args.baseline_weights,
                args.baseline_report,
                args.candidate_dir,
                args.candidate_report,
                args.output_dir,
                args.bootstrap_repeats,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
