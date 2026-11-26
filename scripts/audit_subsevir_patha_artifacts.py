"""Verify best-checkpoint reload parity and the predeclared radar-shuffle control."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from train_subsevir_patha_lstm import (
    TemporalDecision,
    _group_bootstrap,
    _per_example_losses,
    metrics,
    predict,
    sha256_file,
)


def _load_validation(
    manifest_path: Path,
    feature_cache: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows = [window for window in manifest["windows"] if window["split"] == "validation"]
    indices = np.load(feature_cache / "frame_indices.npy", allow_pickle=False)
    splits = np.load(feature_cache / "frame_splits.npy", allow_pickle=False)
    event_ids = np.load(feature_cache / "frame_event_ids.npy", allow_pickle=False)
    features = np.load(feature_cache / "frame_features_fp32.npy", mmap_mode="r", allow_pickle=False)
    positions = {int(source_index): position for position, source_index in enumerate(indices)}
    rows = []
    sequences = []
    for window in windows:
        frames = []
        for source_index in window["input_indices"]:
            position = positions[int(source_index)]
            if splits[position] != "validation" or event_ids[position] != window["event_id"]:
                raise ValueError(f"validation frame provenance mismatch in {window['sample_id']}")
            frames.append(np.asarray(features[position], dtype=np.float32))
        sequences.append(np.stack(frames))
        rows.append(window)
    x = np.stack(sequences).astype(np.float32, copy=False)
    y = np.asarray([int(window["target"]) for window in windows], dtype=np.int64)
    return x, y, rows, indices


def _load_saved_predictions(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    results_path = args.run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("training result belongs to a different frozen manifest")
    cache_meta = json.loads((args.feature_cache / "metadata.json").read_text())
    if results["feature_metadata_sha256"] != sha256_file(args.feature_cache / "metadata.json"):
        raise ValueError("training result references different feature metadata")
    feature_path = args.feature_cache / "frame_features_fp32.npy"
    if sha256_file(feature_path) != cache_meta["feature_file_sha256"]:
        raise ValueError("feature cache hash mismatch")

    x, y, windows, _indices = _load_validation(args.manifest, args.feature_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parity = {}
    predictions_by_mode = {}
    probabilities_by_mode = {}
    for mode in ("satellite", "radar", "joint"):
        candidate_dir = args.run_dir / mode
        summary = json.loads((candidate_dir / "summary.json").read_text())
        best_path = candidate_dir / "best.safetensors"
        final_path = candidate_dir / "final.safetensors"
        if sha256_file(best_path) != summary["best_checkpoint_sha256"]:
            raise ValueError(f"best checkpoint hash mismatch for {mode}")
        if sha256_file(final_path) != summary["final_checkpoint_sha256"]:
            raise ValueError(f"final checkpoint hash mismatch for {mode}")
        model = TemporalDecision(mode).to(device)
        model.load_state_dict(load_file(str(best_path), device="cpu"))
        logits = predict(model, torch.from_numpy(x), device)
        metric = metrics(y, logits)
        probabilities = metric["probabilities"]
        saved = _load_saved_predictions(candidate_dir / "validation-predictions.jsonl")
        if len(saved) != len(windows):
            raise ValueError(f"prediction count mismatch for {mode}")
        saved_probabilities = np.asarray([row["probabilities"] for row in saved])
        if any(
            row["sample_id"] != window["sample_id"]
            for row, window in zip(saved, windows, strict=True)
        ):
            raise ValueError(f"prediction IDs/order mismatch for {mode}")
        max_probability_delta = float(np.max(np.abs(probabilities - saved_probabilities)))
        if max_probability_delta > 1e-6:
            raise ValueError(
                f"best reload prediction parity failed for {mode}: {max_probability_delta}"
            )
        parity[mode] = {
            "best_epoch": summary["best_epoch"],
            "best_and_final_hashes_differ": sha256_file(best_path) != sha256_file(final_path),
            "max_probability_delta": max_probability_delta,
            "reloaded_metrics": {
                key: metric[key]
                for key in ("accuracy", "nll", "brier", "ece_15", "mean_confidence")
            },
        }
        predictions_by_mode[mode] = metric["predictions"]
        probabilities_by_mode[mode] = probabilities

    # Match donor event groups by their available forecast horizons so that
    # the shuffle does not exchange a missing/short sequence with a full one.
    group_rows: dict[str, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        group_rows[str(window["event_id"])].append(index)
    grouped: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for event_id, positions in group_rows.items():
        base_index = min(
            int(source_index)
            for position in positions
            for source_index in windows[position]["input_indices"]
        )
        offsets = tuple(
            int(windows[position]["target_index"]) - base_index
            for position in positions
        )
        grouped[offsets].append(event_id)

    shuffled_x = x.copy()
    rng = np.random.default_rng(17)
    shuffled_groups = 0
    for _signature, group_ids in grouped.items():
        if len(group_ids) < 2:
            continue
        ordered = sorted(group_ids)
        permuted = list(rng.permutation(ordered))
        if all(
            source == destination
            for source, destination in zip(ordered, permuted, strict=True)
        ):
            permuted = permuted[1:] + permuted[:1]
        donor_rows = {
            event_id: sorted(group_rows[event_id], key=lambda row: windows[row]["target_index"])
            for event_id in ordered
        }
        for destination, source in zip(ordered, permuted, strict=True):
            target_positions = donor_rows[destination]
            source_positions = donor_rows[source]
            for target_row, source_row in zip(target_positions, source_positions, strict=True):
                shuffled_x[target_row, :, 1, :] = x[source_row, :, 1, :]
            shuffled_groups += 1

    joint_path = args.run_dir / "joint" / "best.safetensors"
    joint_model = TemporalDecision("joint").to(device)
    joint_model.load_state_dict(load_file(str(joint_path), device="cpu"))
    shuffled_logits = predict(joint_model, torch.from_numpy(shuffled_x), device)
    shuffled_metrics = metrics(y, shuffled_logits)
    intact_probs = probabilities_by_mode["joint"]
    shuffled_probs = shuffled_metrics["probabilities"]
    intact_predictions = predictions_by_mode["joint"]
    shuffled_nll, shuffled_brier = _per_example_losses(y, shuffled_probs)
    intact_nll, intact_brier = _per_example_losses(y, intact_probs)
    group_ids = [str(window["event_id"]) for window in windows]
    paired = _group_bootstrap(
        y,
        (shuffled_metrics["predictions"] == y).astype(np.float64),
        shuffled_nll,
        shuffled_brier,
        group_ids,
        (intact_predictions == y).astype(np.float64),
        intact_nll,
        intact_brier,
    )
    result = {
        "schema": "tiny-omni-decision/subsevir-patha-artifact-audit-v1",
        "status": "pass",
        "manifest_sha256": sha256_file(args.manifest),
        "feature_cache_metadata_sha256": sha256_file(args.feature_cache / "metadata.json"),
        "run_results_sha256": sha256_file(results_path),
        "best_checkpoint_reload_parity": parity,
        "selected_joint_radar_event_shuffle": {
            "description": (
                "validation radar sequences shuffled across event groups "
                "matched by forecast horizons; no refit"
            ),
            "shuffled_event_groups": shuffled_groups,
            "unshuffled_event_groups": sum(len(ids) == 1 for ids in grouped.values()),
            "metrics": {
                key: shuffled_metrics[key]
                for key in (
                    "accuracy",
                    "nll",
                    "brier",
                    "ece_15",
                    "mean_confidence",
                    "correct_count",
                )
            },
            "paired_group_bootstrap_95pct_difference_shuffled_minus_intact": paired[
                "paired_difference_a_minus_b"
            ],
        },
        "validation_event_groups": len(group_rows),
        "bootstrap": {"replicates": 10_000, "seed": 17, "unit": "storm event ID"},
        "sealed_audit_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
