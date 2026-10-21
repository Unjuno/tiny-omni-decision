"""Recompute saved synthetic-AV prediction metrics without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    return rows, hashlib.sha256(raw).hexdigest()


def _probability_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    ids: set[str] = set()
    values: list[list[float]] = []
    targets: list[int] = []
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError("prediction rows require unique, nonempty string IDs")
        ids.add(sample_id)
        options = row.get("probabilities")
        target = row.get("target_index")
        if not isinstance(options, list) or len(options) != 2:
            raise ValueError("this audit accepts binary option probabilities only")
        if not isinstance(target, int) or target not in (0, 1):
            raise ValueError("target_index must be 0 or 1")
        probabilities = [option.get("probability") for option in options]
        try:
            pair = [float(value) for value in probabilities]
        except (TypeError, ValueError) as exc:
            raise ValueError("probabilities must be numeric") from exc
        vector = np.asarray(pair, dtype=np.float64)
        if not np.isfinite(vector).all() or np.any(vector < 0) or np.any(vector > 1):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not np.isclose(vector.sum(), 1.0, atol=1e-6, rtol=0):
            raise ValueError("each probability row must sum to one")
        expected = options[int(np.argmax(vector))].get("option")
        if row.get("prediction") != expected:
            raise ValueError("saved prediction does not match saved probabilities")
        values.append(pair)
        targets.append(target)
    return np.asarray(values, dtype=np.float64), np.asarray(targets, dtype=np.int64)


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == targets
    rows = np.arange(len(targets))
    nll = -np.log(np.clip(probabilities[rows, targets], 1e-15, 1.0))
    one_hot = np.eye(2, dtype=np.float64)[targets]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        mask = (confidence >= lower) & (
            (confidence < upper) if index < 14 else (confidence <= 1.0)
        )
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
    }


def _assert_paired_rows(
    reference: list[dict[str, Any]], other: list[dict[str, Any]], *, name: str
) -> None:
    fields = ("id", "target", "target_index", "pair_id", "scene_group_id")
    if len(reference) != len(other):
        raise ValueError(f"{name} row count differs from the reference")
    for index, (left, right) in enumerate(zip(reference, other, strict=True)):
        if any(left.get(field) != right.get(field) for field in fields):
            raise ValueError(f"{name} row identity/target mismatch at position {index}")


def _scene_bootstrap(
    rows: list[dict[str, Any]],
    left_probabilities: np.ndarray,
    left_targets: np.ndarray,
    right_probabilities: np.ndarray,
    right_targets: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if not np.array_equal(left_targets, right_targets):
        raise ValueError("paired targets differ")
    scene_indices: dict[str, list[int]] = defaultdict(list)
    pair_scenes: dict[str, str] = {}
    for index, row in enumerate(rows):
        scene = row.get("scene_group_id")
        pair = row.get("pair_id")
        if not isinstance(scene, str) or not scene or not isinstance(pair, str) or not pair:
            raise ValueError("each row needs scene_group_id and pair_id")
        if pair in pair_scenes and pair_scenes[pair] != scene:
            raise ValueError("paired variants cross scene groups")
        pair_scenes[pair] = scene
        scene_indices[scene].append(index)
    if any(len(indices) != 2 for indices in _pair_indices(rows).values()):
        raise ValueError("each event pair must contain exactly two rows")
    scenes = sorted(scene_indices)
    if len(scenes) < 2:
        raise ValueError("scene bootstrap needs at least two scene groups")

    def outcomes(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
        predictions = probabilities.argmax(axis=1)
        row_indexes = np.arange(len(targets))
        return {
            "accuracy": (predictions == targets).astype(np.float64),
            "nll": -np.log(np.clip(probabilities[row_indexes, targets], 1e-15, 1.0)),
            "brier": np.square(probabilities - np.eye(2)[targets]).sum(axis=1),
        }

    left = outcomes(left_probabilities, left_targets)
    right = outcomes(right_probabilities, right_targets)
    deltas = {
        "accuracy": left["accuracy"] - right["accuracy"],
        "nll": right["nll"] - left["nll"],
        "brier": right["brier"] - left["brier"],
    }
    scene_means = {
        metric: np.asarray([values[scene_indices[scene]].mean() for scene in scenes])
        for metric, values in deltas.items()
    }
    rng = np.random.default_rng(seed)
    sampled_scene_indices = rng.integers(0, len(scenes), size=(repetitions, len(scenes)))
    output = {}
    for metric, per_scene in scene_means.items():
        bootstrap = per_scene[sampled_scene_indices].mean(axis=1)
        output[metric] = {
            "difference": float(per_scene.mean()),
            "percentile_95_ci": [
                float(np.quantile(bootstrap, 0.025, method="linear")),
                float(np.quantile(bootstrap, 0.975, method="linear")),
            ],
        }
    return {
        "scene_count": len(scenes),
        "event_pair_count": len(pair_scenes),
        "repetitions": repetitions,
        "seed": seed,
        "method": "paired bootstrap over scenes; per-scene mean example difference",
        "difference_directions": {
            "accuracy": "joint minus audio-only",
            "nll": "audio-only minus joint",
            "brier": "audio-only minus joint",
        },
        "metrics": output,
    }


def _pair_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("each row needs a nonempty pair_id")
        groups[pair_id].append(index)
    return groups


def _swap_metrics(
    rows: list[dict[str, Any]],
    original_probabilities: np.ndarray,
    swapped_rows: list[dict[str, Any]],
    swapped_probabilities: np.ndarray,
) -> dict[str, float | int]:
    _assert_paired_rows(rows, swapped_rows, name="audio-swap")
    groups = _pair_indices(rows)
    if any(len(indices) != 2 for indices in groups.values()):
        raise ValueError("audio-swap audit requires exactly two variants per pair")
    original_targets = np.asarray([row["target_index"] for row in rows], dtype=np.int64)
    swapped_targets = np.asarray([row["target_index"] for row in swapped_rows], dtype=np.int64)
    swapped_predictions = swapped_probabilities.argmax(axis=1)
    original_predictions = original_probabilities.argmax(axis=1)
    donor_correct = []
    for indexes in groups.values():
        donor_correct.extend(
            [
                int(swapped_predictions[indexes[0]] == original_targets[indexes[1]]),
                int(swapped_predictions[indexes[1]] == original_targets[indexes[0]]),
            ]
        )
    return {
        "count": len(rows),
        "pair_count": len(groups),
        "swapped_accuracy_on_original_target": float(
            (swapped_predictions == swapped_targets).mean()
        ),
        "swapped_accuracy_on_opposite_donor_target": float(np.mean(donor_correct)),
        "prediction_flip_rate": float((original_predictions != swapped_predictions).mean()),
    }


def replay(run_dir: Path, *, repetitions: int = 5000, seed: int = 17) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    joint_rows, joint_hash = _load_jsonl(run_dir / "predictions-best-joint_lstm.jsonl")
    audio_rows, audio_hash = _load_jsonl(run_dir / "predictions-best-audio_only_lstm.jsonl")
    swap_rows, swap_hash = _load_jsonl(
        run_dir / "predictions-best-joint_lstm-audio-paired-swap.jsonl"
    )
    _assert_paired_rows(joint_rows, audio_rows, name="audio-only")
    joint_probabilities, joint_targets = _probability_matrix(joint_rows)
    audio_probabilities, audio_targets = _probability_matrix(audio_rows)
    swap_probabilities, _swap_targets = _probability_matrix(swap_rows)
    return {
        "mode": "saved-prediction replay only; no model, checkpoint, encoder, or media loaded",
        "count": len(joint_rows),
        "scene_count": len({row["scene_group_id"] for row in joint_rows}),
        "hashes": {
            "joint_predictions_sha256": joint_hash,
            "audio_only_predictions_sha256": audio_hash,
            "audio_swap_predictions_sha256": swap_hash,
        },
        "joint": _metrics(joint_probabilities, joint_targets),
        "audio_only": _metrics(audio_probabilities, audio_targets),
        "paired_scene_bootstrap_joint_vs_audio_only": _scene_bootstrap(
            joint_rows,
            joint_probabilities,
            joint_targets,
            audio_probabilities,
            audio_targets,
            repetitions=repetitions,
            seed=seed,
        ),
        "joint_audio_swap": _swap_metrics(
            joint_rows, joint_probabilities, swap_rows, swap_probabilities
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(replay(args.run_dir, repetitions=args.repetitions, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
